import os

import hydra

import wandb
import json

from vital.config import Config, load_config_store
from vital.sampler import DeterministicImbalancedSampler
from vital.transformations import make_transformations
from vital.lungevity import Lungevity
from vital.metrics import get_censoring_dist, compute_and_log_metrics_risk, log_targets
from tools.loop_conditions import to_log, to_save_checkpoint
from tools.checkpointing import save_checkpoint, load_checkpointed_state
from tools.wnb import setup_wandb

from monai.data import Dataset
from torch_scatter import scatter
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F

#------------------------------------------------------------------------#
# Use this to set a higher limit on open file descriptors to avoid "Too many open files" error with many workers
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
soft_limit = min(100000, rlimit[1])
resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, rlimit[1]))
#------------------------------------------------------------------------#

device = ( "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")



@hydra.main(version_base=None, config_path="./configs/", config_name="survival.yaml")
def main(cfg: DictConfig):
    ckpt_root_dir = setup_wandb(cfg)

    # Data
    with open(cfg.data.monai_dict_train) as fp:
        monai_dict_train = json.load(fp)
    with open(cfg.data.monai_dict_dev) as fp:
        monai_dict_dev = json.load(fp)
        
    train_censoring_distribution = get_censoring_dist(monai_dict_train)

    train_transforms = make_transformations(tf_dict=cfg.transform.train_tf)
    dev_transforms = make_transformations(tf_dict=cfg.transform.dev_tf)

    train_ds = Dataset(data=monai_dict_train, transform=train_transforms)
    dev_ds = Dataset(data=monai_dict_dev, transform=dev_transforms)

    sampler = get_sampler(cfg, train_ds, monai_dict_train)
    
    
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, 
                              shuffle=cfg.training.shuffle, 
                              num_workers=cfg.training.num_workers, prefetch_factor=cfg.training.prefetch_factor,
                              persistent_workers=True, pin_memory=False,
                              drop_last=True, sampler=sampler)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.training.batch_size, shuffle=False,
                        num_workers=cfg.training.dev_num_workers, prefetch_factor=cfg.training.prefetch_factor,
                        persistent_workers=True, pin_memory=False,
                        drop_last=True)

    # Model
    model = get_model(cfg)
    
    # Optimizer, Scheduler, and Scaler
    optimizer, scheduler, optimizer_phase2 = get_optimizer_scheduler(cfg, model, train_loader)
    
    scaler = torch.GradScaler(device=device, enabled=cfg.training.use_amp) 
    
    if cfg.training.resume == True:
        start_epoch = load_checkpointed_state(cfg.log.ckpt_load_loc, cfg.log.mae_use_checkpoint, device, model, optimizer, scheduler, scaler, cfg.paek_lr)
    else:
        start_epoch = 0
        checkpoint = torch.load(os.path.join(cfg.log.ckpt_load_loc, cfg.log.mae_use_checkpoint), map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)

    steps_per_epoch = len(sampler) // cfg.training.batch_size
    dev_steps_per_epoch = len(monai_dict_dev) // cfg.training.batch_size

    # # Init running value arrays
    probs = np.zeros((steps_per_epoch, cfg.training.batch_size, cfg.data.max_followup))
    golds = np.zeros((steps_per_epoch, cfg.training.batch_size))
    censors = np.zeros((steps_per_epoch, cfg.training.batch_size))

    dev_probs = np.zeros((dev_steps_per_epoch, cfg.training.batch_size, cfg.data.max_followup))
    dev_golds = np.zeros((dev_steps_per_epoch, cfg.training.batch_size))
    dev_censors = np.zeros((dev_steps_per_epoch, cfg.training.batch_size))
    ckpt_metric = 0
    save_step = 0
    
    
    model.train()
    
    # Set default freeze epochs if not specified in config
    freeze_encoder_epochs = getattr(cfg.training, 'freeze_encoder_epochs', 0)
    
    # Track phase switching for two-phase OneCycleLR
    phase_switched = False
    
    # Init storage variables (pre-allocate reused arrays)
    probs = np.zeros((steps_per_epoch, cfg.training.batch_size, cfg.data.max_followup))
    golds = np.zeros((steps_per_epoch, cfg.training.batch_size))
    censors = np.zeros((steps_per_epoch, cfg.training.batch_size))
    
    dev_probs = np.zeros((dev_steps_per_epoch, cfg.training.batch_size, cfg.data.max_followup))
    dev_golds = np.zeros((dev_steps_per_epoch, cfg.training.batch_size))
    dev_censors = np.zeros((dev_steps_per_epoch, cfg.training.batch_size))
    
    for epoch in range(start_epoch, cfg.training.epochs):
        # Handle phase switching for two-phase OneCycleLR
        optimizer, scheduler, phase_switched = check_training_phase_switch(
            cfg, epoch, freeze_encoder_epochs, phase_switched, optimizer_phase2, train_loader, optimizer, scheduler
        )
        
        # Freeze encoder for first x epochs
        if epoch < freeze_encoder_epochs:
            set_encoder_frozen(model, frozen=True)
        else:
            set_encoder_frozen(model, frozen=False)
            
        # Run Training Epoch (reuses memory buffers)
        run_train_epoch(cfg, model, train_loader, optimizer, scheduler, epoch, steps_per_epoch, train_censoring_distribution,
                        probs, golds, censors)

        # Run Validation Epoch (reuses memory buffers)
        survival_metrics = run_validation_epoch(cfg, model, dev_loader, dev_steps_per_epoch, train_censoring_distribution,
                                               dev_probs, dev_golds, dev_censors)

        # Step scheduler after each epoch for non-OneCycleLR schedulers
        if cfg.optimizer.lr_scheduler != 'onecycle':
            scheduler.step()

        # Checkpointing
        ckpt_metric, save_step = save_model_checkpoint(
            cfg, epoch, ckpt_root_dir, model, optimizer, scheduler, 
            scaler, ckpt_metric, save_step, survival_metrics
        )

    return



def get_model(cfg):
    model = Lungevity(
        transformer=cfg.model.transformer,
        patch_size=cfg.model.patch_size,
        grid_size=[
            int(cfg.data.img_size[0]/cfg.model.patch_size[0]), 
            int(cfg.data.img_size[1]/cfg.model.patch_size[1]), 
            int(cfg.data.img_size[2]/cfg.model.patch_size[2])
            ],
        enc_dim=cfg.model.enc_dim,
        enc_blocks=cfg.model.enc_depth,
        enc_heads=cfg.model.enc_heads,
        dropout_rate=cfg.model.dropout_rate,
        num_reg_tokens=cfg.model.num_reg_tokens,
        use_cls=cfg.model.use_cls,
        hidden_dim=cfg.model.enc_dim,
        max_followup=cfg.data.max_followup,
        fusion_layer=cfg.model.fusion_layer,
        guided_attention_heads=cfg.model.guided_attention_heads,
        use_mean_token=cfg.model.use_mean_token,
        task="survival",
        num_classes=cfg.data.max_followup,
    )
    
    model = model.to(device)
    model = torch.compile(model, mode="max-autotune", fullgraph=True)
    model = model.to(dtype=torch.bfloat16)
    return model


def get_optimizer_scheduler(cfg, model, train_loader):
    freeze_encoder_epochs = getattr(cfg.training, 'freeze_encoder_epochs', 0)
    optimizer_phase2 = None
    
    # Phase 1: Create optimizer and scheduler for classifier-only training
    if cfg.optimizer.lr_scheduler == 'onecycle' and freeze_encoder_epochs > 0:
        # Phase 1: Only classifier parameters (encoder will be frozen)
        classifier_params = []
        for name, param in model.named_parameters():
            if 'encoder' not in name:  # All non-encoder parameters
                classifier_params.append(param)
        
        optimizer = torch.optim.AdamW(classifier_params, 
                                           lr=(cfg.optimizer.peak_lr/cfg.optimizer.div_factor), 
                                           weight_decay=cfg.optimizer.weight_decay)
        
        # OneCycleLR for phase 1 (classifier only)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.optimizer.peak_lr,
            epochs=freeze_encoder_epochs, steps_per_epoch=len(train_loader),
            pct_start=0.1,  # 10% warmup as requested
            div_factor=cfg.optimizer.div_factor, 
            final_div_factor=cfg.optimizer.final_div_factor
        )
        
        # Phase 2: All parameters for remaining epochs
        optimizer_phase2 = torch.optim.AdamW(model.parameters(), 
                                           lr=(cfg.optimizer.peak_lr/cfg.optimizer.div_factor), 
                                           weight_decay=cfg.optimizer.weight_decay)
        
        # Note: scheduler_phase2 will be created fresh when switching to phase 2
        # to ensure step counter starts at 0
        
    elif cfg.optimizer.lr_scheduler == 'onecycle':
        # Original single-phase OneCycleLR
        optimizer = torch.optim.AdamW(model.parameters(), lr=(cfg.optimizer.peak_lr/cfg.optimizer.div_factor), weight_decay=cfg.optimizer.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.optimizer.peak_lr,
                                                    epochs=cfg.training.epochs, steps_per_epoch=len(train_loader),
                                                    pct_start=cfg.optimizer.pct_start, div_factor=cfg.optimizer.div_factor, final_div_factor=cfg.optimizer.final_div_factor)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.peak_lr, weight_decay=cfg.optimizer.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=1.0
        )
    
    return optimizer, scheduler, optimizer_phase2

def check_training_phase_switch(cfg, epoch, freeze_encoder_epochs, phase_switched, optimizer_phase2, train_loader, optimizer, scheduler):
    # Handle phase switching for two-phase OneCycleLR
    if (cfg.optimizer.lr_scheduler == 'onecycle' and freeze_encoder_epochs > 0 and 
        epoch == freeze_encoder_epochs and not phase_switched):
        print(f"Switching to Phase 2: Unfreezing encoder and starting new OneCycleLR")
        
        # Create fresh scheduler for phase 2 to ensure step counter starts at 0
        remaining_epochs = cfg.training.epochs - freeze_encoder_epochs 
        
        # Use optimizer_phase2 which handles all parameters
        new_optimizer = optimizer_phase2
        
        # New scheduler for phase 2
        new_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            new_optimizer, max_lr=cfg.optimizer.peak_lr,
            epochs=remaining_epochs, steps_per_epoch=len(train_loader),
            pct_start=cfg.optimizer.pct_start,  # This will start warmup from beginning of phase 2
            div_factor=cfg.optimizer.div_factor, 
            final_div_factor=cfg.optimizer.final_div_factor
        )
        phase_switched = True
        
        print(f"Phase 2: New OneCycleLR will warm up for {cfg.optimizer.pct_start*100:.1f}% of {remaining_epochs} epochs")
        return new_optimizer, new_scheduler, phase_switched
    
    return optimizer, scheduler, phase_switched

def run_train_epoch(cfg, model, train_loader, optimizer, scheduler, epoch, steps_per_epoch, train_censoring_distribution,
                    probs, golds, censors):
    running_loss, running_survival_loss, running_annotation_loss = 0, 0, 0
    
    # Reuse memory by filling with zeros instead of reallocating
    probs.fill(0)
    golds.fill(0)
    censors.fill(0)
    
    for step, batch in enumerate(train_loader):
        images, annotations, y_seq, y_mask = batch['image'], batch['annotation'], batch['y_seq'], batch['y_mask']
        laterality, laterality_label, lobes, sides = batch['laterality'], batch['laterality_label'], batch['lobes'], batch['sides']
        
        loss, segregated_loss, _probs = train_step(
            model, images, annotations, laterality, laterality_label, lobes, sides,
            y_seq, y_mask, 
            (cfg.loss.sw, cfg.loss.aw), optimizer, cfg.model.patch_size, loss_strategy=cfg.loss.loss_strategy
        )
        
        # Step scheduler after each batch for OneCycleLR
        if cfg.optimizer.lr_scheduler == 'onecycle':
            scheduler.step()
        
        running_loss += loss
        running_survival_loss += segregated_loss[0]
        running_annotation_loss += segregated_loss[1]
        probs[step, :, :] = np.array(_probs)
        golds[step, :] = batch['y'].numpy()
        censors[step, :] = batch['time_at_event'].numpy()
        
        if to_log(step, steps_per_epoch, cfg.log.log_at_these_steps):
            print(f"Epoch {epoch}. Step {step}/{steps_per_epoch}: Loss {loss}")
            wandb.log({"train/loss_step": loss})
            wandb.log({"train/survival_loss": segregated_loss[0]})
            wandb.log({"train/annotation_loss": segregated_loss[1]})
            wandb.log({"lr": optimizer.param_groups[0]['lr']})

    wandb.log({"train/loss": running_loss / steps_per_epoch})
    compute_and_log_metrics_risk(censors, probs, golds, train_censoring_distribution, cfg.data.max_followup, mode="train")
    log_targets(probs, golds, censors, cfg.log.num_predictions, "train")

def run_validation_epoch(cfg, model, dev_loader, dev_steps_per_epoch, train_censoring_distribution,
                         dev_probs, dev_golds, dev_censors):
    running_loss, running_survival_loss, running_annotation_loss = 0, 0, 0
    
    # Reuse memory
    dev_probs.fill(0)
    dev_golds.fill(0)
    dev_censors.fill(0)
    
    model.eval()
    for step, batch in enumerate(dev_loader):
        images, annotations, y_seq, y_mask = batch['image'], batch['annotation'], batch['y_seq'], batch['y_mask']
        
        loss, segregated_loss, _probs = dev_step(
            model, images, annotations, y_seq, y_mask, 
            (cfg.loss.sw, cfg.loss.aw), cfg.model.patch_size
        )
        running_loss += loss
        running_survival_loss += segregated_loss[0]
        running_annotation_loss += segregated_loss[1]
        dev_probs[step, :, :] = np.array(_probs)
        dev_golds[step, :] = batch['y'].numpy()
        dev_censors[step, :] = batch['time_at_event'].numpy()

    wandb.log({"dev/loss": running_loss / dev_steps_per_epoch})
    survival_metrics, _ = compute_and_log_metrics_risk(dev_censors, dev_probs, dev_golds, train_censoring_distribution, cfg.data.max_followup, mode="dev")
    log_targets(dev_probs, dev_golds, dev_censors, cfg.log.num_predictions, "dev")
    return survival_metrics

def save_model_checkpoint(cfg, epoch, ckpt_root_dir, model, optimizer, scheduler, scaler, ckpt_metric, save_step, survival_metrics):
    if to_save_checkpoint(epoch, cfg.training.epochs, cfg.log.checkpoint_at_epoch, cfg.training.to_checkpoint):
        print("Saving checkpoint")
        sum_metric = survival_metrics['dev/1_year_auc'] + survival_metrics['dev/2_year_auc'] + survival_metrics['dev/3_year_auc'] \
            + survival_metrics['dev/4_year_auc'] + survival_metrics['dev/5_year_auc'] + survival_metrics['dev/6_year_auc'] 
        
        if sum_metric >= ckpt_metric:
            file_name = f"best.pt"
            ckpt_metric = sum_metric
            save_checkpoint(ckpt_root_dir, file_name, model, epoch, optimizer, scheduler, scaler, ckpt_metric, save_step)
            save_step += 1
            
        file_name = f"last.pt"
        save_checkpoint(ckpt_root_dir, file_name, model, epoch, optimizer, scheduler, scaler, sum_metric, save_step)
        
    return ckpt_metric, save_step


def set_encoder_frozen(model, frozen=True):
    if hasattr(model, 'encoder'):
        for param in model.encoder.parameters():
            param.requires_grad = not frozen
        print(f"Encoder {'frozen' if frozen else 'unfrozen'}")


def train_step(
        model,
        images: torch.Tensor,
        annotations: torch.Tensor,
        laterality: torch.Tensor,
        laterality_label: torch.Tensor,
        lobes: torch.Tensor,
        sides: torch.Tensor,
        y_seq: torch.Tensor,
        y_mask: torch.Tensor,
        loss_weights: tuple,
        optimizer,
        patch_size,
        loss_strategy='full' # 'full' 'lobe' 'side' 'off'
):
    # Move tensors to device and correct dtype
    laterality_label = laterality_label.to(device)
    lobes = lobes.to(device)
    sides = sides.to(device)
    laterality = laterality.to(device, dtype=torch.int64)
    
    optimizer.zero_grad(set_to_none=True)  # Faster than default zero_grad()

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
        loss, (survival_loss, annotation_loss, probs) = loss_fn(
            model, images, annotations, laterality, laterality_label, lobes, sides,
            patch_size, y_seq, y_mask, 
            loss_weights[0], loss_weights[1], loss_strategy=loss_strategy
        )
    
    loss.backward()
    optimizer.step()
    
    return loss.item(), (survival_loss.item(), annotation_loss.item()), probs.detach().cpu()


def dev_step(
        model,
        images: torch.Tensor,
        annotations: torch.Tensor,
        y_seq: torch.Tensor,
        y_mask: torch.Tensor,
        loss_weights: tuple,
        patch_size,
):
    # Move tensors to device and correct dtype
    images = images.to(device, dtype=torch.bfloat16)
    annotations = annotations.to(device, dtype=torch.bfloat16)
    y_seq = y_seq.to(device, dtype=torch.bfloat16)
    y_mask = y_mask.to(device, dtype=torch.bfloat16)
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            loss, (survival_loss, annotation_loss, probs) = dev_loss_fn(
                model, images, annotations, patch_size, y_seq, y_mask, 
                loss_weights[0], loss_weights[1]
            )
    
    return loss.item(), (survival_loss.item(), annotation_loss.item()), probs.detach().cpu()


def loss_fn(model, images, annotations, laterality, laterality_label, lobes, is_side,
            patch_size, y_seq, y_mask, sw, aw, loss_strategy=[1, 1]):
    # Survival loss (always computed)
    n_year_logits, attn_weights, _ = model(images)
    survival_loss = F.binary_cross_entropy_with_logits(n_year_logits, y_seq, reduction='none') * y_mask
    survival_loss = survival_loss.sum() / y_mask.sum()

    # Early return if no annotation loss needed
    if loss_strategy == [0, 0]:
        annotation_loss = torch.tensor(0.0, device=survival_loss.device)
        return (sw * survival_loss + aw * annotation_loss), (survival_loss, annotation_loss, torch.sigmoid(n_year_logits))

    # Process attention weights (needed for all annotation strategies)
    attn_weights = attn_weights.mean(dim=1)  # Average attention weights across heads
    attn_weights = attn_weights.mean(dim=1)  # Average attention weights across tokens
    
    # Initialize annotation_loss
    annotation_loss = torch.tensor(0.0, device=survival_loss.device)
    
    # Compute base annotation loss for 'full' strategy
    if loss_strategy[0] == 1:
        attn_scores = F.log_softmax(attn_weights, dim=-1)
        annotations_mask = (annotations > 0).any(dim=(1, 2))
        mask_area = annotations.sum(dim=(-1, -2))
        mask_area = torch.where(mask_area == 0, 1, mask_area)
        annotations_gold = annotations.sum(dim=-1) / mask_area[:, None]

        base_annotation_loss = F.kl_div(attn_scores, annotations_gold, reduction='none') * annotations_mask[:, None]
        num_annotations = torch.where(annotations_mask.sum() == 0, 1, annotations_mask.sum())
        annotation_loss = base_annotation_loss.sum() / num_annotations

    # Compute lobe and/or side losses based on strategy
    if loss_strategy[1] == 1:
        # Setup for laterality-based losses
        id = laterality
        sides = torch.where(id == 0, 0, torch.where(id < 3, 1, 2))
        
        # Lobe loss
        predictions = scatter(attn_weights, id, dim=1)[:, 1:]
        labels = torch.where(~lobes, 0, laterality_label)
        lobe_loss = F.cross_entropy(predictions, labels, reduction='none') * lobes
        num_lobes = lobes.sum()
        num_lobes = torch.where(num_lobes == 0, 1, num_lobes)
        lobe_loss = lobe_loss.sum() / num_lobes

        # Side loss  
        side_predictions = scatter(attn_weights, sides, dim=1)[:, 1:]
        labels = torch.where(~is_side, 0, laterality_label)
        side_loss = F.cross_entropy(side_predictions, labels, reduction='none') * (is_side)
        num_sides = (is_side).sum(); num_sides = torch.where(num_sides == 0, 1, num_sides)
        side_loss = side_loss.sum() / num_sides

        annotation_loss += lobe_loss + side_loss
            
    return (sw * survival_loss + aw * annotation_loss), (survival_loss, annotation_loss, torch.sigmoid(n_year_logits))


def dev_loss_fn(model, images, annotations, patch_size, y_seq, y_mask, sw, aw):
    # Survival loss
    n_year_logits, _, _ = model(images)
    survival_loss = F.binary_cross_entropy_with_logits(n_year_logits, y_seq, reduction='none') * y_mask
    survival_loss = survival_loss.sum() / y_mask.sum()

    # Annotation loss
    annotation_loss = torch.tensor(0.0)

    return (sw * survival_loss + aw * annotation_loss), (survival_loss, annotation_loss, torch.sigmoid(n_year_logits))

def get_sampler(cfg, train_ds, monai_dict_train):
    if cfg.training.sampler == "weighted":
        dataset_gnr = torch.Generator(device="cpu")
        dataset_gnr.manual_seed(0)
        labels = [sample['y'] for sample in monai_dict_train]
        _, counts = np.unique(labels, return_counts=True)
        y_weight = np.array([1, counts[0] / counts[1]], dtype=np.float16)
        samples_weights = y_weight[np.array(labels)]
        sampler = WeightedRandomSampler(
            weights=samples_weights,
            num_samples=len(samples_weights),
            replacement=True,
            generator=dataset_gnr
        )
    else:
        sampler_gnr = torch.Generator(device="cpu")
        sampler = DeterministicImbalancedSampler(
            dataset=train_ds,
            batch_size=cfg.training.batch_size,
            minority_class_label=1, 
            minority_samples_per_batch=cfg.training.minority_samples_per_batch,
            label_key="y",
            generator=sampler_gnr,
            drop_last=True
        )
    return sampler

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main() 