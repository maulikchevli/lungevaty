from utils.visualization import visualized_images
import wandb
import numpy as np
import torch
from functools import partial
import os
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from PIL import Image, ImageSequence
import matplotlib.cm as cm

cancer_dict = {
    0: "no_cancer",
    1: "cancer",
    2: 'future_cancer'
}


def compute_and_log_losses(step, epoch, total_steps, total_epochs, running_loss, running_segregated_loss, mode="train"):
    segregated_loss = unravel_running_metric(running_segregated_loss, np.average)
    loss = np.average(running_loss)
    wandb.log({f"{mode}/loss": loss})
    wandb.log(segregated_loss)
    print(f"{mode}. Epoch {epoch} / {total_epochs}, step {step} / {total_steps}. Loss {loss}")
    return loss

def unravel_running_metric(running_metric, aggregate_fn):
    return {key: aggregate_fn([entry[key] for entry in running_metric]) for key in running_metric[0]}

def log_gradients(model, step, epoch):
    total_norm = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2).item()
        total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2)
    wandb.log({"grad_norm": total_norm})
    return

def log_learning_rate(learning_rate):
    wandb.log({"lr": learning_rate})

def log_images_3d(  image, recon, masked_indices, patch_size,
                    epoch, pid, timepoint, cancer_label,
                    batch_index, mode='train'):

    scan_video, recon_video, masked_video = visualized_images(image[batch_index], recon[batch_index], masked_indices[batch_index],
                                    original_image_shape=image[batch_index].shape,
                                    patch_size=patch_size)
    scan_video = torch.clip(scan_video, min=-1, max=1)
    recon_video = torch.clip(recon_video, min=-1, max=1)
    masked_video = torch.clip(masked_video, min=-1, max=1)
    scaled_scan_video = ((scan_video + 1) * 127.5).to(torch.uint8)
    scaled_recon_video = ((recon_video + 1) * 127.5).to(torch.uint8)
    scaled_masked_video = ((masked_video + 1) * 127.5).to(torch.uint8)
    
    #cmap = cm.get_cmap('jet')
    # if annotation[0]:
    #     rgb_annotation = cmap(annotation_video.squeeze(0))

    #     scaled_rgb_annotation = torch.round(torch.tensor(rgb_annotation * 255)).to(torch.uint8) #this does not work with wandb logging

    #     rgb_annotation = scaled_rgb_annotation[:,:,:,:3]

    #     blended_annotation_recon = combine_videos(scaled_recon_video.permute(1,0,2,3).repeat(1,3,1,1), rgb_annotation.permute(2,3,0,1), alpha=0.85)
    #     blended_annotation_scan = combine_videos(scaled_scan_video.permute(1,0,2,3).repeat(1,3,1,1), rgb_annotation.permute(2,3,0,1), alpha=0.85)
    #     blended_annotation_masked = combine_videos(scaled_masked_video.permute(1,0,2,3).repeat(1,3,1,1), rgb_annotation.permute(2,3,0,1), alpha=0.85)

    # cancer visible, fu, no cancer

    # if annotation[0]:
    #     wandb.log({f"{mode}_media/Epoch_{epoch}/{cancer_dict[int(cancer_label[batch_index])]}/{int(pid[batch_index])}_T{int(timepoint[batch_index])}": wandb.Video(np.array(blended_annotation_scan), fps=10, format="gif",  caption=f"scan")})
    #     wandb.log({f"{mode}_media/Epoch_{epoch}/{cancer_dict[int(cancer_label[batch_index])]}/{int(pid[batch_index])}_T{int(timepoint[batch_index])}": wandb.Video(np.array(blended_annotation_recon), fps=10, format="gif", caption=f"recon")})
    #     wandb.log({f"{mode}_media/Epoch_{epoch}/{cancer_dict[int(cancer_label[batch_index])]}/{int(pid[batch_index])}_T{int(timepoint[batch_index])}": wandb.Video(np.array(blended_annotation_masked), fps=10, format="gif",  caption=f"masked")})
    wandb.log({f"{mode}_media/Epoch_{epoch}/{cancer_dict[int(cancer_label[batch_index])]}/{int(pid[batch_index])}_T{int(timepoint[batch_index])}": wandb.Video(np.array(scaled_scan_video.permute(1,0,2,3).repeat(1,3,1,1)), fps=10, format="gif",  caption=f"scan")})
    wandb.log({f"{mode}_media/Epoch_{epoch}/{cancer_dict[int(cancer_label[batch_index])]}/{int(pid[batch_index])}_T{int(timepoint[batch_index])}": wandb.Video(np.array(scaled_recon_video.permute(1,0,2,3).repeat(1,3,1,1)), fps=10, format="gif",  caption=f"recon")})
    wandb.log({f"{mode}_media/Epoch_{epoch}/{cancer_dict[int(cancer_label[batch_index])]}/{int(pid[batch_index])}_T{int(timepoint[batch_index])}": wandb.Video(np.array(scaled_masked_video.permute(1,0,2,3).repeat(1,3,1,1)), fps=10, format="gif",  caption=f"masked")})

    #np.save(os.path.join(vis_folder, f"attention_epoch{epoch}_step{step}.npy"), attention_video.detach().numpy())
    #np.save(os.path.join(vis_folder, f"scan_epoch{epoch}_step{step}.npy"), scan_video.detach().numpy())

def compute_and_log_attention_stub(image, attn_vector, lung_hull_region, 
                               original_shape, patch_size,
                               step, epoch):
    pass


def combine_videos(scan, attention, alpha=0.3):
    blended = (alpha * scan + (1 - alpha) * attention).to(torch.uint8)
    return blended



