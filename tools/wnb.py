import wandb
from omegaconf import DictConfig, OmegaConf
import os

def setup_wandb(cfg: DictConfig):
    if cfg.wandb.dry_run:
        os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project_name, config=OmegaConf.to_container(cfg))

    if wandb.run.name is None:
        name = "test"
    else:
        name = wandb.run.name
    ckpt_root_dir = os.path.join(cfg.log.ckpt_loc, name)
    os.makedirs(ckpt_root_dir, exist_ok=True)
    return ckpt_root_dir