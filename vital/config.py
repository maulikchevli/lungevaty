from enum import Enum
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from pathlib import Path
from omegaconf import MISSING
from typing import List, Any

def load_config_store():
    configstore = ConfigStore.instance()
    configstore.store(name="base_config", node=Config)

@dataclass
class DataConfig:
    dataset_file: Path = MISSING
    dataset_file_100: Path = MISSING
    slice_thickness_threshold: float = 2.5
    max_followup: int =  6
    num_classes: int = 1
    assign_splits: bool = True
    use_thinnest_cut: bool = True
    corrupted_paths: Path =  "./files/corrupted_img_paths.pkl"
    google_splits_filename: Path = "./files/Shetty_et_al(Google)_data_splits.p"
    data_root: Path = MISSING
    monai_dict_train: Path = MISSING 
    monai_dict_dev: Path = MISSING
    monai_dict_test: Path = MISSING
    max_train_len: int = MISSING
    img_size: List[int] = MISSING

class LRScheduler(Enum):
    cawr = 0
    onecycle = 1
    cycle = 2
    none = 3

class LossFn(Enum):
    bce = 0
    mse = 1
    softlabel = 2

@dataclass
class TrainingConfig:
    dtype: str = 'bfloat16'
    use_amp: bool = False
    mask_ratio: float = 0.4
    seed: int = 0
    num_workers: int =  8
    dev_num_workers: int = 4
    test_num_workers: int = 1
    prefetch_factor: int = 2
    dev_prefetch_factor: int = 2
    train_persistent_workers: bool = True
    dev_persistent_workers: bool = True
    pin_memory: bool = False
    train_drop_last: bool = True
    dev_drop_last: bool = True
    shuffle: bool = True
    epochs: int = MISSING
    batch_size: int = MISSING
    accumulation_steps: int = 1
    to_checkpoint: bool = MISSING
    sampler: str = "weighted"
    underrepresented_weight: float = 14
    minority_samples_per_batch: int = 1
    freeze_encoder: bool = False
    freeze_mha: bool = False
    resume: bool = False

@dataclass
class OptimizerConfig:
    lr_scheduler: LRScheduler = MISSING
    peak_lr: float = MISSING
    init_lr: float = 1e-5 
    end_lr: float = 1e-6
    warmup_epochs: int = 5

class WeightingStrategy(Enum):
    heirarchical = 0
    flat = 1
    loss = 2

@dataclass
class WeightingConfig:
    use_sqrt: bool = MISSING
    weight_decay_factor: float = MISSING
    decay_epochs: int = MISSING
    strategy: WeightingStrategy = MISSING
    underrepresented_weight: float = MISSING

@dataclass
class LossConfig:
    sw: float = 1.0
    aw: float = 1.0
    loss_strategy: str = 'full'

class AttentionStrategy(Enum):
    joint = 0
    separate = 1

class AttentionBlocks(Enum):
    all = 0
    lastn = 1

class AttentionToken(Enum):
    cls = 0

@dataclass
class AttentionConfig:
    heads: int = 12
    use_fusion_layer: bool = False
    use_attention: bool = True
    use_cls: bool = True
    use_mean_token: bool = True
    use_attention_pooling: bool = True

@dataclass
class LoggingConfig:
    ckpt_loc: str = 'checkpoints'
    ckpt_load_loc: str = 'checkpoints'
    mae_ckpt_load: str = 'sybil-vit-last'
    mae_use_checkpoint: str = 'test'
    use_test_checkpoint: str = 'test'
    ckpt_best: str = 'sybil-vit'
    ckpt_last: str = 'sybil-vit-last'
    ckpt_load: str = 'sybil-vit-last'
    finetuned_use_checkpoint: str = MISSING
    finetuned_ckpt_load: str = MISSING
    continue_use_checkpoint: str = MISSING
    continue_log_ckpt_load: str = MISSING
    use_checkpoint: str = MISSING
    checkpoint_at_epoch: int = MISSING
    log_at_these_steps: int = MISSING
    log_scans_at_these_epochs: int = MISSING
    num_predictions: int = MISSING
    cancer_cases_to_log: int = 10
    laterality_cases_to_log: int = 10
    healthy_cases_to_log: int = 10
    pretrained_model_type: str = "pretrained"
    

@dataclass
class ModelConfig:
    patch_size: int = MISSING
    enc_dim: int = MISSING
    dec_dim: int = MISSING
    dropout_rate: float = MISSING
    enc_heads: int = 12
    enc_depth: int = 12
    dec_heads: int = MISSING
    dec_depth: int = MISSING
    in_chans: int = 1
    rng: int = 24
    fusion_layer: bool = False
    mlp_hidden_dim: int = 768
    sequential_blocks: int = 6
    sequential_heads: int = 8

@dataclass
class LongitudinalConfig:
    rnn_hidden_dim: int = 768
    rnn_cell: str = "simple"
    blocks: int = 5
    heads: int = 12
    bidirectional: bool = True
    dropout_rate: float = 0.2
    model: str = "rnn"

@dataclass
class WandBConfig:
    dry_run: bool = MISSING
    project_name: str = "vital"
    task: str = "Sybil Transformer"
    entity: str = "mri-ai-lab"

@dataclass
class TransformsConfig:
    train_tf: dict[str, Any] = field(default_factory=dict)
    dev_tf: dict[str, Any] = field(default_factory=dict)
    test_tf: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    wandb: WandBConfig = field(default_factory=WandBConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    longitudinal: LongitudinalConfig = field(default_factory=LongitudinalConfig)
    attention: AttentionConfig = field(default=AttentionConfig)
    training: TrainingConfig = field(default=TrainingConfig)
    optimizer: OptimizerConfig = field(default=OptimizerConfig)
    loss: LossConfig = field(default=LossConfig)
    log: LoggingConfig = field(default_factory=LoggingConfig)
    transform: TransformsConfig = field(default_factory=TransformsConfig)  