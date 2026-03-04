import os
import torch 
from collections import OrderedDict


def load_checkpoint(mngr):
    try:
        state = mngr.restore(mngr.latest_step())
        start_epoch = mngr.latest_step() + 1
    except FileNotFoundError as e:
        start_epoch = 0
        state = None

    return start_epoch, state


def load_checkpointed_state(loc, ckpt_name, device, model, optimizer, scheduler, scaler, new_learning_rate):
    loc = os.path.join(loc, ckpt_name)
    if not os.path.exists(loc):
        return 0
    else:
        print("Resuming from checkpoint")
    checkpoint = torch.load(loc, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    if scaler is not None and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
    
    # Change the learning rate if a new one is provided
    if new_learning_rate is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_learning_rate
    
    return checkpoint['epochs'] + 1

def save_checkpoint(loc, file_name, model, epoch, optimizer, scheduler, scaler, ckpt_metric, step):
    checkpoint = {
        "epochs": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ckpt_metric": ckpt_metric,
        "save_step": step + 1,
    }

    with open(os.path.join(loc, file_name), "wb") as fp:
        torch.save(checkpoint, fp)


def _normalize_state_dict_for_compile_mismatch(state_dict, model_state_dict):
    """Align checkpoint keys regardless of torch.compile wrapping."""
    model_has_prefix = any(k.startswith("_orig_mod.") for k in model_state_dict.keys())
    ckpt_has_prefix = any(k.startswith("_orig_mod.") for k in state_dict.keys())

    if model_has_prefix == ckpt_has_prefix:
        return state_dict

    normalized = OrderedDict()
    if ckpt_has_prefix and not model_has_prefix:
        prefix = "_orig_mod."
        for key, val in state_dict.items():
            new_key = key[len(prefix):] if key.startswith(prefix) else key
            normalized[new_key] = val
    elif model_has_prefix and not ckpt_has_prefix:
        for key, val in state_dict.items():
            new_key = key if key.startswith("_orig_mod.") else f"_orig_mod.{key}"
            normalized[new_key] = val
    else:
        normalized = state_dict
    return normalized


def normalize_state_dict_for_compile(state_dict, model):
    """Public wrapper to normalize checkpoint keys against a model."""
    return _normalize_state_dict_for_compile_mismatch(state_dict, model.state_dict())