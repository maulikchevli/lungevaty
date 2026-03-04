"""Standalone DICOM inference script for the Lungevity survival model.

Runs the full DICOM pipeline (orient → resample → HU window → lung-mask crop → resize)
on a dataset described by a JSON manifest, then writes a per-patient results table.

Lung masks are generated on-the-fly with the R231 lungmask model and cached to
`testing.mask_dir`.  If a mask already exists it is reused.

Usage
-----
Minimal (masks generated automatically):

    uv run dcm_inference.py \\
        data.monai_dict_dicom=/path/to/dicom_manifest.json \\
        testing.use_checkpoint=best.pt \\
        log.ckpt_loc=/path/to/checkpoints

With pre-computed masks in a specific directory:

    uv run dcm_inference.py \\
        data.monai_dict_dicom=/path/to/dicom_manifest.json \\
        testing.mask_dir=/path/to/masks \\
        testing.use_checkpoint=best.pt \\
        log.ckpt_loc=/path/to/checkpoints

Input JSON format
-----------------
Each entry must contain at least:
    "image"  : path to a DICOM series directory (folder containing *.dcm files)

Optional but used when present:
    "mask"         : path to a pre-computed NIfTI lung mask (.nii.gz)
                     — if provided AND the file exists it is used directly,
                       skipping on-the-fly generation for that entry
    "pid"          : patient identifier used for mask caching and the results table
    "series"       : series identifier used for deduplication and reporting
    "y"            : binary cancer label  (0 / 1)
    "time_at_event": follow-up time in years
    "y_seq"        : per-year label sequence  (list of length max_followup)
    "y_mask"       : per-year validity mask   (list of length max_followup)

Outputs (written to `testing.output_dir`)
-----------------------------------------
    results.csv   — one row per patient with PID, label, time, per-year probabilities
    results.json  — same data in JSON format
"""

import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import SimpleITK as sitk
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import tqdm
from monai.data import Dataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from lungmask import LMInferer
from vital.lungevity import Lungevity
from vital.transformations import make_transformations
from tools.checkpointing import normalize_state_dict_for_compile

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_for_filename(value: str) -> str:
    safe = [c if c.isalnum() or c in {"-", "_", "."} else "_" for c in value]
    return "".join(safe).strip("._") or "sample"


def _load_json(path: str) -> List[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p}")
    return json.loads(p.read_text())


def _filter_existing(entries: List[dict], required_keys: List[str]) -> List[dict]:
    out = []
    for entry in entries:
        if all(entry.get(k) and Path(entry[k]).exists() for k in required_keys):
            out.append(entry)
    return out


# ---------------------------------------------------------------------------
# On-the-fly lung mask generation
# ---------------------------------------------------------------------------

def _ensure_masks(entries: List[dict], cfg: DictConfig) -> List[dict]:
    """Populate / verify the 'mask' key for every DICOM entry.

    Priority:
      1. Entry already has a 'mask' key pointing to an existing file → keep it.
      2. Mask file already cached in mask_dir → reuse it.
      3. Generate with R231 lungmask, save to mask_dir.
    """
    mask_root = Path(cfg.testing.get("mask_dir", "./outputs/dcm_inference/masks"))
    mask_root.mkdir(parents=True, exist_ok=True)

    force_cpu = device != "cuda"
    inferer: Optional[LMInferer] = None  # lazy init — only needed if masks are missing

    updated = []
    for idx, entry in enumerate(tqdm.tqdm(entries, desc="Checking / generating masks")):
        dicom_dir = Path(entry.get("image", ""))
        if not dicom_dir.exists():
            print(f"  [mask] DICOM dir missing, skipping idx={idx}: {dicom_dir}")
            continue

        # --- priority 1: explicit mask in the entry ---
        existing_mask = entry.get("mask", "")
        if existing_mask and Path(existing_mask).exists():
            updated.append(dict(entry))
            continue

        # --- priority 2 / 3: cache path ---
        pid = str(entry.get("pid") or entry.get("patient_id") or "unknown")
        series_uid = entry.get("series") or dicom_dir.name
        safe_name = _sanitize_for_filename(str(series_uid))
        mask_path = mask_root / pid / f"{safe_name}.nii.gz"
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        if not mask_path.exists():
            if inferer is None:
                print("[mask] Initialising R231 lungmask model...")
                inferer = LMInferer(modelname="R231", force_cpu=force_cpu, tqdm_disable=True)
            try:
                dicom_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir))
                if not dicom_names:
                    print(f"  [mask] No DICOM files in {dicom_dir}, skipping.")
                    continue
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(dicom_names)
                sitk_image = reader.Execute()

                segmentation = inferer.apply(sitk_image)          # (Z, H, W)  values 0/1/2
                binary_mask = (segmentation > 0).astype(np.uint8) # binarise left+right lung

                mask_sitk = sitk.GetImageFromArray(binary_mask)
                mask_sitk.CopyInformation(sitk_image)
                sitk.WriteImage(mask_sitk, str(mask_path))
                print(f"  [mask] Saved {mask_path}")
            except Exception as exc:
                print(f"  [mask] Failed for idx={idx} ({series_uid}): {exc}")
                continue
        else:
            print(f"  [mask] Cached → {mask_path}")

        new_entry = dict(entry)
        new_entry["mask"] = str(mask_path)
        updated.append(new_entry)

    return updated


# ---------------------------------------------------------------------------
# MONAI transform pipeline  (DICOM only)
# ---------------------------------------------------------------------------

def _dicom_transforms(cfg: DictConfig) -> OrderedDict:
    spatial_size = cfg.data.img_size
    device_str = "cuda" if device == "cuda" else "cpu"
    dtype = cfg.training.get("dtype", "bfloat16")

    pp = cfg.get("preprocess", {})
    axcodes     = pp.get("axcodes", "PLS")
    pixdim      = list(pp.get("pixdim", [1.4, 1.4, 2.5]))
    hu_min      = pp.get("hu_min", -1350)
    hu_max      = pp.get("hu_max", 150)
    crop_margin = list(pp.get("crop_margin", [2, 2, 1]))

    return OrderedDict([
        ("LoadImaged",         {"keys": ["image", "mask", "annotation"], "allow_missing_keys": True}),
        ("EnsureChannelFirstd", {"keys": ["image", "mask", "annotation"], "allow_missing_keys": True}),
        ("Orientationd",       {"keys": ["image", "mask", "annotation"], "axcodes": axcodes, "allow_missing_keys": True}),
        ("Spacingd",           {
            "keys": ["image", "mask", "annotation"],
            "pixdim": pixdim,
            "mode": ("bilinear", "nearest", "nearest"),
            "allow_missing_keys": True,
        }),
        ("ScaleIntensityRanged", {
            "keys": ["image"],
            "a_min": hu_min, "a_max": hu_max,
            "b_min": -1.0,   "b_max": 1.0,
            "clip": True,
        }),
        ("CropForegroundd", {
            "keys": ["image", "mask", "annotation"],
            "source_key": "mask",
            "margin": crop_margin,
            "mode": "constant",
            "allow_smaller": False,
            "allow_missing_keys": True,
        }),
        ("Resized", {
            "keys": ["image", "mask", "annotation"],
            "mode": ["bilinear", "nearest", "nearest"],
            "spatial_size": spatial_size,
            "allow_missing_keys": True,
        }),
        ("ToDeviced", {
            "keys": ["image", "annotation", "mask"],
            "allow_missing_keys": True,
            "device": device_str,
        }),
        ("Permuted_our", {"keys": ["image", "annotation", "mask"]}),
        ("ToTensord", {
            "keys": ["y_mask", "y_seq"],
            "allow_missing_keys": True,
            "track_meta": False,
            "device": device_str,
            "dtype": dtype,
        }),
    ])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _build_model(cfg: DictConfig) -> Lungevity:
    grid_size = [
        int(cfg.data.img_size[0] / cfg.model.patch_size[0]),
        int(cfg.data.img_size[1] / cfg.model.patch_size[1]),
        int(cfg.data.img_size[2] / cfg.model.patch_size[2]),
    ]
    model = Lungevity(
        transformer=cfg.model.transformer,
        patch_size=cfg.model.patch_size,
        grid_size=grid_size,
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
    )
    model = model.to(device)

    ckpt_path = Path(cfg.log.ckpt_loc) / cfg.testing.use_checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), weights_only=False, map_location=device)
    aligned_state = normalize_state_dict_for_compile(checkpoint['model'], model)
    model.load_state_dict(aligned_state, strict=True)
    print(f"Loaded checkpoint: {ckpt_path}")
    model = torch.compile(model, mode="max-autotune")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_inference(
    model: Lungevity,
    loader: DataLoader,
    cfg: DictConfig,
    entries: List[dict],
) -> List[dict]:
    """Run inference and return one result dict per sample."""
    model.eval()
    results = []
    max_followup = cfg.data.max_followup

    for batch_idx, batch in enumerate(tqdm.tqdm(loader, desc="Inference")):
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=cfg.training.use_amp):
                image = batch["image"].to(device)

                # y_seq / y_mask may be absent for datasets without survival labels
                y_seq  = batch.get("y_seq",  torch.zeros(image.shape[0], max_followup)).to(device)
                y_mask = batch.get("y_mask", torch.zeros(image.shape[0], max_followup)).to(device)

                n_year_logits, _, _ = model(image, return_attention=False)
                probs = torch.sigmoid(n_year_logits).detach().cpu().float()

        batch_size = image.shape[0]
        for b in range(batch_size):
            global_idx = batch_idx * loader.batch_size + b
            entry = entries[global_idx] if global_idx < len(entries) else {}

            pid = str(entry.get("pid") or entry.get("patient_id") or entry.get("series") or f"sample_{global_idx:04d}")
            label = entry.get("y", None)
            time_at_event = entry.get("time_at_event", None)
            series = entry.get("series", pid)
            y_seq_entry = entry.get("y_seq", None)
            y_mask_entry = entry.get("y_mask", None)

            prob_list = probs[b].tolist()

            results.append({
                "pid": pid,
                "series": series,
                "label": label,
                "time_at_event": time_at_event,
                "y_seq": y_seq_entry,
                "y_mask": y_mask_entry,
                "probabilities": prob_list,
            })

    return results


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------

def _print_summary(results: List[dict], max_followup: int) -> None:
    year_headers = [f"yr{y+1}" for y in range(max_followup)]
    header = f"{'PID':<20} {'label':>5} {'time':>6}  " + "  ".join(f"{h:>6}" for h in year_headers)
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print("RESULTS SUMMARY")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        label_str = f"{int(r['label'])}" if r["label"] is not None else "  ?"
        time_str  = f"{r['time_at_event']:.1f}" if r["time_at_event"] is not None else "  ?"
        prob_str  = "  ".join(f"{p:6.3f}" for p in r["probabilities"])
        print(f"{r['pid']:<20} {label_str:>5} {time_str:>6}  {prob_str}")
    print(f"{'='*len(header)}\n")


def _write_results(results: List[dict], cfg: DictConfig) -> None:
    out_dir = Path(cfg.testing.get("output_dir", "./outputs/dcm_inference"))
    out_dir.mkdir(parents=True, exist_ok=True)

    max_followup = len(results[0]["probabilities"]) if results else 0
    year_keys  = [f"prob_yr{y+1}"  for y in range(max_followup)]
    y_seq_keys = [f"y_seq_yr{y+1}" for y in range(max_followup)]
    y_mask_keys = [f"y_mask_yr{y+1}" for y in range(max_followup)]

    # --- CSV ---
    csv_path = out_dir / f"{cfg.testing.use_checkpoint}_results.csv"
    fieldnames = ["pid", "series", "label", "time_at_event"] + y_seq_keys + y_mask_keys + year_keys
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "pid": r["pid"],
                "series": r["series"],
                "label": r["label"] if r["label"] is not None else "",
                "time_at_event": r["time_at_event"] if r["time_at_event"] is not None else "",
            }
            y_seq  = r.get("y_seq")  or [""] * max_followup
            y_mask = r.get("y_mask") or [""] * max_followup
            for i in range(max_followup):
                row[y_seq_keys[i]]  = y_seq[i]  if i < len(y_seq)  else ""
                row[y_mask_keys[i]] = y_mask[i] if i < len(y_mask) else ""
            for i, k in enumerate(year_keys):
                row[k] = f"{r['probabilities'][i]:.6f}"
            writer.writerow(row)
    print(f"Saved {csv_path}")

    # --- JSON ---
    json_path = out_dir / f"{cfg.testing.use_checkpoint}_results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Saved {json_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="./configs", config_name="dcm_inference.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    dicom_json = cfg.data.get("monai_dict_dicom")
    if not dicom_json:
        raise ValueError("cfg.data.monai_dict_dicom is required")

    print(f"Loading DICOM manifest: {dicom_json}")
    entries = _load_json(dicom_json)
    print(f"  {len(entries)} entries loaded")

    # --- ensure masks ---
    entries = _ensure_masks(entries, cfg)

    # --- filter to entries with both image and mask on disk ---
    entries = _filter_existing(entries, ["image", "mask"])
    if not entries:
        raise RuntimeError("No valid entries remain after filtering. Check paths in the manifest.")
    print(f"  {len(entries)} entries with valid image + mask")

    # --- dataset / loader ---
    tf = make_transformations(_dicom_transforms(cfg))
    dataset = Dataset(data=entries, transform=tf)
    num_workers = cfg.training.get("test_num_workers", 0)
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=False,
        drop_last=False,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg.training.get("prefetch_factor", 2)
    loader = DataLoader(**loader_kwargs)

    # --- model ---
    model = _build_model(cfg)

    # --- inference ---
    results = _run_inference(model, loader, cfg, entries)

    # --- output ---
    _print_summary(results, cfg.data.max_followup)
    _write_results(results, cfg)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
