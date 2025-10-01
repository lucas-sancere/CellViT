"""
Prepare dataset produced by `concat_binaryformat.py` for CellViT training.

Supports two input mask layouts produced by your concatenation step:
  1) **Single-channel** semantic mask with values in {0,1,2}.
  2) **One-hot** multi-channel mask (e.g., 6 channels) where only two
     channels are used for your positive classes (default: ch=4→class 1, ch=5→class 2).

Outputs (per fold):
  <out_root>/foldK/images/XXXX.png
  <out_root>/foldK/labels/XXXX.npy        # dict with keys {'inst_map', 'type_map'}

Usage examples
--------------
Single-channel masks in {0,1,2}:
  python prepare_cellvit_from_concat.py \
      --input_path /path/to/out_binary_56 \
      --output_path /path/to/cellvit_ready \
      --input_format single012

One-hot masks (6 channels) with class1 at ch4 and class2 at ch5 (CellViT style):
  python prepare_cellvit_from_concat.py \
      --input_path /path/to/out_binary_56 \
      --output_path /path/to/cellvit_ready \
      --input_format onehot \
      --class1_channel 4 --class2_channel 5

Auto-detect format (tries single012 first, then onehot):
  python prepare_cellvit_from_concat.py --input_path ... --output_path ... --input_format auto

Training config (examples)
--------------------------
num_nuclei_classes: 3
nuclei_types: {"background": 0, "class5": 1, "class6": 2}
# or rename to your liking, e.g. {"background":0, "tumor":1, "epithelial":2}

Notes
-----
- `inst_map` is generated on-the-fly by connected components over `type_map>0` (8-connectivity),
  with unique instance IDs across both positive classes.
- Overlaps in one-hot masks are resolved with **class1 taking precedence** over class2.
  Flip the writing order if you prefer the opposite.
"""

import argparse
import os
from pathlib import Path
import re
import sys
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# Optional: use your remapper if present
try:
    from cell_segmentation.utils.metrics import remap_label  # type: ignore
except Exception:
    def remap_label(arr: np.ndarray) -> np.ndarray:
        """Fallback: remap positive labels to 1..K in scan order."""
        arr = arr.astype(np.int64, copy=False)
        out = np.zeros_like(arr, dtype=np.int64)
        current = 0
        seen = {}
        pos = arr > 0
        vals = np.unique(arr[pos])
        for v in vals:
            current += 1
            seen[int(v)] = current
        if seen:
            lut_max = max(seen) + 1
            lut = np.zeros((lut_max + 1,), dtype=np.int64)
            for k, v in seen.items():
                lut[k] = v
            out[pos] = lut[arr[pos]]
        return out.astype(np.int32)


def cc_label(binary: np.ndarray, connectivity: int = 8) -> Tuple[np.ndarray, int]:
    """Connected components for a binary map. Prefers SciPy, falls back to pure NumPy.
    Returns (labels, num).
    """
    binary = (binary > 0).astype(np.uint8)
    try:
        from scipy import ndimage as ndi  # type: ignore
        structure = np.ones((3, 3), dtype=np.uint8) if connectivity == 8 else np.array(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8
        )
        lab, num = ndi.label(binary, structure=structure)
        return lab.astype(np.int32), int(num)
    except Exception:
        pass

    # Pure NumPy flood-fill (8-connected)
    H, W = binary.shape
    labels = np.zeros((H, W), dtype=np.int32)
    cur = 0
    for y in range(H):
        for x in range(W):
            if binary[y, x] and labels[y, x] == 0:
                cur += 1
                stack = [(y, x)]
                labels[y, x] = cur
                while stack:
                    yy, xx = stack.pop()
                    y0, y1 = max(0, yy - 1), min(H - 1, yy + 1)
                    x0, x1 = max(0, xx - 1), min(W - 1, xx + 1)
                    for ny in range(y0, y1 + 1):
                        for nx in range(x0, x1 + 1):
                            if (ny == yy and nx == xx):
                                continue
                            if binary[ny, nx] and labels[ny, nx] == 0:
                                labels[ny, nx] = cur
                                stack.append((ny, nx))
    return labels, cur


def discover_folds(root: Path) -> List[Path]:
    """Find subdirs named 'fold*'. Sort by natural order (fold0, fold1, fold2, ...)."""
    folds = [p for p in root.iterdir() if p.is_dir() and re.match(r"fold\d+", p.name)]
    def natkey(p: Path):
        m = re.search(r"(\d+)", p.name)
        return int(m.group(1)) if m else 1_000_000
    return sorted(folds, key=natkey)


def load_arrays(fold_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    images = np.load(fold_dir / "images.npy")  # (N,H,W,3)
    masks  = np.load(fold_dir / "masks.npy")   # (N,H,W,1) in {0,1,2} OR (N,H,W,C) one-hot
    return images, masks


def masks_to_type_map_single012(masks: np.ndarray) -> np.ndarray:
    """Expect (N,H,W[,1]) values in {0,1,2}. Returns (N,H,W) int32."""
    if masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]
    if masks.ndim != 3:
        raise ValueError(f"Expected single-channel masks with shape (N,H,W[,1]); got {masks.shape}")
    m = masks.astype(np.int32)
    uniq = np.unique(m)
    if not np.isin(uniq, [0, 1, 2]).all():
        raise ValueError(f"Single-channel mask has unexpected values {uniq}; expected subset of {{0,1,2}}")
    return m


def masks_to_type_map_onehot(masks: np.ndarray, class1_ch: int, class2_ch: int) -> np.ndarray:
    """Expect (N,H,W,C). Map two channels to {1,2}, others to 0. Returns (N,H,W) int32.
    Overlaps are resolved with class1 taking precedence.
    """
    if masks.ndim != 4 or masks.shape[-1] < max(class1_ch, class2_ch) + 1:
        raise ValueError(
            f"One-hot masks must be (N,H,W,C) with C>max(class1_ch,class2_ch); got {masks.shape}"
        )
    m012 = np.zeros(masks.shape[:3], dtype=np.int32)
    # write class2 first
    m012[masks[..., class2_ch] > 0] = 2
    # then class1 overrides overlaps
    m012[masks[..., class1_ch] > 0] = 1
    return m012


def build_inst_map_from_type_map(type_map: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """Label connected components across positive classes with unique IDs per nucleus."""
    N, H, W = type_map.shape
    inst_stack = np.zeros((N, H, W), dtype=np.int32)
    for i in range(N):
        next_offset = 0
        for cls in (1, 2):
            lab, num = cc_label(type_map[i] == cls, connectivity=connectivity)
            pos = lab > 0
            inst_stack[i][pos] = lab[pos] + next_offset
            next_offset += num
        inst_stack[i] = remap_label(inst_stack[i])
    return inst_stack


def save_fold(output_fold_path: Path, images: np.ndarray, inst_maps: np.ndarray, type_maps: np.ndarray) -> None:
    output_fold_path.mkdir(parents=True, exist_ok=True)
    (output_fold_path / "images").mkdir(parents=True, exist_ok=True)
    (output_fold_path / "labels").mkdir(parents=True, exist_ok=True)

    N = images.shape[0]
    for i in tqdm(range(N), desc=f"save {output_fold_path.name}"):
        outname_png = f"{output_fold_path.name.split('fold')[-1]}_{i}.png"
        outname_npy = f"{output_fold_path.name.split('fold')[-1]}_{i}.npy"
        Image.fromarray(images[i].astype(np.uint8)).save(output_fold_path / "images" / outname_png)
        np.save(output_fold_path / "labels" / outname_npy, {
            "inst_map": inst_maps[i].astype(np.int32),
            "type_map": type_maps[i].astype(np.int32),
        })


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert concatenated tiles (class5/class6) into CellViT-ready PNG + label dicts.",
    )
    ap.add_argument("--input_path", type=str, required=True,
                    help="Root directory containing fold*/images.npy and fold*/masks.npy")
    ap.add_argument("--output_path", type=str, required=True,
                    help="Destination root for CellViT-ready dataset")
    ap.add_argument("--input_format", type=str, default="auto",
                    choices=["auto", "single012", "onehot"],
                    help="Single-channel {0,1,2}, one-hot, or try to auto-detect")
    ap.add_argument("--class1_channel", type=int, default=4, help="One-hot channel index for class 1 (orig 5)")
    ap.add_argument("--class2_channel", type=int, default=5, help="One-hot channel index for class 2 (orig 6)")
    ap.add_argument("--connectivity", type=int, default=8, choices=[4, 8], help="Connectivity for instances")
    args = ap.parse_args()

    in_root = Path(args.input_path)
    out_root = Path(args.output_path)
    out_root.mkdir(parents=True, exist_ok=True)

    folds = discover_folds(in_root)
    if not folds:
        raise SystemExit(f"No fold* directories found under {in_root}")

    for fold_dir in folds:
        images, masks = load_arrays(fold_dir)
        N, H, W = images.shape[0], images.shape[1], images.shape[2]

        # Derive type_map depending on format
        fmt = args.input_format
        type_maps: np.ndarray

        if fmt == "single012" or (fmt == "auto" and (masks.ndim == 3 or (masks.ndim == 4 and masks.shape[-1] == 1))):
            type_maps = masks_to_type_map_single012(masks)
        elif fmt == "onehot" or (fmt == "auto" and masks.ndim == 4 and masks.shape[-1] > 1):
            type_maps = masks_to_type_map_onehot(masks, args.class1_channel, args.class2_channel)
        else:
            raise SystemExit(
                f"Could not auto-detect input format from masks shape {masks.shape}. "
                f"Please set --input_format explicitly."
            )

        # Safety: ensure only {0,1,2}
        uniq = np.unique(type_maps)
        if not np.isin(uniq, [0, 1, 2]).all():
            raise SystemExit(f"type_map contains unexpected values {uniq}; expected subset of {0,1,2}.")

        inst_maps = build_inst_map_from_type_map(type_maps, connectivity=args.connectivity)

        # Save
        out_fold = out_root / fold_dir.name
        save_fold(out_fold, images, inst_maps, type_maps)
        print(f"[OK] Saved {N} samples to {out_fold}")


if __name__ == "__main__":
    main()
