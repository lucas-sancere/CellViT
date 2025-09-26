




"""
how to use:

python prepare_monuseg.py --input_path /path/to/raw --output_path /path/to/out --mode threeclass

Then in your training config set:
num_nuclei_classes: 3
nuclei_types: {"background": 0, "classA": 1, "classB": 2}
"""




import inspect
import os
import sys
from pathlib import Path
import argparse

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

import numpy as np
from PIL import Image
from tqdm import tqdm
from cell_segmentation.utils.metrics import remap_label

# Default collapse for 3-class: PanNuke {1,2} -> 1, {3,4,5} -> 2
PANNUKE_TO_3CLASS = {
    0: 0,     # background
    1: 1, 2: 1,   # group A -> class 1
    3: 2, 4: 2, 5: 2,  # group B -> class 2
}

def _collapse_to_three_classes(raw_type_map: np.ndarray) -> np.ndarray:
    """Map PanNuke layer ids 0..5 to 0..2 according to PANNUKE_TO_3CLASS."""
    vmap = np.vectorize(lambda v: PANNUKE_TO_3CLASS.get(int(v), 0))
    out = vmap(raw_type_map).astype(np.int32)
    return out

def process_fold(fold: int, input_path: Path, output_path: Path, mode: str) -> None:
    """
    mode:
      - 'binary'     -> type_map in {0,1}
      - 'threeclass' -> type_map in {0,1,2} via PANNUKE_TO_3CLASS
    """
    fold_path = Path(input_path) / f"fold{fold}"
    output_fold_path = Path(output_path) / f"fold{fold}"
    output_fold_path.mkdir(exist_ok=True, parents=True)
    (output_fold_path / "images").mkdir(exist_ok=True, parents=True)
    (output_fold_path / "labels").mkdir(exist_ok=True, parents=True)

    print(f"Fold: {fold}")
    print("Loading large numpy files, this may take a while")
    images = np.load(fold_path / "images.npy")        # (N, H, W, 3) uint8
    masks  = np.load(fold_path / "masks.npy")         # (N, H, W, C) int - per-type instance maps

    # Derive H, W from data instead of hard-coding
    H, W = masks.shape[1], masks.shape[2]
    C    = masks.shape[3]

    print("Process images")
    for i in tqdm(range(len(images)), total=len(images)):
        outname = f"{fold}_{i}.png"
        im = Image.fromarray(images[i].astype(np.uint8))
        im.save(output_fold_path / "images" / outname)

    print("Process masks")
    for i in tqdm(range(len(images)), total=len(images)):
        outname = f"{fold}_{i}.npy"

        # masks[i] contains C layers; each layer is an instance map (0 = background)
        mask = masks[i]  # (H, W, C)

        # --- 1) Build a single global instance map by stacking the layers ---
        inst_map = np.zeros((H, W), dtype=np.int32)
        num_nuc = 0
        for j in range(C):
            # make instance ids in this layer contiguous starting at 0
            layer_res = remap_label(mask[:, :, j])
            # paste layer instances into global map with an offset
            inst_map = np.where(layer_res != 0, layer_res + num_nuc, inst_map)
            num_nuc += int(layer_res.max())
        inst_map = remap_label(inst_map).astype(np.int32)

        # --- 2) Build a raw type map 0..C where each pixel takes the layer id (j+1) it belongs to ---
        # NOTE: If your "masks" are already binary (presence), this is correct.
        #       If they contain raw instance ids, the comparison (> 0) keeps presence logic.
        raw_type_map = np.zeros((H, W), dtype=np.int32)
        for j in range(C):
            layer_bin = (mask[:, :, j] > 0).astype(np.int32)
            # pixels belonging to layer j become type id (j+1)
            raw_type_map = np.where(layer_bin != 0, j + 1, raw_type_map)

        # --- 3) Collapse to the requested number of classes (and validate) ---
        if mode == "binary":
            # Any nucleus -> 1
            type_map = (inst_map > 0).astype(np.int32)  # {0,1}
            valid_min, valid_max = 0, 1
        elif mode == "threeclass":
            # Collapse PanNuke 1..5 -> {1,2}
            type_map = _collapse_to_three_classes(raw_type_map)  # {0,1,2}
            valid_min, valid_max = 0, 2
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'binary' or 'threeclass'.")

        # --- 4) Safety checks ---
        uniq_vals = np.unique(type_map)
        if uniq_vals.min() < valid_min or uniq_vals.max() > valid_max:
            raise ValueError(
                f"[process_fold] type_map has values outside [{valid_min},{valid_max}] "
                f"for fold {fold}, sample {i}: uniques={uniq_vals.tolist()}"
            )
        if inst_map.shape != (H, W) or type_map.shape != (H, W):
            raise ValueError("inst_map/type_map have unexpected shape.")

        # --- 5) Save dict (compatible with trainer expectations) ---
        outdict = {"inst_map": inst_map, "type_map": type_map}
        np.save(output_fold_path / "labels" / outname, outdict)




def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Prepare dataset for CellViT (images + instance/type maps).",
    )
    parser.add_argument("--input_path", type=str, required=True,
                        help="Input path to the original dataset (with fold*/images.npy, fold*/masks.npy).")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path to store the processed dataset.")
    parser.add_argument("--mode", type=str, default="binary", choices=["binary", "threeclass"],
                        help="Output type map mode: 'binary' -> {0,1}, 'threeclass' -> {0,1,2} via PANNUKE_TO_3CLASS.")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    for fold in [0, 1, 2]:
        process_fold(fold, input_path, output_path, mode=args.mode)

if __name__ == "__main__":
    main()
