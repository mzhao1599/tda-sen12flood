import argparse, warnings
from pathlib import Path
import numpy as np
import rasterio
from tqdm import tqdm
import cripser

def compute_diagram(img: np.ndarray):
    """
    Run Cubical Ripser on a 2‑D numpy array.

    Parameters
    ----------
    img     : 2‑D ndarray (float64 recommended)

    Returns
    -------
    ph      : ndarray of shape (n, 9) as described in cripser docs
              Note: May contain infinite death values for persistent classes
    """
    return cripser.computePH(np.ascontiguousarray(img, dtype=np.float64),maxdim=1)

TARGET_SIZE = 120  

def walk_and_cache(root: Path,
                   overwrite: bool = False):
    """
    Iterate over all *.tif under *root*, write <fname>.npy with the diagram
    in a 'cubical' subdirectory structure. Always bilinear-resize to
    TARGET_SIZE x TARGET_SIZE prior to computing persistence.

    A diagram is only recomputed when the .npy file is missing or
    --overwrite is given.

    Saved diagrams are filtered to contain only finite birth/death values
    and stored as (dim, birth, death) arrays for clean downstream processing.
    """
    tif_paths = sorted(Path(root).rglob("*.tif"))
    if not tif_paths:
        warnings.warn(f"No .tif images found under {root}")
        return

    subdir_name = "cubical"
    
    for tif in tqdm(tif_paths, desc=f"{subdir_name.title()} persistence", unit="img"):
        rel_path = tif.relative_to(root)
        output_dir = root / subdir_name / rel_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        npy = output_dir / (tif.stem + ".npy")
        
        if npy.exists() and not overwrite:
            continue                           
        with rasterio.open(tif) as src:
            img = src.read(1).astype(np.float32)   
        img[np.isnan(img)] = np.nan 
        if (img.shape[0] != TARGET_SIZE) or (img.shape[1] != TARGET_SIZE):
            import torch.nn.functional as F, torch
            t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  
            img = F.interpolate(t, size=(TARGET_SIZE, TARGET_SIZE),
                                mode="bilinear", align_corners=False).squeeze().numpy()
        
        pd = compute_diagram(img)
        max_reasonable_value = 1e100  
        finite_mask = (np.isfinite(pd[:, 1]) & np.isfinite(pd[:, 2]) & (pd[:, 1] < max_reasonable_value) & (pd[:, 2] < max_reasonable_value))
        pd_clean = pd[finite_mask, :3]  
        nonzero = pd_clean[:, 2] > pd_clean[:, 1]
        pd_clean = pd_clean[nonzero]
        np.save(npy, pd_clean)

def main():
    p = argparse.ArgumentParser(description="Cache cubical persistence diagrams (fixed resize=120)")
    p.add_argument("root", help="root folder containing .tif data")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute even when <img>.npy exists")
    args = p.parse_args()
    walk_and_cache(Path(args.root), overwrite=args.overwrite)

if __name__ == "__main__":
    main()