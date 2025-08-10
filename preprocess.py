from __future__ import annotations
import argparse
import rasterio
import csv, glob, json, os, re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import groupby
from typing import Dict, List, Optional, Tuple
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
S1_MIN_VALID_FRAC = 0.75
S2_MIN_VALID_FRAC = 0.75  
S1_MIN_STD = 0.01       
S2_MIN_STD = 0.01 

def build_flood_start(idx_s1: dict, idx_s2: dict, seq: str) -> Optional[str]:
    """Return the ISO date of the first flood event, or None if none."""
    dates = []
    for idx in (idx_s1, idx_s2):
        for ent in idx.get(seq, {}).values():
            if isinstance(ent, dict) and ent.get("FLOODING"):
                dates.append(ent["date"])
    return min(dates) if dates else None

def build_flood_dates(idx_s1: dict, idx_s2: dict, seq: str) -> set[str]:
    """Return all dates for seq where *either* sensor index marks FLOODING = true."""
    flood_dates = set()
    for idx in (idx_s1, idx_s2):
        for ent in idx.get(seq, {}).values():
            if isinstance(ent, dict) and ent.get("FLOODING"):
                flood_dates.add(ent["date"])
    return flood_dates

S2_BANDS = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]
DATE_RE = re.compile(r"(\d{4})[^\d]?(\d{2})[^\d]?(\d{2})")

def safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def parse_date_from_text(text: str) -> Optional[datetime]:
    matches = list(DATE_RE.finditer(text))
    if not matches:
        return None
    
    for match in matches:
        y, mo, d = map(int, match.groups())
        try:
            if 1900 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31:
                return datetime(y, mo, d)
        except ValueError:
            continue
    
    return None

def load_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def flood_flag(index: Dict, seq: str, date_iso: str) -> int:
    for ent in index.get(seq, {}).values():
        if isinstance(ent, dict) and ent.get("date") == date_iso:
            return int(bool(ent.get("FLOODING", False)))
    return 0

def write_geotiff(path: str, arrays: List[np.ndarray], ref_meta: dict, band_desc: List[str]) -> None:
    if os.path.exists(path):
        return
    meta = ref_meta.copy()
    meta.update(
        driver="GTiff",
        count=len(arrays),
        dtype="float32",
        compress="DEFLATE",
        predictor=3,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        nodata=None  
    )
    with rasterio.open(path, "w", **meta) as dst:
        for i, (arr, desc) in enumerate(zip(arrays, band_desc), start=1):
            dst.write(arr.astype(np.float32), i)
            try:
                dst.set_band_description(i, desc)
            except Exception:
                pass

def collect_s1_pairs(folder: str) -> Dict[str, Dict[str, str]]:
    vv_files = glob.glob(os.path.join(folder, "*_VV*.tif"))
    vh_files = glob.glob(os.path.join(folder, "*_VH*.tif"))
    by = defaultdict(dict)
    
    for f in vv_files:
        basename = os.path.basename(f)
        base = basename.replace(".tif", "")
        if base.endswith("_VV"):
            base = base[:-3] 
        by[base]["VV"] = f
    for f in vh_files:
        basename = os.path.basename(f)
        base = basename.replace(".tif", "")
        if base.endswith("_VH"):
            base = base[:-3]  
        by[base]["VH"] = f
    
    return {b: p for b, p in by.items() if "VV" in p and "VH" in p}

def load_s1_linear(pair: Dict[str, str], seq: str = "") -> Optional[Tuple[np.ndarray, np.ndarray, dict]]:
    try:
        with rasterio.open(pair["VV"]) as vv_src, rasterio.open(pair["VH"]) as vh_src:
            vv = vv_src.read(1)
            if (vh_src.height, vh_src.width) != (vv_src.height, vv_src.width):
                vh = vh_src.read(1, out_shape=(vv_src.height, vv_src.width), resampling=Resampling.bilinear)
            else:
                vh = vh_src.read(1)
            meta = vv_src.meta.copy()
    except RasterioIOError as e:
        print(f"Warning: could not read S1 pair {pair}: {e}")
        return None
    
    vv = vv.astype(np.float32)
    vh = vh.astype(np.float32)
    
    vv_mask = np.isfinite(vv) & (vv > 0)
    vh_mask = np.isfinite(vh) & (vh > 0)
    
    total_pixels = vv.size
    vv_valid_pixels = int(vv_mask.sum())
    vh_valid_pixels = int(vh_mask.sum())
    
    min_valid_pixels = max(1, int(S1_MIN_VALID_FRAC * total_pixels))
    
    basename = os.path.basename(pair["VV"])
    base_name = basename.replace(".tif", "")
    if base_name.endswith("_VV"):
        base_name = base_name[:-3]
    seq_prefix = f"[{seq}] " if seq else ""
    
    if vv_valid_pixels < min_valid_pixels or vh_valid_pixels < min_valid_pixels:
        print(f"Warning: {seq_prefix}S1 {base_name} insufficient valid fraction "
              f"(VV {vv_valid_pixels/total_pixels:.2%}, VH {vh_valid_pixels/total_pixels:.2%}) — dropping")
        return None
    
    vv_std = float(np.std(vv[vv_mask])) if vv_valid_pixels else 0.0
    vh_std = float(np.std(vh[vh_mask])) if vh_valid_pixels else 0.0
    if vv_std < S1_MIN_STD or vh_std < S1_MIN_STD:
        print(f"Warning: {seq_prefix}S1 {base_name} near‑zero variation "
              f"(std VV={vv_std:.6g}, VH={vh_std:.6g}) — dropping")
        return None
    
    return vv, vh, meta

def s1_to_db(a: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.clip(a, 1e-6, None)).astype(np.float32)

def make_s1_products(vv_lin: np.ndarray, vh_lin: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    vv_db  : σ⁰(VV) in dB
    vh_db  : σ⁰(VH) in dB
    gray   : **log-scaled grayscale** = 10·log10(VV+VH)
             (compressed dynamic range for cubical PH)
    """
    valid = (vv_lin > 0) & (vh_lin > 0)
    vv_db = s1_to_db(vv_lin)
    vh_db = s1_to_db(vh_lin)
    denom_db = s1_to_db(vv_lin + vh_lin + 1e-6)
    denom_out = np.full_like(vv_lin, np.nan, dtype=np.float32)
    denom_out[valid] = denom_db[valid]
    finite = np.isfinite(denom_out)
    sentinel = np.nan
    denom_out[~finite] = sentinel
    return vv_db, vh_db, denom_out

def find_s2_bases(folder: str) -> List[str]:
    b03 = glob.glob(os.path.join(folder, "*_B03.tif"))
    return [os.path.basename(p).split("_B03.tif")[0] for p in b03]

def load_s2_stack(base: str, folder: str, seq: str = "") -> Optional[Tuple[np.ndarray, dict]]:
    arrays: List[np.ndarray] = []
    ref_meta = None; H = W = None
    for b in S2_BANDS:
        p = os.path.join(folder, f"{base}_{b}.tif")
        if not os.path.exists(p):
            print(f"Warning: missing S2 band {b} for base {base}")
            return None
        try:
            with rasterio.open(p) as src:
                if ref_meta is None:
                    ref_meta = src.meta.copy(); H, W = src.height, src.width
                arr = src.read(1)
                if (src.height, src.width) != (H, W):
                    arr = src.read(1, out_shape=(H, W), resampling=Resampling.bilinear)
        except RasterioIOError as e:
            print(f"Warning: could not read S2 band {b} for base {base}: {e}")
            return None
        arrays.append(arr.astype(np.float32))
    
    stack = np.stack(arrays, axis=0) / 10000.0
    
    iG = S2_BANDS.index("B03"); iN = S2_BANDS.index("B08")
    b03 = stack[iG]; b08 = stack[iN]
    valid_g = np.isfinite(b03) & (b03 > 0)
    valid_n = np.isfinite(b08) & (b08 > 0)
    
    total_pixels = b03.size
    min_valid_pixels = max(1, int(S2_MIN_VALID_FRAC * total_pixels))
    seq_prefix = f"[{seq}] " if seq else ""
    
    if valid_g.sum() < min_valid_pixels or valid_n.sum() < min_valid_pixels:
        print(f"Warning: {seq_prefix}S2 {base} insufficient valid fraction "
              f"(B03 {valid_g.mean():.2%}, B08 {valid_n.mean():.2%}) — dropping")
        return None
    
    valid_both = valid_g & valid_n
    if valid_both.sum() > 0:
        ndwi = (b03 - b08) / (b03 + b08 + 1e-6)
        ndwi = np.clip(ndwi, -1.0, 1.0)
        ndwi_std = float(np.std(ndwi[valid_both]))
        if ndwi_std < S2_MIN_STD:
            print(f"Warning: {seq_prefix}S2 {base} near‑zero NDWI variation "
                  f"(std NDWI={ndwi_std:.6g}) — dropping")
            return None
    
    return stack, ref_meta

def make_ndwi_gray(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns: gray_tda, ndwi_phys, valid_mask
            gray_tda : value stored in GeoTIFF for TDA (sublevel filtration only)
            ndwi_phys: physical NDWI (not inverted) for PNG
            valid_mask: boolean mask of invalid pixels (for PNG masking)
        """
        iG = S2_BANDS.index("B03"); iN = S2_BANDS.index("B08")
        G = stack[iG]; N = stack[iN]
        ndwi_phys = (G - N) / (G + N + 1e-6)
        ndwi_phys = np.clip(ndwi_phys, -1.0, 1.0)
        valid = (G>0) & (N>0) & np.isfinite(G) & np.isfinite(N)
        arr = -ndwi_phys
        sentinel = np.nan
        gray_tda = arr.copy()
        gray_tda[~valid] = sentinel
        gray_tda = gray_tda.astype(np.float32)
        return gray_tda, ndwi_phys.astype(np.float32), ~valid

@dataclass
class S1Item:
    dt: datetime
    vv_db: np.ndarray
    vh_db: np.ndarray
    denom: np.ndarray        
    meta: dict
    flood: int
    base_name: str           
    idx_m: int = -1
    idx_c: int = -1

@dataclass
class S2Item:
    dt: datetime
    stack: np.ndarray
    gray_tda: np.ndarray     
    gray_disp: np.ndarray    
    mask: np.ndarray         
    meta: dict
    flood: int
    base_name: str           
    idx_m: int = -1
    idx_c: int = -1

def extract_s1_timestamp(filename: str) -> Optional[str]:
    """Extract the timestamp from S1 filename format: ...T030014... -> 030014"""
    import re
    match = re.search(r'T(\d{6})', filename)
    return match.group(1) if match else None

def assign_indices(s1_items: List[S1Item], s2_items: List[S2Item]) -> None:
    s1_items.sort(key=lambda x: x.dt)
    s2_items.sort(key=lambda x: x.dt)
    for i, it in enumerate(s1_items): 
        it.idx_m = i
    for i, it in enumerate(s2_items): 
        it.idx_m = i
    
    records = []
    for it in s1_items:
        records.append({
            'dt': it.dt,
            'date': it.dt.date().isoformat(),
            'sat': 's1',
            'flood': it.flood,
            'fname': it.base_name,
            'item': it
        })
    for it in s2_items:
        records.append({
            'dt': it.dt,
            'date': it.dt.date().isoformat(), 
            'sat': 's2',
            'flood': it.flood,
            'fname': it.base_name,
            'item': it
        })
    
    records.sort(key=lambda r: r['date'])
    
    ordered = []
    for date, group in groupby(records, key=lambda r: r['date']):
        grp = list(group)
        n = len(grp)
        if n == 2:
            f0, f1 = grp[0]['flood'], grp[1]['flood']
            s0, s1 = grp[0]['sat'], grp[1]['sat']
            fn0, fn1 = grp[0]['fname'], grp[1]['fname']
            if f0 != f1:
                grp = sorted(grp, key=lambda r: r['flood'])
                print(f"Ambiguous two images for date {date}: {s0}={f0}, {s1}={f1}: resolved based on flood flag ({grp[0]['sat']} before {grp[1]['sat']})")
            elif s0 == s1 == 's1':
                ts0 = extract_s1_timestamp(fn0)
                ts1 = extract_s1_timestamp(fn1)
                if ts0 and ts1:
                    grp = sorted(grp, key=lambda r: extract_s1_timestamp(r['fname']))
                    print(f"Ambiguous two images for date {date}: {s0}={f0}, {s1}={f1}: resolved by S1 timestamp order (T{ts0} before T{ts1})")
                else:
                    grp = sorted(grp, key=lambda r: r['fname'])
                    print(f"Ambiguous two images for date {date}: {s0}={f0}, {s1}={f1}: resolved by filename order ({grp[0]['fname']} before {grp[1]['fname']})")
            else:
                grp = sorted(grp, key=lambda r: r['fname'])
                print(f"Ambiguous two images for date {date}: {s0}={f0}, {s1}={f1}: resolved by filename order ({grp[0]['fname']} before {grp[1]['fname']})")
        elif n > 2:
            floods = {r['flood'] for r in grp}
            sats = {r['sat'] for r in grp}
            files = [r['fname'] for r in grp]
            
            if len(floods) == 1 and len(sats) == 1 and 's1' in sats:
                s1_with_timestamps = []
                for r in grp:
                    ts = extract_s1_timestamp(r['fname'])
                    if ts:
                        s1_with_timestamps.append((ts, r))
                
                if len(s1_with_timestamps) == n:
                    s1_with_timestamps.sort(key=lambda x: x[0])
                    grp = [r for ts, r in s1_with_timestamps]
                    timestamps = [ts for ts, r in s1_with_timestamps]
                    print(f"Multiple ({n}) S1 images for date {date}: all flood={floods.pop()}; resolved by timestamp order {timestamps}")
                else:
                    grp = sorted(grp, key=lambda r: r['fname'])
                    print(f"Multiple ({n}) S1 images for date {date}: all flood={floods.pop()}; resolved by filename order {files}")
            elif len(floods) == 1:
                sorted_files = sorted(files)
                print(f"Multiple ({n}) images for date {date}: all flood={floods.pop()}; resolved by filename order {sorted_files}")
                fname_to_rec = {r['fname']: r for r in grp}
                grp = [fname_to_rec[fn] for fn in sorted_files]
            else:
                print(f"ERROR: {n} images found for date {date}; mixed flood statuses; images {files}")
        ordered.extend(grp)
    
    for idx_c, record in enumerate(ordered):
        record['item'].idx_c = idx_c

def main():
    parser = argparse.ArgumentParser(description='Process satellite data for flood detection')
    parser.add_argument('--s1-dir', required=True, help='Directory for S1 output files')
    parser.add_argument('--s2-dir', required=True, help='Directory for S2 output files')
    parser.add_argument('--sen12-root', default='.', help='Root directory containing sequence data (default: current directory)')
    parser.add_argument('--json-s1', default='S1list.json', help='S1 index JSON file (default: S1list.json)')
    parser.add_argument('--json-s2', default='S2list.json', help='S2 index JSON file (default: S2list.json)')
    
    args = parser.parse_args()
    
    s1_dir = args.s1_dir
    s2_dir = args.s2_dir
    sen12_root = args.sen12_root
    
    safe_mkdir(s1_dir)
    safe_mkdir(s2_dir)
    safe_mkdir(os.path.join(s1_dir, "gray"))
    safe_mkdir(os.path.join(s1_dir, "stacked"))
    safe_mkdir(os.path.join(s2_dir, "gray"))
    safe_mkdir(os.path.join(s2_dir, "stacked"))

    idx_s1 = load_json(args.json_s1)
    idx_s2 = load_json(args.json_s2)

    all_dirs = os.listdir(sen12_root)
    seq_dirs = [d for d in all_dirs if d.isdigit()]
    print(f"Found {len(seq_dirs)} sequence directories out of {len(all_dirs)} total directories")
    
    seq_ids = sorted(seq_dirs, key=lambda s: (len(s) == 4, int(s)))

    rows: List[Tuple[str,str,str,int,int,str,int]] = []

    skipped_dates = []
    skipped_s1_pairs = []
    skipped_s2_bases = []

    for seq in seq_ids:
        print(f"Processing sequence: {seq}")
        seq_dir = os.path.join(sen12_root, seq)


        s1_items: List[S1Item] = []
        for base, files in collect_s1_pairs(seq_dir).items():
            dt = parse_date_from_text(base)
            if not dt:
                skipped_dates.append(base)
                continue
            loaded = load_s1_linear(files, seq)
            if loaded is None:
                skipped_s1_pairs.append(base)
                continue
            vv_lin, vh_lin, meta = loaded
            vv_db, vh_db, denom = make_s1_products(vv_lin, vh_lin)
            date_iso = dt.date().isoformat()
            is_flood = flood_flag(idx_s1, seq, date_iso)
            s1_items.append(S1Item(dt, vv_db, vh_db, denom, meta, is_flood, base))

        s2_items: List[S2Item] = []
        for base in find_s2_bases(seq_dir):
            dt = parse_date_from_text(base)
            if not dt:
                skipped_dates.append(base)
                continue
            loaded = load_s2_stack(base, seq_dir, seq)
            if loaded is None:
                skipped_s2_bases.append(base)
                continue
            stack, meta = loaded
            gray_tda, ndwi_phys, mask = make_ndwi_gray(stack)
            date_iso = dt.date().isoformat()
            is_flood = flood_flag(idx_s2, seq, date_iso)
            s2_items.append(S2Item(dt, stack, gray_tda, ndwi_phys, mask, meta, is_flood, base))

        assign_indices(s1_items, s2_items)

        # Hypothesis enforcement: after the first flood date, all subsequent frames are labeled as flooded (1)
        flood_start_iso = build_flood_start(idx_s1, idx_s2, seq)
        if flood_start_iso:
            try:
                flood_start_dt = datetime.fromisoformat(flood_start_iso)
                for it in s1_items:
                    if it.dt >= flood_start_dt:
                        it.flood = 1
                for it in s2_items:
                    if it.dt >= flood_start_dt:
                        it.flood = 1
                print(f"Hypothesis applied: sequence {seq} flood start {flood_start_iso}; all frames from that date forward set to FLOOD=1")
            except ValueError:
                print(f"Warning: could not parse flood start date '{flood_start_iso}' for sequence {seq}; skipping hypothesis relabeling")

        for it in s1_items:
            date_str = it.dt.strftime("%Y%m%d")
            stem = f"{seq}_{it.idx_c:02d}_s1_{it.idx_m:02d}_{date_str}_{it.flood}"
            stacked_path = os.path.join(s1_dir, "stacked", stem + ".tif")
            gray_path = os.path.join(s1_dir, "gray", stem + ".tif")

            write_geotiff(stacked_path, [it.vv_db, it.vh_db], it.meta, ["VV_dB","VH_dB"])
            write_geotiff(gray_path, [it.denom], it.meta, ["s1_gray"])

            rows.append((seq, date_str, "s1", it.idx_c, it.idx_m, os.path.basename(stem+".tif"), it.flood))

        for it in s2_items:
            date_str = it.dt.strftime("%Y%m%d")
            stem = f"{seq}_{it.idx_c:02d}_s2_{it.idx_m:02d}_{date_str}_{it.flood}"
            stacked_path = os.path.join(s2_dir, "stacked", stem + ".tif")
            gray_path = os.path.join(s2_dir, "gray", stem + ".tif")

            write_geotiff(stacked_path, list(it.stack), it.meta, S2_BANDS)
            write_geotiff(gray_path, [it.gray_tda], it.meta, ["s2_gray_tda"])

            rows.append((seq, date_str, "s2", it.idx_c, it.idx_m, os.path.basename(stem+".tif"), it.flood))

    rows.sort(key=lambda r: ((len(r[0]) == 4, int(r[0])), r[3]))

    with open(os.path.join(s2_dir, "all_images.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_id","date","sat","idx_c","idx_m","filename","flood"])
        w.writerows(rows)

    if skipped_dates:
        print(f"Skipped {len(skipped_dates)} items due to date parsing failures")
        print(f"Skipped date items: {skipped_dates}")
    if skipped_s1_pairs:
        print(f"Skipped {len(skipped_s1_pairs)} S1 pairs due to IO errors")
        print(f"Skipped S1 pairs: {skipped_s1_pairs}")
    if skipped_s2_bases:
        print(f"Skipped {len(skipped_s2_bases)} S2 acquisitions due to missing bands")
        print(f"Skipped S2 bases: {skipped_s2_bases}")

    print(f"Done. Wrote {len(rows)} rows to all_images.csv")

if __name__ == "__main__":
    main()