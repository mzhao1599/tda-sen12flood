import os, math, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

RESNET_TXT   = "{modality}_{dir_tag}_resnet_detailed_predictions.txt"
TOPOGE_TXT   = "{modality}_topoGE_{dir_tag}_detailed_predictions.txt"
FUSION_TXT   = "{modality}_{dir_tag}_latefusion_detailed_predictions.txt"

@dataclass
class FrameRec:
    seq_id: str
    idx: int
    prob: float
    label: int

def _parse_resnet_txt(path: str) -> List[FrameRec]:
    """Parse resnet detailed predictions supporting legacy tabular and new topoGE-style.
    Returns a flat list of FrameRec.
    """
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.read().splitlines()
    new_style = any(l.startswith('SEQ ') for l in lines) or any(l.startswith('Optimal threshold:') for l in lines)
    out: List[FrameRec] = []
    if new_style:
        current_sid = None
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if s.startswith('Optimal threshold:') or s.startswith('Format:'):
                continue
            if s.startswith('SEQ '):
                parts = s.split()
                current_sid = parts[1] if len(parts) >= 2 else 'unknown'
                continue
            parts = s.split()
            if len(parts) >= 3 and parts[0].isdigit():
                try:
                    frame_idx = int(parts[0])
                    prob = float(parts[1])
                    label = int(parts[2])
                except Exception:
                    continue
                out.append(FrameRec(current_sid or 'unknown', frame_idx, prob, label))
        return out
    header_seen = False
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith('#'):
            continue
        if not header_seen:
            if 'seq_id' in s and 'frame_idx' in s:
                header_seen = True
                continue
            else:
                continue
        parts = s.split()
        if len(parts) < 4:
            continue
        try:
            seq_id = parts[0]
            frame_idx = int(parts[1])
            prob = float(parts[2])
            label = int(parts[3])
        except Exception:
            continue
        out.append(FrameRec(seq_id, frame_idx, prob, label))
    return out

def _parse_topoge_txt(path: str) -> List[FrameRec]:
    out = []
    current_sid = None
    with open(path, "r", encoding='utf-8', errors='replace') as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if s.startswith("SEQ "):
                parts = s.split()
                current_sid = parts[1] if len(parts) >= 2 else "unknown"
                continue
            toks = s.split()
            if len(toks) >= 5 and toks[0].isdigit():
                idx = int(toks[0]); prob = float(toks[1]); lbl = int(toks[2])
                out.append(FrameRec(current_sid, idx, prob, lbl))
    return out

def _print_file_stats(name: str, recs: List[FrameRec]):
    """Print basic stats and metrics for a single predictions file."""
    n = len(recs)
    if n == 0:
        print(f"[late_fusion] {name} file: empty")
        return
    probs = np.array([r.prob for r in recs], dtype=np.float64)
    labels = np.array([r.label for r in recs], dtype=np.int64)
    pos = int(labels.sum()); neg = int(n - pos)
    mean_p = float(probs.mean()); std_p = float(probs.std())
    m05 = _metrics(probs, labels, 0.50)
    r90 = _best_recall_at_90p(probs, labels)
    if r90 is not None:
        print(
            f"[late_fusion] {name} file: total={n} pos={pos} neg={neg} mean_prob={mean_p:.3f} std={std_p:.3f} "
            f"@0.50 P={m05['precision']:.3f} R={m05['recall']:.3f} F1={m05['f1']:.3f}; "
            f"R@P≥0.90 th={r90['threshold']:.2f} P={r90['precision']:.3f} R={r90['recall']:.3f} F1={r90['f1']:.3f}"
        )
    else:
        print(
            f"[late_fusion] {name} file: total={n} pos={pos} neg={neg} mean_prob={mean_p:.3f} std={std_p:.3f} "
            f"@0.50 P={m05['precision']:.3f} R={m05['recall']:.3f} F1={m05['f1']:.3f}; R@P≥0.90: N/A"
        )

def _join_by_seq_idx(a: List[FrameRec], b: List[FrameRec]) -> List[Tuple[FrameRec, FrameRec]]:
    key = lambda r: (r.seq_id, r.idx)
    a_sorted = sorted(a, key=key); b_sorted = sorted(b, key=key)
    out = []
    ia = ib = 0
    while ia < len(a_sorted) and ib < len(b_sorted):
        ka = key(a_sorted[ia]); kb = key(b_sorted[ib])
        if ka == kb:
            out.append((a_sorted[ia], b_sorted[ib]))
            ia += 1; ib += 1
        elif ka < kb:
            ia += 1
        else:
            ib += 1
    return out

def _confusion(preds: np.ndarray, labels: np.ndarray) -> Tuple[int,int,int,int]:
    TP = int(((preds==1) & (labels==1)).sum())
    FP = int(((preds==1) & (labels==0)).sum())
    TN = int(((preds==0) & (labels==0)).sum())
    FN = int(((preds==0) & (labels==1)).sum())
    return TP, FP, TN, FN

def _metrics(probs: np.ndarray, labels: np.ndarray, th: float) -> Dict:
    preds = (probs >= th).astype(int)
    TP, FP, TN, FN = _confusion(preds, labels)
    P = TP/(TP+FP) if (TP+FP) > 0 else 0.0
    R = TP/(TP+FN) if (TP+FN) > 0 else 0.0
    F1 = 2*P*R/(P+R) if (P+R) > 0 else 0.0
    ACC = (TP+TN)/(TP+FP+TN+FN) if (TP+FP+TN+FN) > 0 else 0.0
    return {"threshold": float(th), "precision": float(P), "recall": float(R),
            "f1": float(F1), "accuracy": float(ACC),
            "tp": int(TP), "fp": int(FP), "tn": int(TN), "fn": int(FN)}

def _best_recall_at_90p(probs: np.ndarray, labels: np.ndarray) -> Optional[Dict]:
    thresholds = np.linspace(0, 1, 101)
    best = None
    for th in thresholds:
        m = _metrics(probs, labels, th)
        if m["precision"] >= 0.90:
            if best is None or m["recall"] > best["recall"]:
                best = m
    return best

def _fbeta_from_metrics(m: Dict, beta: float) -> float:
    P, R = m["precision"], m["recall"]
    denom = (beta*beta)*P + R
    return (1+beta*beta)*P*R/denom if denom > 0 else 0.0

def _grid_search_weight_05(p_res: np.ndarray, p_top: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict]:
    beta = math.sqrt(2.0)
    best = None
    best_w = 0.5
    for w in np.linspace(0, 1, 101):
        fused = w*p_res + (1.0 - w)*p_top
        m05 = _metrics(fused, labels, 0.50)
        fb = _fbeta_from_metrics(m05, beta)
        if (best is None) or (fb > best["fbeta"]) or (abs(fb - best["fbeta"]) < 1e-12 and w > best_w):
            best = {"metrics_at_05": m05, "fbeta": fb, "w": float(w)}
            best_w = float(w)
    return best_w, best

def _load_pair(modality: str, dir_tag: str) -> Tuple[List[FrameRec], List[FrameRec]]:
    res_path = RESNET_TXT.format(modality=modality, dir_tag=dir_tag)
    topo_path = TOPOGE_TXT.format(modality=modality, dir_tag=dir_tag)
    if not os.path.exists(res_path):
        raise FileNotFoundError(res_path)
    if not os.path.exists(topo_path):
        raise FileNotFoundError(topo_path)
    return _parse_resnet_txt(res_path), _parse_topoge_txt(topo_path)

def _write_fused(modality: str, dir_tag: str, pairs, fused_probs: np.ndarray, opt_th: Optional[float]):
    out_path = FUSION_TXT.format(modality=modality, dir_tag=dir_tag)
    if opt_th is None:
        opt_th = 0.50
    with open(out_path, "w", encoding='utf-8', errors='replace') as f:
        f.write(f"# modality={modality} optimal_threshold={opt_th:.4f}\n")
        f.write("seq_id\tframe_idx\tprob\tlabel\tpred_05\tpred_opt\n")
        for (ra, ta), p in zip(pairs, fused_probs):
            lbl = ra.label
            pred05 = 1 if p >= 0.5 else 0
            predOpt = 1 if p >= opt_th else 0
            f.write(f"{ra.seq_id}\t{ra.idx}\t{p:.6f}\t{lbl}\t{pred05}\t{predOpt}\n")
    print(f"[late_fusion] Wrote fused predictions -> {out_path}")

def _write_ckpt(modality: str, dir_tag: str, chosen: Dict):
    print(f"[late_fusion] Fusion summary ({modality}-{dir_tag})")
    print(f"  selection: {chosen.get('selection')}")
    print(f"  winner: {chosen.get('winner')}")
    print(f"  weights: resnet={chosen.get('weight_resnet'):.2f} topoge={chosen.get('weight_topoge'):.2f}")
    m05 = chosen.get('metrics_at_05', {})
    print("  @0.50 metrics:")
    print(f"    th={m05.get('threshold', 0.5):.2f} P={m05.get('precision', 0):.3f} R={m05.get('recall', 0):.3f} F1={m05.get('f1', 0):.3f} ACC={m05.get('accuracy', 0):.3f}")
    print(f"    TP={m05.get('tp', 0)} FP={m05.get('fp', 0)} TN={m05.get('tn', 0)} FN={m05.get('fn', 0)}")
    r90 = chosen.get('r90p', {})
    if r90.get('available'):
        rm = r90.get('metrics', {})
        print("  R@P≥0.90:")
        print(f"    th={r90.get('threshold', 0):.2f} P={rm.get('precision', 0):.3f} R={rm.get('recall', 0):.3f} F1={rm.get('f1', 0):.3f} ACC={rm.get('accuracy', 0):.3f}")
        print(f"    TP={rm.get('tp', 0)} FP={rm.get('fp', 0)} TN={rm.get('tn', 0)} FN={rm.get('fn', 0)}")
    diag = chosen.get('diagnostics', {})
    fbc = diag.get('fbeta_candidates', {})
    print("  Fβ@0.50 candidates (β=√2):")
    print(f"    fusion={fbc.get('fusion', 0):.3f} resnet={fbc.get('resnet', 0):.3f} topoge={fbc.get('topoge', 0):.3f}")
    join = diag.get('join', {})
    print("  join:")
    print(f"    paired={join.get('paired_frames', 0)} res_only={join.get('res_only', 0)} topo_only={join.get('topo_only', 0)} union={join.get('union', 0)} coverage={join.get('coverage', 0):.3f}")

def run_one(modality: str, args):
    beta = math.sqrt(2.0)
    dir_tag = "bi" if args.bidirectional else "uni"
    res_list, topo_list = _load_pair(modality, dir_tag)
    _print_file_stats("ResNet", res_list)
    _print_file_stats("topoGE", topo_list)
    pairs = _join_by_seq_idx(res_list, topo_list)
    if not pairs:
        raise RuntimeError(f"No matching (seq_id, frame_idx) pairs for {modality}-{dir_tag}.")
    res_keyset = {(r.seq_id, r.idx) for r in res_list}
    topo_keyset = {(r.seq_id, r.idx) for r in topo_list}
    pairset = {(r.seq_id, r.idx) for r,_ in pairs}
    cover = len(pairset) / max(1, len(res_keyset | topo_keyset))
    print(f"[late_fusion] Join coverage: {len(pairset)} frames; res-only: {len(res_keyset - pairset)}, topo-only: {len(topo_keyset - pairset)}, total-union: {len(res_keyset | topo_keyset)}; coverage={cover:.3f}")
    if cover < 0.8:
        print("[late_fusion] WARNING: paired subset <80% of union; weights may be skewed by subset.")
    p_res = np.array([a.prob for a,b in pairs], dtype=np.float64)
    p_top = np.array([b.prob for a,b in pairs], dtype=np.float64)
    labels = np.array([a.label for a,b in pairs], dtype=np.int64)
    best_w, best = _grid_search_weight_05(p_res, p_top, labels)
    fused_A = best_w*p_res + (1.0 - best_w)*p_top
    m05_A = best["metrics_at_05"]; fb_A = best["fbeta"]
    r90_A = _best_recall_at_90p(fused_A, labels)
    print(f"[late_fusion] Fβ@0.50 best weight: w_res={best_w:.2f}, w_top={1-best_w:.2f}; P={m05_A['precision']:.3f} R={m05_A['recall']:.3f} F1={m05_A['f1']:.3f} Fβ={fb_A:.3f} (β=√2)")
    m05_R = _metrics(p_res, labels, 0.50); fb_R = _fbeta_from_metrics(m05_R, beta)
    r90_R = _best_recall_at_90p(p_res, labels)
    print(f"[late_fusion] ResNet-only @0.50: P={m05_R['precision']:.3f} R={m05_R['recall']:.3f} F1={m05_R['f1']:.3f} Fβ={fb_R:.3f}")
    m05_T = _metrics(p_top, labels, 0.50); fb_T = _fbeta_from_metrics(m05_T, beta)
    r90_T = _best_recall_at_90p(p_top, labels)
    print(f"[late_fusion] Topo-only  @0.50: P={m05_T['precision']:.3f} R={m05_T['recall']:.3f} F1={m05_T['f1']:.3f} Fβ={fb_T:.3f}")
    winner = "fusion"; chosen_w = best_w; fused = fused_A; m05 = m05_A; r90 = r90_A
    best_fb = fb_A
    if fb_R > best_fb + 1e-12:
        winner, best_fb = "resnet", fb_R
        chosen_w, fused, m05, r90 = 1.0, p_res, m05_R, r90_R
    if fb_T > best_fb + 1e-12:
        winner, best_fb = "topoge", fb_T
        chosen_w, fused, m05, r90 = 0.0, p_top, m05_T, r90_T
    if winner != "fusion":
        print(f"[late_fusion] No-degrade guard: overriding to {winner} (higher Fβ@0.50).")
    _write_fused(modality, dir_tag, pairs, fused, r90["threshold"] if r90 else None)
    chosen = {
        "selection": "no_degrade_fbeta@0.50",
        "winner": winner,
        "weight_resnet": float(chosen_w),
        "weight_topoge": float(1.0 - chosen_w),
        "metrics_at_05": m05,
        "r90p": {
            "available": r90 is not None,
            **({"threshold": float(r90["threshold"]), "metrics": r90} if r90 is not None else {}),
        },
        "diagnostics": {
            "fbeta_candidates": {"fusion": fb_A, "resnet": fb_R, "topoge": fb_T},
            "join": {
                "paired_frames": len(pairset),
                "res_only": len(res_keyset - pairset),
                "topo_only": len(topo_keyset - pairset),
                "union": len(res_keyset | topo_keyset),
                "coverage": cover,
            },
            "beta": float(beta)
        }
    }
    _write_ckpt(modality, dir_tag, chosen)

def main():
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
        def flush(self):
            for f in self.files:
                try:
                    f.flush()
                except Exception:
                    pass
    log_path = "latefusion.txt"
    orig_stdout = sys.stdout
    try:
        with open(log_path, "w", encoding='utf-8', errors='replace') as log_fh:
            sys.stdout = Tee(sys.__stdout__, log_fh)
            print(f"[late_fusion] Logging print output to {log_path}")
            for dir_tag in ("uni", "bi"):
                args = SimpleNamespace(bidirectional=(dir_tag == "bi"))
                for modality in ("s1", "s2", "dual"):
                    try:
                        print(f"\n=== LATE FUSION: {modality.upper()} ({dir_tag}) ===")
                        run_one(modality, args)
                    except Exception as e:
                        print(f"[late_fusion] ERROR in {modality} ({dir_tag}): {e}")
    finally:
        sys.stdout = orig_stdout

if __name__ == "__main__":
    main()
