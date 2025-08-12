import os, sys, re, glob, math, json, atexit, argparse, random
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
HIDDEN_SIZE   = 256
BATCH_SIZE    = 8
NUM_EPOCHS    = 100
LR            = 1e-3
BETA_F        = math.sqrt(2)
GRID_SIZE     = 10
TOP_K_PD      = 200
DEFAULT_SEED  = 1599

RESNET_DIM    = 2048
TOPO_DIM      = 2 * (GRID_SIZE * GRID_SIZE)
FUSION_DIM    = RESNET_DIM + TOPO_DIM
MOD_EMBED_DIM = 8

def set_global_determinism(seed: int):
    """Force strict determinism; raise if PyTorch cannot comply.

    Notes:
      - Sets CUBLAS_WORKSPACE_CONFIG if absent (':4096:8').
      - Enables torch.use_deterministic_algorithms(True) and validates.
      - Disables cudnn.benchmark to avoid heuristic selection variance.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        raise RuntimeError(f"Deterministic algorithms could not be enabled: {e}. Set CUBLAS_WORKSPACE_CONFIG before launch or upgrade PyTorch.")
    # Sanity confirmation
    if hasattr(torch, 'are_deterministic_algorithms_enabled'):
        if not torch.are_deterministic_algorithms_enabled():
            raise RuntimeError("torch reports deterministic algorithms not enabled after request.")
    print("[Determinism] Strict deterministic mode active (seed=%d)." % seed)

class Tee:
    """Mirror stdout to both console and a file."""
    def __init__(self, fh):
        self.fh = fh
    def write(self, x):
        sys.__stdout__.write(x)
        if self.fh and not self.fh.closed:
            self.fh.write(x)
    def flush(self):
        if self.fh and not self.fh.closed:
            try: self.fh.flush()
            except: pass

NAME_RE_TIF = re.compile(
    r"(?P<seq>\d+)_(?P<idx_c>\d+)_([a-zA-Z0-9]+)_(?P<idx_m>\d+)_(?P<date>\d{8})_(?P<label>[01])\.tif$"
)
NAME_RE_NPY = re.compile(
    r"(?P<seq>\d+)_(?P<idx_c>\d+)_([A-Za-z0-9]+)_(?P<idx_m>\d+)_(?P<date>\d{8})_(?P<label>[01])\.npy$"
)

def parse_name(path: str, is_tif: bool = True):
    rx = NAME_RE_TIF if is_tif else NAME_RE_NPY
    m = rx.match(os.path.basename(path)); assert m, f"Bad filename: {path}"
    return m['seq'], int(m['idx_c']), int(m['label'])
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

class ResNetBackboneLite(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        from torchvision.models import resnet50
        net = resnet50(weights=None)
        net.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*list(net.children())[:-1])  # pool output 2048
    def forward(self, x):
        return self.features(x).flatten(1)

def load_encoder_from_finetuned_ckpt(backbone: ResNetBackboneLite, ckpt_path: str):
    """Load ONLY the encoder.* weights from a fine-tuned ResNet-GRU checkpoint."""
    sd = torch.load(ckpt_path, map_location='cpu')
    # state dict may be raw or under 'state_dict'
    if isinstance(sd, dict) and 'state_dict' in sd and not any(k.startswith('encoder.') for k in sd.keys()):
        sd = sd['state_dict']
    encoder_sd = {k.replace('encoder.', ''): v for k, v in sd.items() if k.startswith('encoder.')}
    missing, unexpected = backbone.load_state_dict(encoder_sd, strict=False)
    if missing:
        print(f"  [WARN] Missing backbone keys: {missing[:4]}{'...' if len(missing)>4 else ''}")
    if unexpected:
        print(f"  [WARN] Unexpected backbone keys: {unexpected[:4]}{'...' if len(unexpected)>4 else ''}")
    print(f"  Loaded encoder from {os.path.basename(ckpt_path)}")

def filter_top_k_per_dim(pd: np.ndarray, k: int) -> np.ndarray:
    if pd.size == 0: return pd
    out = []
    for dim in [0,1]:
        sub = pd[pd[:,0]==dim]
        if sub.size == 0: continue
        life = sub[:,2] - sub[:,1]
        if len(sub) > k:
            sub = sub[np.argsort(life)[-k:]]
        out.append(sub)
    return np.concatenate(out, axis=0) if out else pd

def compute_quartile_centers(train_pd_paths: List[str], grid_size: int, top_k: int, modality_id: int):
    """Compute centers and sigma2 separately for H0 and H1 for the given modality."""
    quantiles = list(np.linspace(0, 1, grid_size))
    births = {0: [], 1: []}
    deaths = {0: [], 1: []}
    for p in train_pd_paths:
        arr = np.load(p, allow_pickle=False)
        if arr.size == 0: continue
        if arr.ndim != 2 or arr.shape[1] < 3:
            continue
        pd = arr[:, :3]
        mask = (pd[:,0] <= 1) & np.isfinite(pd[:,1]) & np.isfinite(pd[:,2])
        if not mask.any(): continue
        pd = filter_top_k_per_dim(pd[mask], top_k)
        for dim in [0,1]:
            sub = pd[pd[:,0]==dim]
            if sub.size == 0: continue
            births[dim].append(sub[:,1]); deaths[dim].append(sub[:,2])
    centers = {}
    sigma2s = {}
    for dim in [0,1]:
        assert births[dim], f"No valid bars to init centers for modality {modality_id} H{dim}"
        all_b = np.concatenate(births[dim]); all_d = np.concatenate(deaths[dim])
        all_b = all_b[np.isfinite(all_b)]; all_d = all_d[np.isfinite(all_d)]
        qb = np.quantile(all_b, quantiles); qd = np.quantile(all_d, quantiles)
        std_b = np.std(all_b); std_d = np.std(all_d); scale = (std_b + std_d) / 2
        sigma = 0.1 * (scale if np.isfinite(scale) and scale>1e-8 else 1.0)
        gb, gd = np.meshgrid(qb, qd, indexing='ij')
        ctrs = np.stack([gb.ravel(), gd.ravel()], axis=1).astype(np.float32)
        centers[dim] = torch.tensor(ctrs, dtype=torch.float32)
        sigma2s[dim] = float(sigma**2)
    return centers, sigma2s

def quartile_embed(pd: np.ndarray, centers: Dict[int, torch.Tensor], sigma2s: Dict[int, float]) -> np.ndarray:
    feats = []
    for dim in [0,1]:
        ctrs = centers[dim]; s2 = sigma2s[dim]
        sub = pd[pd[:,0]==dim]
        if sub.size == 0:
            raise ValueError("No bars in PD for required homology dimension")
        births = torch.from_numpy(sub[:,1].astype(np.float32))
        deaths = torch.from_numpy(sub[:,2].astype(np.float32))
        life = (deaths - births).clamp(min=0.)
        pts = torch.stack([births, deaths], dim=1)
        diff = pts.unsqueeze(1) - ctrs.unsqueeze(0)
        dist2 = (diff * diff).sum(-1)
        rbf = torch.exp(-0.5 * dist2 / s2)
        weighted = (rbf * life.unsqueeze(1)).sum(0)
        feats.append(weighted)
    return torch.cat(feats, dim=0).numpy()

IMAGE_SIZE = 120  

def cache_resnet_features(modality: str, stacked_dir: str, ckpt_path: str, out_dir: str):
    """Extract per-frame 2048-d features using fine-tuned encoder weights."""
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, modality))
    from rasterio import open as rio_open
    in_ch = 2 if modality == 's1' else 10
    backbone = ResNetBackboneLite(in_ch).to(DEVICE).eval()
    load_encoder_from_finetuned_ckpt(backbone, ckpt_path)
    tif_paths = glob.glob(os.path.join(stacked_dir, "*.tif"))
    assert tif_paths, f"No .tif files in {stacked_dir}"
    for tif in tqdm(tif_paths, desc=f"ResNet feats {modality}", unit="img"):
        base = os.path.basename(tif).replace(".tif", "")
        out_npy = os.path.join(out_dir, modality, base + ".npy")
        if os.path.exists(out_npy):
            continue
        with rio_open(tif) as src:
            arr = src.read()
        t = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)  
        if t.shape[-1] != IMAGE_SIZE or t.shape[-2] != IMAGE_SIZE:
            t = F.interpolate(t, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        with torch.no_grad():
            feat = backbone(t).squeeze(0).detach().cpu().numpy()
        np.save(out_npy, feat.astype(np.float32))

def cache_topo_features(modality: str, cubical_dir: str, out_dir: str,
                        grid_size: int, top_k: int,
                        train_seq_ids_4digit: set):
    """Compute per-frame topo embeddings using centers from TRAIN split only."""
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, modality))
    npy_paths = glob.glob(os.path.join(cubical_dir, "*.npy"))
    assert npy_paths, f"No .npy PD files in {cubical_dir}"
    train_pd = [p for p in npy_paths if (len(parse_name(p, is_tif=False)[0])==4 and parse_name(p, is_tif=False)[0].isdigit())]
    centers, sigma2s = compute_quartile_centers(train_pd, grid_size, top_k, modality_id=(0 if modality=='s1' else 1))
    cdir = os.path.join(out_dir, modality, "centers")
    ensure_dir(cdir)
    torch.save({'centers': {k: v.cpu() for k,v in centers.items()}, 'sigma2s': sigma2s,
                'grid_size': grid_size, 'top_k': top_k}, os.path.join(cdir, f"qp_centers_{modality}.pt"))
    for p in tqdm(npy_paths, desc=f"Topo feats {modality}", unit="pd"):
        base = os.path.basename(p).replace(".npy", "")
        out_npy = os.path.join(out_dir, modality, base + ".npy")
        if os.path.exists(out_npy):
            continue
        pd = np.load(p, allow_pickle=False)
        pd = filter_top_k_per_dim(pd, top_k)
        feat = quartile_embed(pd, centers, sigma2s).astype(np.float32)
        np.save(out_npy, feat)

class FusionSeqDataset(Dataset):
    """Single-modality sequence dataset over cached feature dirs.
       Assumes both resnet and topo features exist for the same set of frames (by stem)."""
    def __init__(self, stacked_dir: str, feat_resnet_dir: str, feat_topo_dir: str, modality: str):
        self.modality = modality
        tif_paths = glob.glob(os.path.join(stacked_dir, "*.tif"))
        assert tif_paths, f"No .tif files in {stacked_dir}"
        by_seq: Dict[str, List[Tuple[int, str, int]]] = defaultdict(list)
        for tif in tif_paths:
            seq, idx_c, lbl = parse_name(tif, is_tif=True)
            stem = os.path.basename(tif).replace(".tif","")
            # require both features to exist
            fr = os.path.join(feat_resnet_dir, modality, stem + ".npy")
            ft = os.path.join(feat_topo_dir, modality, stem + ".npy")
            if not (os.path.exists(fr) and os.path.exists(ft)):
                continue
            by_seq[seq].append((idx_c, stem, lbl))
        self._seqs = []
        for seq_id, items in by_seq.items():
            items.sort(key=lambda t: t[0])
            self._seqs.append(items)
        assert self._seqs, f"No sequences with both features found for {modality}"
        self.feat_resnet_dir = os.path.join(feat_resnet_dir, modality)
        self.feat_topo_dir   = os.path.join(feat_topo_dir, modality)
    def __len__(self): return len(self._seqs)
    def __getitem__(self, idx):
        items = self._seqs[idx]
        rs = []; ts = []; labels = []
        for _, stem, lbl in items:
            r = np.load(os.path.join(self.feat_resnet_dir, stem + ".npy")).astype(np.float32)
            t = np.load(os.path.join(self.feat_topo_dir,   stem + ".npy")).astype(np.float32)
            rs.append(torch.from_numpy(r))
            ts.append(torch.from_numpy(t))
            labels.append(lbl)
        feats = torch.cat([torch.stack(rs), torch.stack(ts)], dim=1)  
        seq_id = items[0][1].split('_')[0]
        return feats, torch.tensor(labels, dtype=torch.long), seq_id

def collate_fusion_single(batch):
    feat_seqs, lbl_seqs, seq_ids = zip(*batch)
    lengths = [s.shape[0] for s in feat_seqs]
    T_max = max(lengths); B = len(batch); Fdim = feat_seqs[0].shape[1]
    feats = torch.zeros(B, T_max, Fdim, dtype=torch.float32)
    lbls  = torch.full((B, T_max), -100, dtype=torch.long)
    for i, (fs, ls) in enumerate(zip(feat_seqs, lbl_seqs)):
        T = fs.shape[0]
        feats[i, :T] = fs
        lbls[i, :T]  = ls
    return feats, lbls, torch.tensor(lengths), list(seq_ids)

class FusionCombinedDataset(Dataset):
    """Dual-modality dataset. Uses union of S1 and S2 frames, ordered by idx_c."""
    def __init__(self,
                 s1_stacked_dir: str, s2_stacked_dir: str,
                 feat_resnet_dir: str, feat_topo_dir: str):
        s1_tifs = glob.glob(os.path.join(s1_stacked_dir, "*.tif"))
        s2_tifs = glob.glob(os.path.join(s2_stacked_dir, "*.tif"))
        all_items = []
        for p in s1_tifs:
            seq, idx_c, lbl = parse_name(p, is_tif=True)
            stem = os.path.basename(p).replace(".tif","")
            fr = os.path.join(feat_resnet_dir, "s1", stem + ".npy")
            ft = os.path.join(feat_topo_dir,   "s1", stem + ".npy")
            if os.path.exists(fr) and os.path.exists(ft):
                all_items.append({'seq':seq,'idx_c':idx_c,'lbl':lbl,'mod':0,'stem':stem})
        for p in s2_tifs:
            seq, idx_c, lbl = parse_name(p, is_tif=True)
            stem = os.path.basename(p).replace(".tif","")
            fr = os.path.join(feat_resnet_dir, "s2", stem + ".npy")
            ft = os.path.join(feat_topo_dir,   "s2", stem + ".npy")
            if os.path.exists(fr) and os.path.exists(ft):
                all_items.append({'seq':seq,'idx_c':idx_c,'lbl':lbl,'mod':1,'stem':stem})
        buckets: Dict[str, List[Dict]] = defaultdict(list)
        for it in all_items:
            buckets[it['seq']].append(it)
        self._seqs = []
        for seq, items in buckets.items():
            items.sort(key=lambda x: x['idx_c'])
            self._seqs.append(items)
        assert self._seqs, "No dual sequences with both features"
        self.feat_resnet_dir = feat_resnet_dir
        self.feat_topo_dir   = feat_topo_dir
    def __len__(self): return len(self._seqs)
    def __getitem__(self, idx):
        items = self._seqs[idx]
        feats = []; labels = []; mods = []
        for it in items:
            mod = 's1' if it['mod']==0 else 's2'
            r = np.load(os.path.join(self.feat_resnet_dir, mod, it['stem'] + ".npy")).astype(np.float32)
            t = np.load(os.path.join(self.feat_topo_dir,   mod, it['stem'] + ".npy")).astype(np.float32)
            feats.append(torch.from_numpy(np.concatenate([r, t], axis=0)))
            labels.append(it['lbl']); mods.append(it['mod'])
        seq_id = items[0]['seq']
        return torch.stack(feats), torch.tensor(labels, dtype=torch.long), torch.tensor(mods, dtype=torch.long), seq_id

def collate_fusion_dual(batch):
    feat_seqs, lbl_seqs, mod_seqs, seq_ids = zip(*batch)
    lengths = [s.shape[0] for s in feat_seqs]
    T_max = max(lengths); B = len(batch); Fdim = feat_seqs[0].shape[1]
    feats = torch.zeros(B, T_max, Fdim, dtype=torch.float32)
    lbls  = torch.full((B, T_max), -100, dtype=torch.long)
    mods  = torch.full((B, T_max), -1, dtype=torch.long)
    for i, (fs, ls, ms) in enumerate(zip(feat_seqs, lbl_seqs, mod_seqs)):
        T = fs.shape[0]
        feats[i, :T] = fs; lbls[i, :T] = ls; mods[i, :T] = ms
    return feats, lbls, mods, torch.tensor(lengths), list(seq_ids)

class FusionGRU(nn.Module):
    def __init__(self, input_dim=FUSION_DIM, hidden=HIDDEN_SIZE, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_dim, hidden, batch_first=True, bidirectional=bidirectional)
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, 1)
    def forward(self, feats, lengths, return_padded=False):
        packed = pack_padded_sequence(feats, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out,_ = self.gru(packed); out,_ = pad_packed_sequence(out, batch_first=True)
        logits = self.head(out).squeeze(-1)
        if return_padded: return logits
        packed_logits = pack_padded_sequence(logits, lengths.cpu(), batch_first=True, enforce_sorted=False)
        return packed_logits.data, None

class DualFusionGRU(nn.Module):
    def __init__(self, input_dim=FUSION_DIM, hidden=HIDDEN_SIZE, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.mod_embed = nn.Embedding(2, MOD_EMBED_DIM)
        total_in = input_dim + MOD_EMBED_DIM
        self.gru = nn.GRU(total_in, hidden, batch_first=True, bidirectional=bidirectional)
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, 1)
    def forward(self, feats, lengths, mods, return_padded=False):
        feats = torch.cat([feats, self.mod_embed(mods.clamp(min=0))], dim=-1)
        packed = pack_padded_sequence(feats, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out,_ = self.gru(packed); out,_ = pad_packed_sequence(out, batch_first=True)
        logits = self.head(out).squeeze(-1)
        if return_padded: return logits
        packed_logits = pack_padded_sequence(logits, lengths.cpu(), batch_first=True, enforce_sorted=False)
        return packed_logits.data, None

def weighted_bce_loss(logits, packed_labels: PackedSequence, pos_weight: float):
    targets = packed_labels.data.float()
    pos_w = torch.tensor([pos_weight], device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_w)

def compute_confusion(preds, labels):
    TP = ((preds==1)&(labels==1)).sum().item()
    FP = ((preds==1)&(labels==0)).sum().item()
    TN = ((preds==0)&(labels==0)).sum().item()
    FN = ((preds==0)&(labels==1)).sum().item()
    return TP,FP,TN,FN

def compute_f1_score(TP,FP,FN):
    precision = TP/(TP+FP) if (TP+FP)>0 else 0.0
    recall    = TP/(TP+FN) if (TP+FN)>0 else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    return f1, precision, recall

def compute_fbeta(TP, FP, FN, beta: float):
    precision = TP/(TP+FP) if (TP+FP)>0 else 0.0
    recall    = TP/(TP+FN) if (TP+FN)>0 else 0.0
    if precision==0 and recall==0: return 0.0, precision, recall
    b2 = beta*beta
    fbeta = (1+b2)*precision*recall/(b2*precision+recall) if (b2*precision+recall)>0 else 0.0
    return fbeta, precision, recall

def calculate_class_weights_and_bias_from_buckets(buckets: List[List[Tuple[int,str,int]]]):
    total = pos = 0
    for seq_items in buckets:
        for _, _, lbl in seq_items:
            total += 1; pos += lbl
    if total==0:
        return 1.0, 0.0, 0.0
    pos_rate = pos/total; neg_rate = 1-pos_rate
    pos_rate = max(0.001, min(0.999, pos_rate))
    neg_rate = max(0.001, min(0.999, neg_rate))
    pos_weight = neg_rate/pos_rate
    log_odds = math.log(pos_rate/neg_rate)
    return pos_weight, log_odds, pos_rate

@torch.no_grad()
def eval_epoch(model, loader, pos_weight, is_dual=False):
    model.eval(); tot=hit=loss_sum=0; TP=FP=TN=FN=0
    if is_dual:
        for batch in tqdm(loader, desc='Val', leave=False):
            if len(batch)==5:
                feats,lbls,mods,lengths,_=batch
            else:
                feats,lbls,mods,lengths=batch
            feats,mods=feats.to(DEVICE),mods.to(DEVICE); lengths=lengths.to(DEVICE)
            packed_lbls=pack_padded_sequence(lbls.to(DEVICE), lengths.cpu(), batch_first=True, enforce_sorted=False)
            logits,_=model(feats,lengths,mods); loss=weighted_bce_loss(logits, packed_lbls, pos_weight)
            probs=torch.sigmoid(logits); preds=(probs>.5).long()
            loss_sum+=loss.item()*packed_lbls.data.numel(); tot+=packed_lbls.data.numel()
            hit+=(preds==packed_lbls.data.long()).sum().item()
            tP,fP,tN,fN=compute_confusion(preds, packed_lbls.data.long()); TP+=tP; FP+=fP; TN+=tN; FN+=fN
    else:
        for batch in tqdm(loader, desc='Val', leave=False):
            if len(batch)==4:
                feats,lbls,lengths,_=batch
            else:
                feats,lbls,lengths=batch
            feats=feats.to(DEVICE); lengths=lengths.to(DEVICE)
            packed_lbls=pack_padded_sequence(lbls.to(DEVICE), lengths.cpu(), batch_first=True, enforce_sorted=False)
            logits,_=model(feats,lengths); loss=weighted_bce_loss(logits, packed_lbls, pos_weight)
            probs=torch.sigmoid(logits); preds=(probs>.5).long()
            loss_sum+=loss.item()*packed_lbls.data.numel(); tot+=packed_lbls.data.numel()
            hit+=(preds==packed_lbls.data.long()).sum().item()
            tP,fP,tN,fN=compute_confusion(preds, packed_lbls.data.long()); TP+=tP; FP+=fP; TN+=tN; FN+=fN
    return loss_sum/max(1,tot), hit/max(1,tot), (TP,FP,TN,FN)

@torch.no_grad()
def validation_sweep(model, val_loader, is_dual=False, thresholds=None, modality: Optional[str]=None):
    if thresholds is None:
        thresholds = [i/100 for i in range(0,101)]
    print(f"\n=== Validation Sweep ({len(thresholds)} thresholds) ===")
    all_probs=[]; all_labels=[]; sequence_details=[]
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting predictions"):
            if is_dual:
                if len(batch)==5:
                    feats,lbls,mods,lengths,seq_ids = batch
                else:
                    feats,lbls,mods,lengths = batch; seq_ids = ['unknown']*feats.size(0)
                feats,mods = feats.to(DEVICE),mods.to(DEVICE); lengths=lengths.to(DEVICE)
                logits_full = model(feats,lengths,mods, return_padded=True); probs_full = torch.sigmoid(logits_full)
                packed_lbls=pack_padded_sequence(lbls.to(DEVICE), lengths.cpu(), batch_first=True, enforce_sorted=False)
                packed_probs=pack_padded_sequence(probs_full, lengths.cpu(), batch_first=True, enforce_sorted=False)
            else:
                if len(batch)==4:
                    feats,lbls,lengths,seq_ids = batch
                else:
                    feats,lbls,lengths = batch; seq_ids = ['unknown']*feats.size(0)
                feats=feats.to(DEVICE); lengths=lengths.to(DEVICE)
                logits_full = model(feats,lengths, return_padded=True); probs_full = torch.sigmoid(logits_full)
                packed_lbls=pack_padded_sequence(lbls.to(DEVICE), lengths.cpu(), batch_first=True, enforce_sorted=False)
                packed_probs=pack_padded_sequence(probs_full, lengths.cpu(), batch_first=True, enforce_sorted=False)
            all_probs.append(packed_probs.data.cpu().numpy()); all_labels.append(packed_lbls.data.cpu().numpy())
            for i, sid in enumerate(seq_ids):
                L = lengths[i].item()
                sequence_details.append({'real_seq_id': sid,
                                         'length': L,
                                         'labels': lbls[i, :L].tolist(),
                                         'probs': probs_full[i, :L].detach().cpu().tolist()})
    all_probs = np.concatenate(all_probs); all_labels = np.concatenate(all_labels)
    results = []
    for th in thresholds:
        preds = (all_probs >= th).astype(int)
        TP = ((preds==1)&(all_labels==1)).sum(); FP=((preds==1)&(all_labels==0)).sum()
        TN = ((preds==0)&(all_labels==0)).sum(); FN=((preds==0)&(all_labels==1)).sum()
        precision = TP/(TP+FP) if (TP+FP)>0 else 0.0
        recall    = TP/(TP+FN) if (TP+FN)>0 else 0.0
        f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        acc       = (TP+TN)/(TP+FP+TN+FN)
        results.append({'threshold': th, 'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': acc,
                        'tp': TP, 'fp': FP, 'tn': TN, 'fn': FN})
    best_f1 = max(results, key=lambda x: x['f1'])
    high_p = [r for r in results if r['precision'] >= 0.90]
    fallback_used = False
    if not high_p:
        if modality == 's1':
            best_hp = max(results, key=lambda x: (x['precision'], x['recall'], -x['threshold']))
            optimal_threshold = best_hp['threshold']; dm = best_hp; source = 'fallback_highest_precision'; fallback_used=True
            print(f"WARNING: No threshold reached >=90% precision for S1; fallback to highest precision: P={best_hp['precision']:.3f} R={best_hp['recall']:.3f} @ th={best_hp['threshold']:.2f}")
        else:
            optimal_threshold = best_f1['threshold']; dm = best_f1; source = 'fallback_best_f1'; fallback_used=True
            print(f"No threshold >=90% precision; fallback best F1 {best_f1['f1']:.3f} @ th={best_f1['threshold']:.2f}")
    else:
        best_r_90 = max(high_p, key=lambda x: x['recall'])
        optimal_threshold = best_r_90['threshold']; dm = best_r_90; source = 'recall_at_90p'
        print(f"Best Recall@90%P R={best_r_90['recall']:.3f} @ th={best_r_90['threshold']:.2f} (P={best_r_90['precision']:.3f})")
    print(f"(Best F1 {best_f1['f1']:.4f} @ {best_f1['threshold']:.2f} shown for reference only)")
    return {'all_results': results, 'best_f1': best_f1, 'optimal_threshold': optimal_threshold,
            'deploy_metrics': dm, 'threshold_source': source, 'sequence_details': sequence_details,
            'fallback_used': fallback_used}

def save_detailed_predictions_to_file(sweep_results, checkpoint_dir, tag, direction_tag: Optional[str] = None):
    """Write per-frame predictions in a compact TSV; filenames avoid the phrase 'latefusion'."""
    if direction_tag:
        predictions_file = os.path.join(checkpoint_dir, f"{tag}_{direction_tag}_fused_detailed_predictions.txt")
    else:
        predictions_file = os.path.join(checkpoint_dir, f"{tag}_fused_detailed_predictions.txt")

    opt_th = sweep_results.get('optimal_threshold', 0.5)
    seq_details = sweep_results.get('sequence_details', [])
    if not seq_details:
        print('[WARN] No sequence_details to save.')
        return
    with open(predictions_file, 'w') as f:
        f.write(
            f"# tag={tag} optimal_threshold={opt_th:.4f} source={sweep_results.get('threshold_source', 'n/a')}\n"
        )
        f.write("seq_id\tframe_idx\tprob\tlabel\tpred_05\tpred_opt\n")
        for sd in seq_details:
            sid = sd.get('real_seq_id', 'unknown')
            probs = sd['probs']
            labels = sd['labels']
            for j, (p, lbl) in enumerate(zip(probs, labels)):
                pred05 = 1 if p >= 0.5 else 0
                pred_opt = 1 if p >= opt_th else 0
                f.write(f"{sid}\t{j}\t{p:.6f}\t{int(lbl)}\t{pred05}\t{pred_opt}\n")
    print(f"Saved detailed predictions -> {predictions_file}")

def train_one_epoch(model, loader, optim, pos_weight, is_dual=False):
    model.train(); tot=hit=total_loss=0
    if is_dual:
        for batch in tqdm(loader, desc='Train', leave=False):
            if len(batch)==5:
                feats,lbls,mods,lengths,_=batch
            else:
                feats,lbls,mods,lengths=batch
            feats,mods=feats.to(DEVICE),mods.to(DEVICE); lengths=lengths.to(DEVICE)
            packed_lbls=pack_padded_sequence(lbls.to(DEVICE), lengths.cpu(), batch_first=True, enforce_sorted=False)
            logits,_=model(feats,lengths,mods); loss=weighted_bce_loss(logits, packed_lbls, pos_weight)
            optim.zero_grad(); loss.backward(); optim.step()
            total_loss += loss.item() * packed_lbls.data.numel()
            preds=(torch.sigmoid(logits)>.5).long(); hit+=(preds==packed_lbls.data.long()).sum().item(); tot+=packed_lbls.data.numel()
    else:
        for batch in tqdm(loader, desc='Train', leave=False):
            if len(batch)==4:
                feats,lbls,lengths,_=batch
            else:
                feats,lbls,lengths=batch
            feats=feats.to(DEVICE); lengths=lengths.to(DEVICE)
            packed_lbls=pack_padded_sequence(lbls.to(DEVICE), lengths.cpu(), batch_first=True, enforce_sorted=False)
            logits,_=model(feats,lengths); loss=weighted_bce_loss(logits, packed_lbls, pos_weight)
            optim.zero_grad(); loss.backward(); optim.step()
            total_loss += loss.item() * packed_lbls.data.numel()
            preds=(torch.sigmoid(logits)>.5).long(); hit+=(preds==packed_lbls.data.long()).sum().item(); tot+=packed_lbls.data.numel()
    return total_loss/max(1,tot), hit/max(1,tot)

def run_late_fusion_single(modality: str, stacked_dir: str, cubical_dir: str,
                           feat_r_dir: str, feat_t_dir: str,
                           ckpt_resnet: str,
                           epochs: int, bidirectional: bool):
    print(f"\n=== {modality.upper()} Fusion ===")
    if not (os.path.exists(os.path.join(feat_r_dir, modality)) and any(glob.glob(os.path.join(feat_r_dir, modality, '*.npy')))):
        print(f"Caching ResNet features for {modality} using {ckpt_resnet}")
        cache_resnet_features(modality, stacked_dir, ckpt_resnet, feat_r_dir)
    else:
        print(f"ResNet features present for {modality}; skipping cache")
    if not (os.path.exists(os.path.join(feat_t_dir, modality)) and any(glob.glob(os.path.join(feat_t_dir, modality, '*.npy')))):
        tif_paths = glob.glob(os.path.join(stacked_dir, "*.tif"))
        train_ids = set([parse_name(p, is_tif=True)[0] for p in tif_paths if (len(parse_name(p, is_tif=True)[0])==4 and parse_name(p, is_tif=True)[0].isdigit())])
        print(f"Caching topo embeddings for {modality} (train ids: {len(train_ids)})")
        cache_topo_features(modality, cubical_dir, feat_t_dir, GRID_SIZE, TOP_K_PD, train_ids)
    else:
        print(f"Topo features present for {modality}; skipping cache")
    ds = FusionSeqDataset(stacked_dir, feat_r_dir, feat_t_dir, modality)
    val_idx, train_idx = [], []
    for i, seq_items in enumerate(ds._seqs):
        seq_id = seq_items[0][1].split('_')[0]
        if len(seq_id)==4 and seq_id.isdigit(): train_idx.append(i)
        else: val_idx.append(i)
    print(f"  Split: {len(train_idx)} train / {len(val_idx)} val sequences")
    train_buckets = [ds._seqs[i] for i in train_idx]
    pos_weight, log_odds, pos_rate = calculate_class_weights_and_bias_from_buckets(train_buckets)
    g = torch.Generator(); g.manual_seed(DEFAULT_SEED)
    train_dl = DataLoader(torch.utils.data.Subset(ds, train_idx), BATCH_SIZE, shuffle=True,  collate_fn=collate_fusion_single, num_workers=2, worker_init_fn=lambda wid: np.random.seed(DEFAULT_SEED+wid), generator=g)
    val_dl   = DataLoader(torch.utils.data.Subset(ds, val_idx),   BATCH_SIZE, shuffle=False, collate_fn=collate_fusion_single, num_workers=2, worker_init_fn=lambda wid: np.random.seed(DEFAULT_SEED+wid), generator=g)
    model = FusionGRU(input_dim=FUSION_DIM, hidden=HIDDEN_SIZE, bidirectional=bidirectional).to(DEVICE)
    with torch.no_grad():
        model.head.bias.fill_(log_odds)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    direction_tag = 'bi' if bidirectional else 'uni'
    ckpt_name = f"fusion_{modality}_{direction_tag}.pt"
    best_score = -1.0; best_epoch = 0
    for epoch in range(1, epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, optim, pos_weight, is_dual=False)
        va_loss, va_acc, (TP,FP,TN,FN) = eval_epoch(model, val_dl, pos_weight, is_dual=False)
        f1,_,_ = compute_f1_score(TP,FP,FN)
        fbeta,prec,rec = compute_fbeta(TP,FP,FN,BETA_F)
        if fbeta > best_score:
            best_score = fbeta; best_epoch = epoch
            torch.save(model.state_dict(), ckpt_name)
        print(f"{modality.upper()} E{epoch:03d} train-loss={tr_loss:.4f} acc={tr_acc:.3f} | val-loss={va_loss:.4f} acc={va_acc:.3f} fβ={fbeta:.3f} (β={BETA_F:.2f}) f1={f1:.3f} P={prec:.3f} R={rec:.3f} | cm TP={TP} FP={FP} TN={TN} FN={FN}")
    print(f"Loading best epoch {best_epoch} model for sweep...")
    model.load_state_dict(torch.load(ckpt_name, map_location=DEVICE))
    sweep = validation_sweep(model, val_dl, is_dual=False, modality=modality)
    save_detailed_predictions_to_file(sweep, '.', f'{modality}', direction_tag)
    print(f"Best {modality.upper()} val-Fβ (β={BETA_F:.2f}): {best_score:.3f} @ epoch {best_epoch}")
    print(f"Deploy threshold ({sweep['threshold_source']}): {sweep['optimal_threshold']:.3f}")
    return sweep

def run_late_fusion_dual(s1_stacked_dir: str, s2_stacked_dir: str,
                         feat_r_dir: str, feat_t_dir: str,
                         epochs: int, bidirectional: bool):
    print(f"\n=== DUAL Fusion ===")
    ds = FusionCombinedDataset(s1_stacked_dir, s2_stacked_dir, feat_r_dir, feat_t_dir)
    val_idx, train_idx = [], []
    for i, items in enumerate(ds._seqs):
        seq_id = items[0]['seq']
        if len(seq_id)==4 and seq_id.isdigit(): train_idx.append(i)
        else: val_idx.append(i)
    print(f"  Split: {len(train_idx)} train / {len(val_idx)} val sequences")
    total = pos = 0
    for i in train_idx:
        for it in ds._seqs[i]:
            pos += it['lbl']; total += 1
    pos_rate = pos/max(1,total); neg_rate = 1 - pos_rate
    pos_rate = max(0.001, min(0.999, pos_rate)); neg_rate = max(0.001, min(0.999, neg_rate))
    pos_weight = neg_rate/pos_rate; log_odds = math.log(pos_rate/neg_rate)
    g = torch.Generator(); g.manual_seed(DEFAULT_SEED)
    train_dl = DataLoader(torch.utils.data.Subset(ds, train_idx), BATCH_SIZE, shuffle=True,  collate_fn=collate_fusion_dual, num_workers=2, worker_init_fn=lambda wid: np.random.seed(DEFAULT_SEED+wid), generator=g)
    val_dl   = DataLoader(torch.utils.data.Subset(ds, val_idx),   BATCH_SIZE, shuffle=False, collate_fn=collate_fusion_dual, num_workers=2, worker_init_fn=lambda wid: np.random.seed(DEFAULT_SEED+wid), generator=g)
    model = DualFusionGRU(input_dim=FUSION_DIM, hidden=HIDDEN_SIZE, bidirectional=bidirectional).to(DEVICE)
    with torch.no_grad():
        model.head.bias.fill_(log_odds)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    direction_tag = 'bi' if bidirectional else 'uni'
    ckpt_name = f"fusion_dual_{direction_tag}.pt"
    best_score = -1.0; best_epoch = 0
    for epoch in range(1, epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, optim, pos_weight, is_dual=True)
        va_loss, va_acc, (TP,FP,TN,FN) = eval_epoch(model, val_dl, pos_weight, is_dual=True)
        f1,_,_ = compute_f1_score(TP,FP,FN)
        fbeta,prec,rec = compute_fbeta(TP,FP,FN,BETA_F)
        if fbeta > best_score:
            best_score = fbeta; best_epoch = epoch
            torch.save(model.state_dict(), ckpt_name)
        print(f"DUAL E{epoch:03d} train-loss={tr_loss:.4f} acc={tr_acc:.3f} | val-loss={va_loss:.4f} acc={va_acc:.3f} fβ={fbeta:.3f} (β={BETA_F:.2f}) f1={f1:.3f} P={prec:.3f} R={rec:.3f} | cm TP={TP} FP={FP} TN={TN} FN={FN}")
    print(f"Loading best epoch {best_epoch} model for sweep...")
    model.load_state_dict(torch.load(ckpt_name, map_location=DEVICE))
    sweep = validation_sweep(model, val_dl, is_dual=True, modality='dual')
    save_detailed_predictions_to_file(sweep, '.', 'dual', direction_tag)
    print(f"Best DUAL val-Fβ (β={BETA_F:.2f}): {best_score:.3f} @ epoch {best_epoch}")
    print(f"Deploy threshold ({sweep['threshold_source']}): {sweep['optimal_threshold']:.3f}")
    return sweep

def main():
    global NUM_EPOCHS, BATCH_SIZE
    p = argparse.ArgumentParser(description="Sequence fusion over cached ResNet + topoGE features (fine-tuned encoder checkpoints)")
    p.add_argument('--s1-dir', default='s1', help='Root dir for S1 outputs (expects stacked/, gray/cubical/)')
    p.add_argument('--s2-dir', default='s2', help='Root dir for S2 outputs (expects stacked/, gray/cubical/)')
    p.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    p.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    p.add_argument('--seed', type=int, default=DEFAULT_SEED)
    p.add_argument('--bidirectional', action='store_true', default=False)
    p.add_argument('--overwrite-cache', action='store_true', default=False, help='(reserved)')
    args = p.parse_args()
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    set_global_determinism(args.seed)
    log_file_path = 'fusion_log.txt'
    log_fh = open(log_file_path, 'a', buffering=1); atexit.register(log_fh.close)
    sys.stdout = Tee(log_fh)
    print(f"Log file: {log_file_path}")
    print(f"Config: epochs={NUM_EPOCHS} batch={BATCH_SIZE} seed={args.seed} bidirectional={args.bidirectional} device={DEVICE} deterministic=TRUE")
    s1_stacked = os.path.join(args.s1_dir, 'stacked')
    s2_stacked = os.path.join(args.s2_dir, 'stacked')
    s1_cubical = os.path.join(args.s1_dir, 'gray', 'cubical')
    s2_cubical = os.path.join(args.s2_dir, 'gray', 'cubical')
    def pick_ckpt(mod):
        uni = f"resnet_gru_{mod}_finetune_uni.pt"
        bi  = f"resnet_gru_{mod}_finetune_bi.pt"
        if os.path.exists(uni): return uni
        if os.path.exists(bi):  return bi
        raise FileNotFoundError(f"Missing fine-tuned ResNet-GRU checkpoint for {mod}: {uni} or {bi}")
    ckpt_s1 = pick_ckpt('s1')
    ckpt_s2 = pick_ckpt('s2')
    feat_resnet_dir = os.path.join('features', 'resnet'); ensure_dir(feat_resnet_dir)
    feat_topo_dir   = os.path.join('features', 'topo');   ensure_dir(feat_topo_dir)
    if os.path.isdir(s1_stacked) and os.path.isdir(s1_cubical):
        run_late_fusion_single('s1', s1_stacked, s1_cubical, feat_resnet_dir, feat_topo_dir, ckpt_s1, NUM_EPOCHS, args.bidirectional)
    else:
        print("S1 directories missing; skipping S1 fusion")    
    if os.path.isdir(s2_stacked) and os.path.isdir(s2_cubical):
        run_late_fusion_single('s2', s2_stacked, s2_cubical, feat_resnet_dir, feat_topo_dir, ckpt_s2, NUM_EPOCHS, args.bidirectional)
    else:
        print("S2 directories missing; skipping S2 fusion")

    if os.path.isdir(s1_stacked) and os.path.isdir(s2_stacked):
        run_late_fusion_dual(s1_stacked, s2_stacked, feat_resnet_dir, feat_topo_dir, NUM_EPOCHS, args.bidirectional)
    else:
        print("Dual fusion skipped (need both S1 and S2 stacked dirs present)")

if __name__ == '__main__':
    main()