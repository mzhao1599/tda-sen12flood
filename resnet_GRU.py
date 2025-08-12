import glob, os, re, sys, numpy as np, torch, rasterio, atexit, argparse, math, multiprocessing, signal, json
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import random, math, os, sys, argparse, signal, atexit, glob, re, json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from safetensors.torch import load_file
#python resnet_GRU.py
#python resnet_GRU.py --bidirectional
HIDDEN_SIZE   = 256
BATCH_SIZE    = 8
NUM_EPOCHS    = 100
LR            = 0.001
LR_HEAD       = 0.001          
LR_BACKBONE   = 0.001          
LR_GRU_HEAD   = 0.01           
IMAGE_SIZE    = 120
DEVICE        = "cuda" 
DEBUG         = False         
PROBE_ONLY    = False          
SEQUENCE_PROBE = False          
DEBUG_EVERY = 34
BETA_F = math.sqrt(2) 
BIDIRECTIONAL = False 
DEFAULT_SEED = 1599
GLOBAL_SEED = DEFAULT_SEED

def set_global_determinism(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _seed_worker(worker_id: int):
    base = GLOBAL_SEED
    worker_seed = base + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

log_file = None
early_stop_requested = False

def signal_handler(signum, frame):
    global early_stop_requested
    print("\n\nCtrl+C detected! Finishing current epoch and performing validation sweep...")
    early_stop_requested = True

class Tee:
    def write(self, x):
        sys.__stdout__.write(x)
        if log_file:
            log_file.write(x)
    def flush(self):
        if log_file and not log_file.closed:
            try:
                log_file.flush()
            except ValueError:
                pass

class FileOnlyLogger:
    """Logger that only writes to file, not console"""
    def __init__(self, file_handle):
        self.file_handle = file_handle
    def write(self, x):
        if self.file_handle and not self.file_handle.closed:
            self.file_handle.write(x)
    def flush(self):
        if self.file_handle and not self.file_handle.closed:
            try:
                self.file_handle.flush()
            except ValueError:
                pass

NAME_RE = re.compile(
    r"(?P<seq>\d+)_(?P<idx_c>\d+)_([a-zA-Z0-9]+)_(?P<idx_m>\d+)_(?P<date>\d{8})_(?P<label>[01])\.(tif)$"
)
def parse_name(path: str):
    m = NAME_RE.match(os.path.basename(path))
    assert m, f"Bad filename: {path}"
    return m["seq"], int(m["idx_c"]), int(m["label"])

class SeqDataset(Dataset):
    """Loads variable-length sequences of Sentinel-1 or Sentinel-2 images.
    Enhanced to return original sequence id for efficient downstream bookkeeping.
    """
    def __init__(self, root_dir: str, modality: str):
        assert modality in {"s1", "s2"}
        self.modality = modality
        paths = glob.glob(os.path.join(root_dir, "*.tif"))
        assert paths, f"No .tif found in {root_dir}"
        buckets = {}
        for p in paths:
            seq, idx_c, lbl = parse_name(p)
            buckets.setdefault(seq, []).append((idx_c, p, lbl))
        self._seqs = []  
        for seq_id, items in buckets.items():
            items.sort(key=lambda t: t[0])
            self._seqs.append(items)
    def __len__(self):
        return len(self._seqs)
    def __getitem__(self, idx):
        items = self._seqs[idx]
        imgs = []
        labels = []
        for _, path, lbl in items:
            with rasterio.open(path) as src:
                img = src.read()
            img = torch.from_numpy(img)
            img = F.interpolate(img.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False).squeeze(0)
            imgs.append(img)
            labels.append(lbl)
        seq_id = os.path.basename(items[0][1]).split('_')[0]
        return torch.stack(imgs), torch.tensor(labels, dtype=torch.long), seq_id

def collate_pad(batch):
    """Collate for single-modality sequences, preserving sequence ids."""
    img_seqs, lbl_seqs, seq_ids = zip(*batch)
    lengths = [s.shape[0] for s in img_seqs]
    max_len = max(lengths)
    C, H, W = img_seqs[0].shape[1:]
    B = len(batch)
    imgs = torch.zeros(B, max_len, C, H, W)
    lbls = torch.full((B, max_len), -100, dtype=torch.long)
    for i, (img_seq, lbl_seq) in enumerate(zip(img_seqs, lbl_seqs)):
        T = img_seq.shape[0]
        imgs[i, :T] = img_seq
        lbls[i, :T] = lbl_seq
        assert torch.all((lbl_seq == 0) | (lbl_seq == 1)), f"Invalid labels in sequence {i}: {torch.unique(lbl_seq)}"
    for i, length in enumerate(lengths):
        active_labels = lbls[i, :length]
        padding_labels = lbls[i, length:]
        assert torch.all((active_labels == 0) | (active_labels == 1)), f"Active labels not binary in seq {i}: {torch.unique(active_labels)}"
        assert torch.all(padding_labels == -100), f"Padding not -100 in seq {i}: {torch.unique(padding_labels)}"
    return imgs, lbls, torch.tensor(lengths), list(seq_ids)

class CombinedSeqDataset(Dataset):
    """
    Expects:
      s1_dir/img/*.tif
      s2_dir/img/*.tif
      
    Combines images from both directories using filename parsing.
    Filename format: {seq_id}_{idx_c}_{sat}_{idx_m}_{date}_{flood}.tif
    The idx_c (second number) determines the order in the combined sequence.
    """
    def __init__(self, s1_dir: str, s2_dir: str):
        self.s1_dir = s1_dir
        self.s2_dir = s2_dir
        all_items = []
        if os.path.exists(s1_dir):
            for fname in os.listdir(s1_dir):
                if fname.endswith('.tif'):
                    parts = fname.replace('.tif', '').split('_')
                    if len(parts) >= 6:
                        seq_id, idx_c, sat, idx_m, date, flood = parts[:6]
                        all_items.append({
                            "seq_id": seq_id,
                            "idx_c": int(idx_c),
                            "sat": sat,
                            "idx_m": int(idx_m),
                            "date": date,
                            "fn": fname,
                            "lbl": int(flood),
                            "path": os.path.join(s1_dir, fname)
                        })
        if os.path.exists(s2_dir):
            for fname in os.listdir(s2_dir):
                if fname.endswith('.tif'):
                    parts = fname.replace('.tif', '').split('_')
                    if len(parts) >= 6:
                        seq_id, idx_c, sat, idx_m, date, flood = parts[:6]
                        all_items.append({
                            "seq_id": seq_id,
                            "idx_c": int(idx_c),
                            "sat": sat,
                            "idx_m": int(idx_m),
                            "date": date,
                            "fn": fname,
                            "lbl": int(flood),
                            "path": os.path.join(s2_dir, fname)
                        })
        seqs = {}
        for item in all_items:
            seq_id = item["seq_id"]
            seqs.setdefault(seq_id, []).append(item)
        self._seqs = []
        for seq_id, items in seqs.items():
            items.sort(key=lambda x: x["idx_c"])
            self._seqs.append(items)
    def __len__(self):
        return len(self._seqs)
    def __getitem__(self, i):
        items = self._seqs[i]
        imgs, labels, mods = [], [], []
        for it in items:
            path = it["path"]
            with rasterio.open(path) as src:
                arr = src.read()
            t = torch.from_numpy(arr)
            assert it["sat"] == "s1" and t.shape[0] == 2 or it["sat"] == "s2" and t.shape[0] == 10
            mods.append(0 if it["sat"] == "s1" else 1)
            t = F.interpolate(t.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False).squeeze(0)
            if it["sat"] == "s1":
                padding = torch.zeros(8, IMAGE_SIZE, IMAGE_SIZE)
                t = torch.cat([t, padding], dim=0)
            imgs.append(t)
            labels.append(it["lbl"])
        seq_id = str(items[0]["seq_id"])
        return torch.stack(imgs), torch.tensor(labels), torch.tensor(mods), seq_id

def collate_combined(batch):
    """Collate for dual-modality sequences, preserving sequence ids."""
    img_seqs, lbl_seqs, mod_seqs, seq_ids = zip(*batch)
    B = len(batch)
    T_max = max(seq.shape[0] for seq in img_seqs)
    C_max = 10
    _, _, H, W = img_seqs[0].shape
    imgs = torch.zeros(B, T_max, C_max, H, W)
    lbls = torch.full((B, T_max), -100, dtype=torch.long)
    mods = torch.full((B, T_max), -1, dtype=torch.long)
    lengths = []
    for i, (imgs_i, lbl_i, mod_i) in enumerate(zip(img_seqs, lbl_seqs, mod_seqs)):
        T = imgs_i.shape[0]
        imgs[i, :T, :imgs_i.shape[1]] = imgs_i
        lbls[i, :T] = lbl_i
        mods[i, :T] = mod_i
        lengths.append(T)
        assert torch.all((lbl_i == 0) | (lbl_i == 1)), f"Invalid labels in sequence {i}: {torch.unique(lbl_i)}"
        assert torch.all((mod_i == 0) | (mod_i == 1)), f"Invalid modalities in sequence {i}: {torch.unique(mod_i)}"
    for i, length in enumerate(lengths):
        active_labels = lbls[i, :length]
        padding_labels = lbls[i, length:]
        active_mods = mods[i, :length]
        padding_mods = mods[i, length:]
        assert torch.all((active_labels == 0) | (active_labels == 1)), f"Active labels not binary in seq {i}: {torch.unique(active_labels)}"
        assert torch.all(padding_labels == -100), f"Padding labels not -100 in seq {i}: {torch.unique(padding_labels)}"
        assert torch.all((active_mods == 0) | (active_mods == 1)), f"Active mods not binary in seq {i}: {torch.unique(active_mods)}"
        assert torch.all(padding_mods == -1), f"Padding mods not -1 in seq {i}: {torch.unique(padding_mods)}"
    return imgs, lbls, torch.tensor(lengths), mods, list(seq_ids)

class ResNetBackbone(nn.Module):
    def __init__(self, in_ch: int, pretrained_path: Optional[str] = None, load_weights: bool = True):
        super().__init__()
        from torchvision.models import resnet50
        net = resnet50(weights=None)
        net.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if load_weights and pretrained_path is not None:
            sd = load_file(pretrained_path)
            print(f"Loading BigEarthNet weights from {pretrained_path}")
            print(f"  Available keys: {len(sd.keys())} total")
            print(f"  Sample keys: {list(sd.keys())[:5]}...")
            new_sd = {}
            for k, v in sd.items():
                assert k.startswith("model.vision_encoder."), f"Unexpected key: {k}"
                if k.endswith("fc.weight") or k.endswith("fc.bias"):
                    continue
                new_sd[k.replace("model.vision_encoder.","")] = v
            print(f"  After filtering: {len(new_sd.keys())} keys for backbone")
            orig_w = new_sd.get("conv1.weight")
            if orig_w is None:
                raise KeyError("conv1.weight missing from checkpoint - aborting")
            if orig_w.size(1) != in_ch:
                raise ValueError(
                    f"Checkpoint conv1 has {orig_w.size(1)} channels, "
                    f"but model was built for {in_ch}. "
                    "Load the matching 2-channel or 10-channel safetensors."
                )
            conv1_weight = new_sd.pop("conv1.weight")
            with torch.no_grad():
                net.conv1.weight.copy_(conv1_weight)
            missing_keys, unexpected_keys = net.load_state_dict(new_sd, strict=False)
            print(f"  Loaded backbone weights:")
            print(f"    Missing keys: {len(missing_keys)} - {missing_keys[:3]}{'...' if len(missing_keys) > 3 else ''}")
            print(f"    Unexpected keys: {len(unexpected_keys)} - {unexpected_keys[:3]}{'...' if len(unexpected_keys) > 3 else ''}")
            critical_layers = ['layer1.0.conv1.weight', 'layer2.0.conv1.weight', 'layer3.0.conv1.weight', 'layer4.0.conv1.weight']
            missing_critical = [k for k in critical_layers if k in missing_keys]
            if missing_critical:
                print(f"  ERROR: Critical ResNet layers missing: {missing_critical}")
                raise RuntimeError(f"Failed to load critical backbone layers: {missing_critical}")
            else:
                print(f"  [OK] All critical ResNet layers loaded successfully")
        else:
            if not load_weights:
                print("[INFO] Skipping pretrained backbone load (will rely on checkpoint weights).")
        self.features = nn.Sequential(*list(net.children())[:-1])
    def forward(self, x):
        return self.features(x).flatten(1)

class ResNetGRU(nn.Module):
    def __init__(self, in_ch: int, pretrained_path: Optional[str] = None, log_odds_bias: float = 0.0, bidirectional: bool = False, load_backbone: bool = True):
        super().__init__()
        self.encoder = ResNetBackbone(in_ch, pretrained_path, load_weights=load_backbone)
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.gru     = nn.GRU(2048, HIDDEN_SIZE, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Identity()
        self.head    = nn.Linear(HIDDEN_SIZE * self.num_directions, 1)
        init_classification_bias(self, log_odds_bias)
    def forward(self, imgs: torch.Tensor, lengths: torch.Tensor):
        """Return logits over packed data PLUS the full PackedSequence for proper reconstruction.

        logits shape: (total_valid_frames,) corresponding to packed_out.data.
        Returned packed_out allows downstream code to recover original per-sequence ordering.
        """
        B, T, C, H, W = imgs.shape
        mask_frames = torch.arange(T, device=imgs.device).unsqueeze(0).expand(B, -1) < lengths.unsqueeze(1)
        flat_imgs  = imgs[mask_frames]
        feats_flat = self.encoder(flat_imgs)
        feats      = torch.zeros(B, T, 2048, device=imgs.device)
        feats[mask_frames] = feats_flat
        packed = pack_padded_sequence(feats, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        logits = self.head(packed_out.data).squeeze(-1)
        return logits, packed_out  
    def step(self, frame_img: torch.Tensor, h_prev=None):
        if self.bidirectional:
            raise NotImplementedError("Bidirectional GRU does not support step-wise inference")
        frame_img = frame_img.unsqueeze(0)
        feat = self.encoder(frame_img)
        out, h_next = self.gru(feat.unsqueeze(1), h_prev)
        prob = torch.sigmoid(self.head(out.squeeze(1)))[0].item()
        return prob, h_next

class DualResNetGRU(nn.Module):
    @torch.no_grad()
    def _encode(self, frame_img: torch.Tensor, modality: int) -> torch.Tensor:
        assert frame_img.shape[-2:] == (IMAGE_SIZE, IMAGE_SIZE), \
            f"Expected {IMAGE_SIZE}x{IMAGE_SIZE}, got {frame_img.shape[-2:]}"
        if modality == 0:
            return self.encoder_s1(frame_img)
        elif modality == 1:
            return self.encoder_s2(frame_img)
        else:
            raise ValueError(f"Invalid modality: {modality}")
    def __init__(self,
                 s1_ckpt: Optional[str] = None,
                 s2_ckpt: Optional[str] = None,
                 be_s1:   str = None,
                 be_s2:   str = None,
                 log_odds_bias: float = 0.0,
                 bidirectional: bool = False):
        super().__init__()
        assert be_s1 is not None, "BigEarthNet S1 weights (be_s1) are required"
        assert be_s2 is not None, "BigEarthNet S2 weights (be_s2) are required"
        self.encoder_s1 = ResNetBackbone(2, be_s1)
        self.encoder_s2 = ResNetBackbone(10, be_s2)
        self.mod_embed = nn.Embedding(2, 16)  
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.gru  = nn.GRU(2048 + 16, HIDDEN_SIZE, batch_first=True, bidirectional=bidirectional)
        self.head = nn.Linear(HIDDEN_SIZE * self.num_directions, 1)
        init_classification_bias(self, log_odds_bias)
        
        if s1_ckpt:
            assert os.path.isfile(s1_ckpt), f"S1 checkpoint not found: {s1_ckpt}"
            sd = torch.load(s1_ckpt, map_location="cpu")
            self.encoder_s1.load_state_dict(
                {k.replace("encoder.",""): v for k,v in sd.items() if k.startswith("encoder.")},
                strict=False)
            print(f"Loaded S1 encoder weights from {s1_ckpt}")
        
        if s2_ckpt:
            assert os.path.isfile(s2_ckpt), f"S2 checkpoint not found: {s2_ckpt}"
            sd = torch.load(s2_ckpt, map_location="cpu")
            self.encoder_s2.load_state_dict(
                {k.replace("encoder.",""): v for k,v in sd.items() if k.startswith("encoder.")},
                strict=False)
            print(f"Loaded S2 encoder weights from {s2_ckpt}")
    def forward(self, imgs, lengths, mods):
        B, T, C, H, W = imgs.shape
        mask_frames = torch.arange(T, device=imgs.device)\
                        .unsqueeze(0).expand(B, -1) < lengths.unsqueeze(1)
        flat_imgs = imgs[mask_frames]
        flat_mods = mods[mask_frames]
        feats_flat = torch.zeros(flat_imgs.size(0), 2048, device=imgs.device)
        idx_s1 = (flat_mods == 0).nonzero(as_tuple=True)[0]
        idx_s2 = (flat_mods == 1).nonzero(as_tuple=True)[0]
        if idx_s1.numel() > 0:
            x_s1 = flat_imgs[idx_s1, :2, :, :]
            feats_flat[idx_s1] = self.encoder_s1(x_s1)
        if idx_s2.numel() > 0:
            x_s2 = flat_imgs[idx_s2]
            feats_flat[idx_s2] = self.encoder_s2(x_s2)
        mod_embeddings = self.mod_embed(flat_mods)
        feats_flat = torch.cat([feats_flat, mod_embeddings], dim=-1)
        feats = torch.zeros(B, T, 2048 + 16, device=imgs.device)
        feats[mask_frames] = feats_flat
        packed = pack_padded_sequence(feats, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        logits = self.head(packed_out.data).squeeze(-1)
        return logits, packed_out.batch_sizes
    def step(self, frame_img: torch.Tensor, modality, h_prev=None):
        assert frame_img.ndim == 4, f"Expected 4D input, got {frame_img.ndim}D"
        feat = self._encode(frame_img, modality)
        mod_tensor = torch.tensor([modality], device=frame_img.device)
        mod_emb = self.mod_embed(mod_tensor)
        feat_with_mod = torch.cat([feat, mod_emb], dim=-1)
        out, h_next = self.gru(feat_with_mod.unsqueeze(1), h_prev)
        prob = torch.sigmoid(self.head(out.squeeze(1)))[0].item()
        return prob, h_next

def weighted_bce_loss(logits, packed_labels: PackedSequence, pos_weight: float = 2.5):
    """
    Weighted Binary Cross Entropy Loss for binary classification with packed sequences.
    Uses inverse-frequency weighting to handle class imbalance.
    
    Args:
        logits: Raw model outputs (before sigmoid)
        packed_labels: Packed sequence of true labels (0 or 1)
        pos_weight: Weight for positive class (calculated via inverse frequency)
    
    Returns:
        torch.Tensor: Scalar loss value
    """
    targets = packed_labels.data.float()
    assert torch.all((targets == 0) | (targets == 1)), \
        f"Invalid label values detected: {torch.unique(targets)}"
    pos_weight_tensor = torch.tensor([pos_weight], device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, targets,pos_weight=pos_weight_tensor,  reduction='mean')
    return loss

def calculate_class_weights_and_bias(dataset, indices):
    """
    Calculate inverse-frequency class weights and log-odds bias initialization.
    
    Args:
        dataset: The dataset (SeqDataset or CombinedSeqDataset)
        indices: List of sequence indices to analyze
        
    Returns:
        tuple: (pos_weight, log_odds_bias, positive_rate)
            - pos_weight: Weight for positive class (inverse frequency ratio)
            - log_odds_bias: Log-odds bias for initialization
            - positive_rate: Proportion of positive samples
    """
    total_frames = 0
    positive_frames = 0
    for idx in indices:
        if hasattr(dataset, '_seqs'):
            seq_items = dataset._seqs[idx]
            if isinstance(seq_items[0], dict):
                frame_labels = [item['lbl'] for item in seq_items]
            elif hasattr(seq_items[0], '__len__') and len(seq_items[0]) >= 3:
                frame_labels = [item[2] for item in seq_items]
            else:
                frame_labels = [item.get('lbl', 0) for item in seq_items]
        else:
            original_idx = dataset.indices[idx] if hasattr(dataset, 'indices') else idx
            base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
            seq_items = base_dataset._seqs[original_idx]
            if isinstance(seq_items[0], dict):
                frame_labels = [item['lbl'] for item in seq_items]
            elif hasattr(seq_items[0], '__len__') and len(seq_items[0]) >= 3:
                frame_labels = [item[2] for item in seq_items]
            else:
                frame_labels = [item.get('lbl', 0) for item in seq_items]
        total_frames += len(frame_labels)
        positive_frames += sum(frame_labels)
    if total_frames == 0:
        print("  WARNING: No frames found, using default weights")
        return 2.5, -2.197, 0.1  
    positive_rate = positive_frames / total_frames
    negative_rate = 1.0 - positive_rate
    positive_rate = max(0.001, min(0.999, positive_rate))
    negative_rate = max(0.001, min(0.999, negative_rate))
    pos_weight = negative_rate / positive_rate
    log_odds_bias = math.log(positive_rate / negative_rate)
    print(f"  Dataset statistics:")
    print(f"    Total frames: {total_frames}")
    print(f"    Positive frames: {positive_frames} ({positive_rate:.3f})")
    print(f"    Negative frames: {total_frames - positive_frames} ({negative_rate:.3f})")
    print(f"    Inverse-frequency pos_weight: {pos_weight:.3f}")
    print(f"    Log-odds bias: {log_odds_bias:.3f}")
    
    return pos_weight, log_odds_bias, positive_rate

def init_classification_bias(model, log_odds_bias: float):
    """
    Initialize classification head bias to the given log-odds value.
    
    Args:
        model: Model with .head linear layer
        log_odds_bias: Pre-calculated log-odds bias value
    """
    with torch.no_grad():
        if hasattr(model, 'head') and hasattr(model.head, 'bias'):
            model.head.bias.fill_(log_odds_bias)
            print(f"  [OK] Initialized head bias to {log_odds_bias:.3f}")
        else:
            print(f"  [WARN] Model has no .head.bias to initialize")

from typing import Optional

def validation_sweep(model, val_loader, is_dual=False, thresholds=None, modality: Optional[str]=None):
    """Embedding-style sweep choosing recall@90% precision threshold with fallbacks.

    Fallbacks:
      1) If no threshold reaches 0.90 precision and modality=='s1': highest precision (tie-break recall).
      2) Else fallback to best F1.
    """
    if thresholds is None:
        thresholds = [i/100 for i in range(0,101)]
    print(f"\n=== Validation Sweep ({len(thresholds)} thresholds) ===")
    all_probs=[]; all_labels=[]; sequence_details=[]
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting predictions"):
            seq_ids=None
            if is_dual:
                if len(batch)==5:
                    imgs,lbls,lengths,mods,seq_ids=batch
                else:
                    imgs,lbls,lengths,mods=batch
                imgs,lengths,mods=imgs.to(DEVICE),lengths.to(DEVICE),mods.to(DEVICE)
                packed_lbls=pack_padded_sequence(lbls.to(DEVICE), lengths.cpu(), batch_first=True, enforce_sorted=False)
                logits,packed_meta=model(imgs,lengths,mods)
            else:
                if len(batch)==4:
                    imgs,lbls,lengths,seq_ids=batch
                else:
                    imgs,lbls,lengths=batch
                imgs,lengths=imgs.to(DEVICE),lengths.to(DEVICE)
                packed_lbls=pack_padded_sequence(lbls.to(DEVICE), lengths.cpu(), batch_first=True, enforce_sorted=False)
                logits,packed_meta=model(imgs,lengths)
            probs=torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(packed_lbls.data.cpu().numpy())
            if isinstance(packed_meta, PackedSequence):
                packed_out = packed_meta
                out_padded, lengths_sorted = pad_packed_sequence(packed_out, batch_first=True)
                if packed_out.unsorted_indices is not None:
                    out_padded = out_padded[packed_out.unsorted_indices]
                B_cur = lengths.size(0)
                for i in range(B_cur):
                    L = lengths[i].item()
                    frame_outs = out_padded[i, :L, :]
                    frame_logits = model.head(frame_outs).squeeze(-1)
                    seq_probs = torch.sigmoid(frame_logits).cpu().numpy()
                    seq_lbls = lbls[i, :L].cpu().numpy()
                    sid = seq_ids[i] if seq_ids is not None else f"seq_{len(sequence_details)}"
                    sequence_details.append({'real_seq_id': sid, 'probs': seq_probs, 'labels': seq_lbls, 'length': L})
            else:
                start = 0; Bc = lengths.size(0)
                for i in range(Bc):
                    L = lengths[i].item()
                    seq_probs = probs[start:start+L].cpu().numpy(); seq_lbls = lbls[i, :L].cpu().numpy()
                    sid = seq_ids[i] if seq_ids is not None else f"seq_{len(sequence_details)}"
                    sequence_details.append({'real_seq_id': sid, 'probs': seq_probs, 'labels': seq_lbls, 'length': L})
    all_probs=np.array(all_probs); all_labels=np.array(all_labels)
    results=[]
    for th in thresholds:
        preds=(all_probs>=th).astype(int)
        TP=((preds==1)&(all_labels==1)).sum(); FP=((preds==1)&(all_labels==0)).sum(); TN=((preds==0)&(all_labels==0)).sum(); FN=((preds==0)&(all_labels==1)).sum()
        precision=TP/(TP+FP) if (TP+FP)>0 else 0.0
        recall=TP/(TP+FN) if (TP+FN)>0 else 0.0
        f1=2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        acc=(TP+TN)/(TP+FP+TN+FN) if (TP+FP+TN+FN)>0 else 0.0
        results.append({'threshold':th,'precision':precision,'recall':recall,'f1':f1,'accuracy':acc,'tp':TP,'fp':FP,'tn':TN,'fn':FN})
    best_f1=max(results, key=lambda x:x['f1'])
    high_p=[r for r in results if r['precision']>=0.90]
    fallback_used=False
    if not high_p:
        if modality=='s1':
            best_prec=max(results, key=lambda x:(x['precision'], x['recall']))
            deploy_metrics=best_prec; optimal_threshold=best_prec['threshold']; source='fallback_highest_precision'; fallback_used=True
            print(f"No th >=90% precision; S1 fallback highest precision {best_prec['precision']:.3f} @ th={best_prec['threshold']:.2f} (R={best_prec['recall']:.3f})")
        else:
            deploy_metrics=best_f1; optimal_threshold=best_f1['threshold']; source='fallback_best_f1'; fallback_used=True
            print(f"No th >=90% precision; fallback best F1 {best_f1['f1']:.3f} @ th={best_f1['threshold']:.2f}")
    else:
        best_r_90=max(high_p, key=lambda x:x['recall'])
        deploy_metrics=best_r_90; optimal_threshold=best_r_90['threshold']; source='recall_at_90p'
        print(f"Best Recall@90%P R={best_r_90['recall']:.3f} @ th={best_r_90['threshold']:.2f} (P={best_r_90['precision']:.3f})")
    print(f"(Best F1 {best_f1['f1']:.4f} @ {best_f1['threshold']:.2f} shown for reference)")
    if fallback_used:
        print(f"Threshold source: {source}")
    return {'all_results':results,'best_f1':best_f1,'best_accuracy':None,'best_recall_at_90p':(None if fallback_used else deploy_metrics),'optimal_threshold':optimal_threshold,'deploy_metrics':deploy_metrics,'sequence_details':sequence_details,'threshold_source':source,'fallback_used':fallback_used}

@torch.no_grad()
def rebuild_sequence_details(model, val_loader, is_dual: bool):
    """Recompute per-sequence probabilities in natural order (slow but reliable).

    Mirrors convert_resnet_predictions.rebuild_sequence_details to avoid any
    PackedSequence ordering ambiguity and to guarantee one-to-one alignment
    between probs and labels per sequence.
    """
    rebuilt = []
    model.eval()
    for batch in val_loader:
        if is_dual:
            if len(batch) == 5:
                imgs, lbls, lengths, mods, seq_ids = batch
            else:
                imgs, lbls, lengths, mods = batch
                seq_ids = [f"seq_{i}" for i in range(lengths.size(0))]
        else:
            if len(batch) == 4:
                imgs, lbls, lengths, seq_ids = batch
            else:
                imgs, lbls, lengths = batch
                seq_ids = [f"seq_{i}" for i in range(lengths.size(0))]
        B = lengths.size(0)
        for i in range(B):
            L = int(lengths[i].item())
            seq_imgs = imgs[i, :L].unsqueeze(0).to(DEVICE)
            seq_lengths = torch.tensor([L], device=DEVICE)
            if is_dual:
                seq_mods = mods[i, :L].unsqueeze(0).to(DEVICE)
                logits, _ = model(seq_imgs, seq_lengths, seq_mods)
            else:
                logits, _ = model(seq_imgs, seq_lengths)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = lbls[i, :L].cpu().numpy()
            sid = seq_ids[i]
            rebuilt.append({'real_seq_id': sid, 'probs': probs, 'labels': labels, 'length': L})
    return rebuilt

def calculate_positive_rate(dataset, indices):
    """
    Calculate the positive rate (flood frames) in the given dataset indices.
    
    Args:
        dataset: The dataset (SeqDataset or CombinedSeqDataset)
        indices: List of sequence indices to analyze
        
    Returns:
        float: Proportion of positive (flood) frames
    """
    _, _, positive_rate = calculate_class_weights_and_bias(dataset, indices)
    return positive_rate


def count_trainable_params(model):
    """Count trainable parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def inspect_checkpoint(checkpoint_path: str):
    """Inspect a BigEarthNet checkpoint to verify its contents"""
    print(f"\n=== Inspecting checkpoint: {checkpoint_path} ===")
    try:
        sd = load_file(checkpoint_path)
        print(f"Total keys: {len(sd.keys())}")

        conv1_keys = [k for k in sd.keys() if 'conv1' in k]
        print(f"Conv1 keys: {conv1_keys}")
        if 'model.vision_encoder.conv1.weight' in sd:
            conv1_weight = sd['model.vision_encoder.conv1.weight']
            print(f"Conv1 weight shape: {conv1_weight.shape}")
            print(f"Conv1 channels: {conv1_weight.shape[1]} (expected: 2 for S1, 10 for S2)")
        else:
            print("WARNING: Conv1 weight not found!")
        critical_prefixes = ['layer1', 'layer2', 'layer3', 'layer4']
        for prefix in critical_prefixes:
            matching_keys = [k for k in sd.keys() if prefix in k]
            print(f"{prefix} keys: {len(matching_keys)} found")
        print(f"Sample keys: {list(sd.keys())[:10]}")
    except Exception as e:
        print(f"ERROR inspecting checkpoint: {e}")
    print("=" * 50)

def filter_positive_sequences(dataset_indices, dataset):
    """Filter dataset indices to only include sequences with at least one flood frame"""
    positive_indices = []
    total_sequences = len(dataset_indices)
    for idx in dataset_indices:
        if hasattr(dataset, '_seqs'):
            seq_items = dataset._seqs[idx]
            has_flood = any(item[2] == 1 for item in seq_items)  
        else:
            original_idx = dataset.indices[idx] if hasattr(dataset, 'indices') else idx
            seq_items = dataset.dataset._seqs[original_idx] if hasattr(dataset, 'dataset') else dataset._seqs[original_idx]
            has_flood = any(item.get('lbl', 0) == 1 for item in seq_items)
        if has_flood:
            positive_indices.append(idx)
    print(f"  Filtered to positive sequences: {len(positive_indices)}/{total_sequences} ({100*len(positive_indices)/total_sequences:.1f}%)")
    return positive_indices

def compute_confusion(preds, labels):
    TP = ((preds == 1) & (labels == 1)).sum().item()
    TN = ((preds == 0) & (labels == 0)).sum().item()
    FP = ((preds == 1) & (labels == 0)).sum().item()
    FN = ((preds == 0) & (labels == 1)).sum().item()
    return TP, FP, TN, FN

def compute_f1_score(TP, FP, FN):
    """Compute F1 score from confusion matrix components"""
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1, precision, recall

def compute_fbeta(TP, FP, FN, beta: float):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if precision == 0 and recall == 0:
        return 0.0, precision, recall
    b2 = beta * beta
    denom = b2 * precision + recall
    fbeta = (1 + b2) * precision * recall / denom if denom > 0 else 0.0
    return fbeta, precision, recall

def setup_differential_optimizer(model, resnet_lr=0.0001, gru_lr=0.001, is_dual=False):
    """Setup optimizer with different learning rates for ResNet and GRU components"""
    if is_dual:
        resnet_params = list(model.encoder_s1.parameters()) + list(model.encoder_s2.parameters())
        gru_head_params = list(model.gru.parameters()) + list(model.head.parameters())
        if hasattr(model, 'mod_embed'):
            gru_head_params.extend(list(model.mod_embed.parameters()))
    else:
        resnet_params = list(model.encoder.parameters())
        gru_head_params = list(model.gru.parameters()) + list(model.head.parameters())
    param_groups = [
        {'params': resnet_params, 'lr': resnet_lr, 'name': 'resnet'},
        {'params': gru_head_params, 'lr': gru_lr, 'name': 'gru_head'}
    ]
    return torch.optim.Adam(param_groups)

def train_one_epoch(model, loader, optim, pos_weight: float = 2.5, is_dual=False):
    model.train(); tot=hit=total_loss=0
    step = 0
    if is_dual:
        for batch in tqdm(loader, desc="Train", leave=False):
            if len(batch) == 5:
                imgs, lbls, lengths, mods, _seq_ids = batch
            elif len(batch) == 4:
                imgs, lbls, lengths, mods = batch
            else:
                raise ValueError(f"Unexpected dual batch length {len(batch)}")
            imgs, lengths, mods = imgs.to(DEVICE), lengths.to(DEVICE), mods.to(DEVICE)
            packed_lbls = pack_padded_sequence(lbls.to(DEVICE), lengths.cpu(), batch_first=True, enforce_sorted=False)
            logits,_ = model(imgs,lengths,mods)
            loss = weighted_bce_loss(logits, packed_lbls, pos_weight)
            optim.zero_grad()
            loss.backward()
            optim.step()
            step += 1
            if DEBUG and step % DEBUG_EVERY == 0:
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    pos   = (probs > 0.5).float().mean().item()
                    gnorm = sum((p.grad.norm().item())**2
                                for p in model.parameters()
                                if p.grad is not None) ** 0.5
                    print(f"[gruv3-DUAL] step {step:05d}  "
                          f"loss={loss.item():.4f}  mean={probs.mean():.3f} std={probs.std():.3f}  "
                          f"pos={pos:.3f}  grad_norm={gnorm:.2f}")
            total_loss += loss.item()*packed_lbls.data.numel()   
            preds = (torch.sigmoid(logits)>.5).long()
            hit += (preds==packed_lbls.data.long()).sum().item()
            tot += len(packed_lbls.data) 
    else:
        for batch in tqdm(loader, desc="Train", leave=False):
            if len(batch) == 4:
                imgs, lbls, lengths, _seq_ids = batch
            else:
                imgs, lbls, lengths = batch
            imgs, lengths = imgs.to(DEVICE), lengths.to(DEVICE)
            packed_lbls = pack_padded_sequence(lbls.to(DEVICE), lengths.cpu(), batch_first=True, enforce_sorted=False)
            logits, _ = model(imgs, lengths)
            loss = weighted_bce_loss(logits, packed_lbls, pos_weight)
            optim.zero_grad()
            loss.backward() 
            optim.step()
            step += 1
            if DEBUG and step % DEBUG_EVERY == 0:
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    pos   = (probs > 0.5).float().mean().item()
                    gnorm = sum((p.grad.norm().item())**2
                                for p in model.parameters()
                                if p.grad is not None) ** 0.5
                    print(f"[gruv3-SINGLE] step {step:05d}  "
                          f"loss={loss.item():.4f}  mean={probs.mean():.3f} std={probs.std():.3f}  "
                          f"pos={pos:.3f}  grad_norm={gnorm:.2f}")
            total_loss += loss.item() * packed_lbls.data.numel()   
            preds = (torch.sigmoid(logits) > .5).long()
            hit  += (preds == packed_lbls.data.long()).sum().item()
            tot  += len(packed_lbls.data)
    return total_loss / tot, hit / tot
@torch.no_grad()
def eval_epoch(model, loader, pos_weight: float = 2.5, is_dual=False):
    model.eval(); tot=hit=running_loss=0
    TP_f=FP_f=TN_f=FN_f=0; TP_s=FP_s=TN_s=FN_s=0
    if is_dual:
        for batch in tqdm(loader, desc="Val", leave=False):
            if len(batch) == 5:
                imgs, lbls, lengths, mods, _seq_ids = batch
            elif len(batch) == 4:
                imgs, lbls, lengths, mods = batch
            else:
                raise ValueError(f"Unexpected dual batch length {len(batch)}")
            imgs, lengths, mods = imgs.to(DEVICE), lengths.to(DEVICE), mods.to(DEVICE)
            packed_lbls = pack_padded_sequence(lbls.to(DEVICE), lengths.cpu(), batch_first=True, enforce_sorted=False)
            logits,_ = model(imgs,lengths,mods)
            loss = weighted_bce_loss(logits, packed_lbls, pos_weight)
            running_loss += loss.item()*packed_lbls.data.numel()   
            preds = (torch.sigmoid(logits)>.5).long()
            labels = packed_lbls.data.long()
            hit += (preds==labels).sum().item(); tot += len(labels)
            tp,fp,tn,fn = compute_confusion(preds,labels)
            TP_f+=tp; FP_f+=fp; TN_f+=tn; FN_f+=fn
            start=0
            for i in range(imgs.size(0)):
                seq_len = lengths[i].item()
                seq_lbl = lbls[i,:seq_len].cpu().numpy()
                seq_pr  = (torch.sigmoid(logits[start:start+seq_len])>.5).cpu().numpy()
                has_lbl = np.any(seq_lbl==1); has_pr = np.any(seq_pr==1)
                if has_lbl and has_pr:   TP_s+=1
                elif not has_lbl and has_pr: FP_s+=1
                elif not has_lbl and not has_pr: TN_s+=1
                else: FN_s+=1
                start += seq_len
    else:
        for batch in tqdm(loader, desc="Val", leave=False):
            if len(batch) == 4:
                imgs, lbls, lengths, _seq_ids = batch
            else:
                imgs, lbls, lengths = batch
            imgs, lengths = imgs.to(DEVICE), lengths.to(DEVICE)
            packed_lbls = pack_padded_sequence(lbls.to(DEVICE), lengths.cpu(), batch_first=True, enforce_sorted=False)
            logits, _ = model(imgs, lengths)
            loss = weighted_bce_loss(logits, packed_lbls, pos_weight)
            running_loss += loss.item() * packed_lbls.data.numel()  
            preds = (torch.sigmoid(logits) > .5).long()
            labels = packed_lbls.data.long()
            hit  += (preds == labels).sum().item()
            tot  += len(labels)
            tp, fp, tn, fn = compute_confusion(preds, labels)
            TP_f += tp; FP_f += fp; TN_f += tn; FN_f += fn
            start = 0
            for i in range(imgs.size(0)):
                seq_len = lengths[i].item()
                seq_labels = lbls[i, :seq_len].cpu().numpy()
                seq_logits = logits[start:start+seq_len]
                seq_preds = (torch.sigmoid(seq_logits) > .5).cpu().numpy()
                has_pos_label = np.any(seq_labels == 1)
                has_pos_pred = np.any(seq_preds == 1)
                if has_pos_label and has_pos_pred:
                    TP_s += 1
                elif not has_pos_label and has_pos_pred:
                    FP_s += 1
                elif not has_pos_label and not has_pos_pred:
                    TN_s += 1
                elif has_pos_label and not has_pos_pred:
                    FN_s += 1
                start += seq_len
    return running_loss / tot, hit / tot, (TP_f, FP_f, TN_f, FN_f), (TP_s, FP_s, TN_s, FN_s)

def run_training(s1_dir: str, s2_dir: str, modality: str = "s1", epochs: int = NUM_EPOCHS, seed: int = DEFAULT_SEED, bidirectional: bool = False):
    """Deterministic unified training for single (s1/s2) or dual modality.

    Builds datasets + loaders, applies inverse-frequency weighting & bias init,
    trains with Fβ (β=√2) model selection, then performs validation threshold sweep.
    """
    print(f"\n=== {modality.upper()} Training (seed={seed}) ===")
    checkpoint_dir = "."
    num_workers = min(4, max(1, multiprocessing.cpu_count() // 4))
    g = torch.Generator(); g.manual_seed(seed)

    if modality == "dual":
        inspect_checkpoint("pretrained_models/resnet50-s1-v0.2.0/model.safetensors")
        inspect_checkpoint("pretrained_models/resnet50-s2-v0.2.0/model.safetensors")
        ds_full = CombinedSeqDataset(s1_dir, s2_dir)
        val_indices, train_indices = [], []
        for i, seq in enumerate(ds_full._seqs):
            seq_id = str(seq[0]['seq_id'])
            is_val = not (len(seq_id) == 4 and seq_id.isdigit())  
            (val_indices if is_val else train_indices).append(i)
        print(f"  Split: {len(train_indices)} train / {len(val_indices)} val sequences")
        print("  [TRAIN SPLIT STATS]")
        pos_weight, log_odds_bias, _ = calculate_class_weights_and_bias(ds_full, train_indices)
        print("  [VAL SPLIT STATS]")
        _val_pw, _val_bias, _ = calculate_class_weights_and_bias(ds_full, val_indices)
        train_set = torch.utils.data.Subset(ds_full, train_indices)
        val_set   = torch.utils.data.Subset(ds_full, val_indices)
        train_dl = DataLoader(train_set, BATCH_SIZE, shuffle=True,  collate_fn=collate_combined,
                              num_workers=num_workers, pin_memory=False, worker_init_fn=_seed_worker, generator=g)
        val_dl   = DataLoader(val_set,   BATCH_SIZE, shuffle=False, collate_fn=collate_combined,
                              num_workers=num_workers, pin_memory=False, worker_init_fn=_seed_worker, generator=g)
        model = DualResNetGRU(
            s1_ckpt="resnet_gru_s1_finetune.pt" if os.path.exists("resnet_gru_s1_finetune.pt") else None,
            s2_ckpt="resnet_gru_s2_finetune.pt" if os.path.exists("resnet_gru_s2_finetune.pt") else None,
            be_s1="pretrained_models/resnet50-s1-v0.2.0/model.safetensors",
            be_s2="pretrained_models/resnet50-s2-v0.2.0/model.safetensors",
            log_odds_bias=log_odds_bias,
            bidirectional=bidirectional).to(DEVICE)
        direction_tag = 'bi' if bidirectional else 'uni'
        checkpoint_name = f"dual_resnet_gru_best_mixed_{direction_tag}.pt"
        try:
            if os.path.exists(checkpoint_name):
                print(f"  Loading existing dual checkpoint: {checkpoint_name}")
                model.load_state_dict(torch.load(checkpoint_name, map_location=DEVICE))
        except Exception as e:
            print(f"  [WARN] Could not load dual checkpoint: {e}")
        is_dual = True
    else:
        channels = 2 if modality == 's1' else 10
        model_path = f"pretrained_models/resnet50-{modality}-v0.2.0/model.safetensors"
        inspect_checkpoint(model_path)
        data_dir = s1_dir if modality == 's1' else s2_dir
        ds_full = SeqDataset(data_dir, modality)
        padded_ids = []
        unpadded_ids = []
        other_ids = []
        for seq_items in ds_full._seqs:
            raw_id = os.path.basename(seq_items[0][1]).split('_')[0]
            if raw_id.isdigit() and len(raw_id) == 4:
                padded_ids.append(raw_id)
            elif raw_id.isdigit():
                unpadded_ids.append(raw_id)
            else:
                other_ids.append(raw_id)
        print(f"  Sequence ID diagnostics: padded4={len(padded_ids)} unpadded_numeric={len(unpadded_ids)} other={len(other_ids)} total={len(ds_full._seqs)}")
        val_indices, train_indices = [], []
        for i, seq in enumerate(ds_full._seqs):
            seq_first = os.path.basename(seq[0][1]).split('_')[0]
            is_val = not (len(seq_first) == 4 and seq_first.isdigit())
            (val_indices if is_val else train_indices).append(i)
        print(f"  Split: {len(train_indices)} train / {len(val_indices)} val sequences")
        print("  [TRAIN SPLIT STATS]")
        pos_weight, log_odds_bias, _ = calculate_class_weights_and_bias(ds_full, train_indices)
        print("  [VAL SPLIT STATS]")
        _val_pw, _val_bias, _ = calculate_class_weights_and_bias(ds_full, val_indices)
        train_set = torch.utils.data.Subset(ds_full, train_indices)
        val_set   = torch.utils.data.Subset(ds_full, val_indices)
        train_dl = DataLoader(train_set, BATCH_SIZE, shuffle=True,  collate_fn=collate_pad,
                              num_workers=num_workers, pin_memory=False, worker_init_fn=_seed_worker, generator=g)
        val_dl   = DataLoader(val_set,   BATCH_SIZE, shuffle=False, collate_fn=collate_pad,
                              num_workers=num_workers, pin_memory=False, worker_init_fn=_seed_worker, generator=g)
        model = ResNetGRU(channels, model_path, log_odds_bias, bidirectional=bidirectional).to(DEVICE)
        direction_tag = 'bi' if bidirectional else 'uni'
        checkpoint_name = f"resnet_gru_{modality}_finetune_{direction_tag}.pt"
        try:
            if os.path.exists(checkpoint_name):
                print(f"  Loading existing {modality.upper()} checkpoint: {checkpoint_name}")
                model.load_state_dict(torch.load(checkpoint_name, map_location=DEVICE))
        except Exception as e:
            print(f"  [WARN] Could not load checkpoint: {e}")
        is_dual = False
    optim = setup_differential_optimizer(model, resnet_lr=0.0001, gru_lr=0.001, is_dual=is_dual)
    best_fbeta = -1.0; best_f1_at_best_fbeta = 0.0; best_epoch = 0; patience = 20
    bias_val = 0.0
    if hasattr(model, 'head') and hasattr(model.head, 'bias') and model.head.bias is not None:
        try:
            bias_val = float(model.head.bias.detach().cpu().view(-1)[0].item())
        except Exception:
            pass
    print(f"  pos_weight={pos_weight:.3f}  bias={bias_val:.3f}")
    direction_tag = 'bi' if bidirectional else 'uni'
    for epoch in range(1, epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, optim, pos_weight, is_dual=is_dual)
        va_loss, va_acc, (TP_f, FP_f, TN_f, FN_f), (TP_s, FP_s, TN_s, FN_s) = eval_epoch(model, val_dl, pos_weight, is_dual=is_dual)
        frame_f1, _, _ = compute_f1_score(TP_f, FP_f, FN_f)
        frame_fbeta, frame_prec, frame_rec = compute_fbeta(TP_f, FP_f, FN_f, BETA_F)
        if frame_fbeta > best_fbeta:
            best_fbeta = frame_fbeta
            best_f1_at_best_fbeta = frame_f1
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_name)
        prefix = ('DUAL' if is_dual else modality.upper()) + ('-BI' if bidirectional else '')
        print(f"{prefix} E{epoch:03d} train-loss={tr_loss:.4f} acc={tr_acc:.3f} | "
              f"val-loss={va_loss:.4f} acc={va_acc:.3f} fβ={frame_fbeta:.3f} (β={BETA_F:.2f}) f1={frame_f1:.3f} P={frame_prec:.3f} R={frame_rec:.3f} | "
              f"val-cm: TP={TP_f} FP={FP_f} TN={TN_f} FN={FN_f}")
        if early_stop_requested:
            print("Early stop requested (Ctrl+C)")
            break
        if epoch - best_epoch >= patience:
            print(f"Early stopping (no Fβ improvement for {patience} epochs)")
            break
    print(f"\nLoading best epoch {best_epoch} (Fβ={best_fbeta:.4f}, F1={best_f1_at_best_fbeta:.4f}) for sweep...")
    model.load_state_dict(torch.load(checkpoint_name, map_location=DEVICE))
    sweep_results = validation_sweep(model, val_dl, is_dual=is_dual, modality=modality)
    try:
        rebuilt = rebuild_sequence_details(model, val_dl, is_dual=is_dual)
        sweep_results['sequence_details'] = rebuilt
    except Exception as e:
        print(f"[WARN] rebuild_sequence_details failed, using sweep reconstruction: {e}")
    save_detailed_predictions_to_file(sweep_results, checkpoint_dir, modality, direction_tag)
    print(f"Best {('DUAL' if is_dual else modality.upper())}{'-BI' if bidirectional else ''} val-Fβ: {best_fbeta:.3f} (F1={best_f1_at_best_fbeta:.3f}) @ epoch {best_epoch}")
    dm = sweep_results['deploy_metrics']; b2 = BETA_F*BETA_F
    deploy_fbeta = (1+b2)*dm['precision']*dm['recall']/(b2*dm['precision']+dm['recall']) if (b2*dm['precision']+dm['recall'])>0 else 0.0
    print("=== DEPLOYMENT RECOMMENDATION ===")
    print(f"Threshold {sweep_results['optimal_threshold']:.3f} (source={sweep_results.get('threshold_source','n/a')})")
    print(f"Expected Fβ={deploy_fbeta:.4f} F1={dm['f1']:.4f} P={dm['precision']:.4f} R={dm['recall']:.4f}")
    return sweep_results

def save_detailed_predictions_to_file(sweep_results, checkpoint_dir, modality, direction_tag: Optional[str] = None):
    """Write per-frame probabilities in topoGE-style format for late fusion consistency.

    Format:
      Optimal threshold: <val>
      Format: frame_idx prob label pred@0.50 pred@opt

      SEQ <seq_id> len=<L> positives=<P>
        000 0.1234 0 0 0
        ...
    """
    if direction_tag:
        predictions_file = os.path.join(checkpoint_dir, f"{modality}_{direction_tag}_resnet_detailed_predictions.txt")
    else:
        predictions_file = os.path.join(checkpoint_dir, f"{modality}_resnet_detailed_predictions.txt")
    opt_th = sweep_results.get('optimal_threshold', 0.5)
    seq_details = sweep_results.get('sequence_details', [])
    if not seq_details:
        print('[WARN] No sequence_details to save.')
        return
    for sd in seq_details:
        if 'length' not in sd:
            probs = sd.get('probs', [])
            sd['length'] = len(probs)
    def _seq_key(d):
        try:
            return int(d.get('real_seq_id', ''))
        except Exception:
            return 10**9
    try:
        with open(predictions_file, 'w') as f:
            f.write(f"Optimal threshold: {opt_th:.4f}\n")
            f.write("Format: frame_idx prob label pred@0.50 pred@opt\n\n")
            for sd in sorted(seq_details, key=_seq_key):
                sid = sd.get('real_seq_id','unknown')
                probs = sd['probs']; labels = sd['labels']
                L = sd.get('length', len(probs)); positives = int(sum(labels))
                f.write(f"SEQ {sid} len={L} positives={positives}\n")
                for j,(p,lbl) in enumerate(zip(probs,labels)):
                    pred05 = 1 if p >= 0.5 else 0
                    predOpt = 1 if p >= opt_th else 0
                    f.write(f"  {j:03d} {p:.4f} {int(lbl)} {pred05} {predOpt}\n")
                f.write("\n")
        print(f"Saved detailed predictions (topoGE format) -> {predictions_file}")
    except Exception as e:
        print(f"[ERROR] Failed writing predictions: {e}")

class ProbeSeqDataset(Dataset):
    """Dataset for linear probe that uses original JSON flood labels instead of modified labels."""
    def __init__(self, root_dir: str, modality: str, json_path: str):
        assert modality in {"s1", "s2"}
        self.modality = modality
        self.json_path = json_path
        with open(json_path, 'r') as f:
            self.json_data = json.load(f)
        paths = glob.glob(os.path.join(root_dir, "*.tif"))
        assert paths, f"No .tif found in {root_dir}"
        buckets = {}
        for p in paths:
            seq, idx_c, _ = parse_name(p)  
            original_label = self._get_original_label(seq, p)
            buckets.setdefault(seq, []).append((idx_c, p, original_label))
        self._seqs = []
        for seq_id, items in buckets.items():
            items.sort(key=lambda t: t[0])
            self._seqs.append(items)
    def _get_original_label(self, seq: str, path: str) -> int:
        """Get original flood label from JSON metadata."""
        filename = os.path.basename(path)
        parts = filename.replace('.tif', '').split('_')
        if len(parts) >= 5:
            date_str = parts[4]  
            if len(date_str) == 8:
                iso_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                
                seq_data = self.json_data.get(seq, {})
                for entry_data in seq_data.values():
                    if isinstance(entry_data, dict) and entry_data.get("date") == iso_date:
                        return int(bool(entry_data.get("FLOODING", False)))
        return 0
    def __len__(self):
        return len(self._seqs)
    def __getitem__(self, idx):
        items = self._seqs[idx]
        imgs, labels = [], []
        for _, path, lbl in items:
            with rasterio.open(path) as src:
                img = src.read()
            img = torch.from_numpy(img)
            img = F.interpolate(img.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False).squeeze(0)
            imgs.append(img)
            labels.append(lbl)
        seq_id = os.path.basename(items[0][1]).split('_')[0]
        return torch.stack(imgs), torch.tensor(labels, dtype=torch.long), seq_id

def has_flood_in_sequence(seq_items, dataset_type="single"):
    """Check if a sequence has at least one flood frame"""
    if dataset_type == "single":
        return any(item[2] == 1 for item in seq_items)
    else:
        return any(item.get('lbl', 0) == 1 for item in seq_items)

def run_linear_probe(root: str, modality: str = "s1", probe_type: str = "frame"):
    """Unified linear probe function using original JSON flood labels."""
    print(f"\n=== {modality.upper()} {probe_type.title()}-Level Linear Probe ===")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    except ImportError:
        print("ERROR: sklearn not available. Install with: pip install scikit-learn")
        return
    json_file = f"S{1 if modality == 's1' else 2}list.json"
    json_path = os.path.join("/home/erhai/Desktop/Flood", json_file)
    if not os.path.exists(json_path):
        print(f"ERROR: JSON file not found: {json_path}")
        return
    print(f"Using original flood labels from: {json_path}")
    
    def _get_single_features(mod_root, mod_name):
        channels = 2 if mod_name == "s1" else 10
        model_path = f"pretrained_models/resnet50-{mod_name}-v0.2.0/model.safetensors"
        ds_full = ProbeSeqDataset(mod_root, mod_name, json_path)
        val_indices, train_indices = [], []
        for i, seq in enumerate(ds_full._seqs):
            seq_id = os.path.basename(seq[0][1]).split('_')[0]
            (val_indices if len(seq_id)<4 and seq_id.isdigit() else train_indices).append(i)
        train_set = torch.utils.data.Subset(ds_full, train_indices)
        val_set = torch.utils.data.Subset(ds_full, val_indices)
        train_dl = DataLoader(train_set, batch_size=16, shuffle=False, collate_fn=collate_pad, num_workers=4)
        val_dl = DataLoader(val_set, batch_size=16, shuffle=False, collate_fn=collate_pad, num_workers=4)
        encoder = ResNetBackbone(channels, model_path).to(DEVICE)
        encoder.eval()
        train_features, train_labels = _extract_features(train_dl, encoder, mod_name, probe_type, "train")
        val_features, val_labels = _extract_features(val_dl, encoder, mod_name, probe_type, "val")
        return train_features, train_labels, val_features, val_labels
    if modality == "dual":
        train_f_s1, train_l_s1, val_f_s1, val_l_s1 = _get_single_features(os.path.join(root, "s1/img"), "s1")
        train_f_s2, train_l_s2, val_f_s2, val_l_s2 = _get_single_features(os.path.join(root, "s2/img"), "s2")
        X_train = torch.stack(train_f_s1 + train_f_s2).numpy()
        y_train = np.array(train_l_s1 + train_l_s2)
        X_test = torch.stack(val_f_s1 + val_f_s2).numpy()
        y_test = np.array(val_l_s1 + val_l_s2)
    else:
        train_features, train_labels, val_features, val_labels = _get_single_features(root, modality)
        X_train = torch.stack(train_features).numpy()
        y_train = np.array(train_labels)
        X_test = torch.stack(val_features).numpy()
        y_test = np.array(val_labels)
    print(f"Original JSON labels - Train: {np.sum(y_train)} positive / {len(y_train)} total ({np.mean(y_train):.3f})")
    print(f"Original JSON labels - Test:  {np.sum(y_test)} positive / {len(y_test)} total ({np.mean(y_test):.3f})")
    lr = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
    lr.fit(X_train, y_train)
    train_preds = lr.predict(X_train)
    test_preds = lr.predict(X_test)
    train_probs = lr.predict_proba(X_train)[:, 1]
    test_probs = lr.predict_proba(X_test)[:, 1]
    train_auc = roc_auc_score(y_train, train_probs)
    test_auc = roc_auc_score(y_test, test_probs)
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    train_cm = confusion_matrix(y_train, train_preds)
    test_cm = confusion_matrix(y_test, test_preds)
    train_f1 = f1_score(y_train, train_preds)
    test_f1 = f1_score(y_test, test_preds)
    train_precision = precision_score(y_train, train_preds)
    test_precision = precision_score(y_test, test_preds)
    train_recall = recall_score(y_train, train_preds)
    test_recall = recall_score(y_test, test_preds)
    print(f"\n=== {modality.upper()} {probe_type.title()}-Level Probe Results (Original JSON Labels) ===")
    print(f"Train AUC: {train_auc:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
    print(f"Test AUC:  {test_auc:.4f} | Test Acc:  {test_acc:.4f} | Test F1:  {test_f1:.4f}")
    print(f"\nTrain Metrics: P={train_precision:.4f} R={train_recall:.4f}")
    print(f"Test Metrics:  P={test_precision:.4f} R={test_recall:.4f}")
    print(f"\nTrain Confusion Matrix:")
    print(f"  TN={train_cm[0,0]} FP={train_cm[0,1]}")
    print(f"  FN={train_cm[1,0]} TP={train_cm[1,1]}")
    print(f"\nTest Confusion Matrix:")
    print(f"  TN={test_cm[0,0]} FP={test_cm[0,1]}")
    print(f"  FN={test_cm[1,0]} TP={test_cm[1,1]}")
    return test_auc, test_acc, test_f1

def _extract_features(dataloader, encoder, modality, probe_type, split_name):
    """Extract features for probe based on type. Supports batches with or without seq_ids."""
    features, labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{modality.upper()} {split_name} {probe_type}s"):
            if len(batch) == 4:
                imgs, lbls, lengths, _ = batch
            else:
                imgs, lbls, lengths = batch
            B, T, C, H, W = imgs.shape
            imgs = imgs.to(DEVICE)
            flat_imgs = imgs.view(-1, C, H, W)
            flat_feats = encoder(flat_imgs)
            feats = flat_feats.view(B, T, -1)
            for i in range(B):
                seq_len = lengths[i].item()
                valid_feats = feats[i, :seq_len]
                valid_labels = lbls[i, :seq_len]
                if probe_type == "frame":
                    for j in range(seq_len):
                        features.append(valid_feats[j].cpu())
                        labels.append(valid_labels[j].item())
                else:
                    seq_feat = valid_feats.mean(dim=0)
                    seq_label = int(torch.any(valid_labels == 1))
                    features.append(seq_feat.cpu())
                    labels.append(seq_label)
    return features, labels

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='ResNet-GRU flood sequence classifier')
    parser.add_argument('--s1-dir', default='s1/stacked', help='Directory containing S1 .tif files')
    parser.add_argument('--s2-dir', default='s2/stacked', help='Directory containing S2 .tif files')
    parser.add_argument('--debug', action='store_true', default=False,
                       help='Enable debug output (default: False)')
    parser.add_argument('--probe', action='store_true', default=False,
                       help='Run ONLY linear probe sanity check on BigEarthNet features (default: False, normal run does probes + training)')
    parser.add_argument('--sequence-probe', action='store_true', default=False,
                       help='Run sequence-level linear probe (mean-pooled features) instead of frame-level (default: False)')
    parser.add_argument('--mode', choices=['full','dual','s1','s2'], default='full',
                       help='full: probes + s1 + s2 + dual (default); dual: only dual training; s1: only s1 training; s2: only s2 training')
    parser.add_argument('--skip-probes', action='store_true', default=False,
                       help='Skip linear probe sanity checks in full or selected mode')
    parser.add_argument('--bidirectional', action='store_true', default=False,
                       help='Use bidirectional GRU (concatenate forward/backward hidden states)')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                       help=f'Number of training epochs (default: {NUM_EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=LR,
                       help=f'Learning rate (default: {LR})')
    parser.add_argument('--lr-head', type=float, default=LR_HEAD,
                       help=f'Learning rate for head-only training (default: {LR_HEAD})')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Global random seed for full determinism')
    return parser.parse_args()

def main():
    global DEBUG, PROBE_ONLY, SEQUENCE_PROBE, NUM_EPOCHS, BATCH_SIZE, LR, LR_HEAD, log_file
    signal.signal(signal.SIGINT, signal_handler)
    args = parse_args()
    DEBUG = args.debug
    PROBE_ONLY = args.probe
    SEQUENCE_PROBE = args.sequence_probe
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr
    LR_HEAD = args.lr_head
    global GLOBAL_SEED
    GLOBAL_SEED = args.seed
    set_global_determinism(GLOBAL_SEED)
    print(f"Determinism: seed={GLOBAL_SEED}")
    s1_dir = args.s1_dir
    s2_dir = args.s2_dir
    s1_exists = os.path.exists(s1_dir) and os.path.isdir(s1_dir)
    s2_exists = os.path.exists(s2_dir) and os.path.isdir(s2_dir)
    if not s1_exists and not s2_exists:
        print(f"ERROR: Neither S1 directory ({s1_dir}) nor S2 directory ({s2_dir}) exists!")
        return
    log_file_path = "resnet_gru.txt"
    log_file = open(log_file_path, "a", buffering=1)  
    atexit.register(log_file.close)
    sys.stdout = Tee() 
    print(f"Log file: {log_file_path}")
    if PROBE_ONLY:
        print("=== LINEAR PROBE SANITY CHECK MODE ===")
        probe_type = "sequence" if SEQUENCE_PROBE else "frame"
        if s1_exists:
            s1_stacked = os.path.join(s1_dir, "stacked")
            s1_probe_dir = s1_stacked if os.path.exists(s1_stacked) else s1_dir
            
            print(f"S1 probe directory: {s1_probe_dir}")
            try:
                print(f"\n=== Running S1 {probe_type}-level probe ===")
                run_linear_probe(s1_probe_dir, "s1", probe_type)
            except Exception as e:
                print(f"S1 probe failed: {e}")
        else:
            print("S1 directory not found - skipping S1 probe")  
        if s2_exists:
            s2_stacked = os.path.join(s2_dir, "stacked")
            s2_probe_dir = s2_stacked if os.path.exists(s2_stacked) else s2_dir
            print(f"S2 probe directory: {s2_probe_dir}")
            try:
                print(f"\n=== Running S2 {probe_type}-level probe ===")
                run_linear_probe(s2_probe_dir, "s2", probe_type)
            except Exception as e:
                print(f"S2 probe failed: {e}")
        else:
            print("S2 directory not found - skipping S2 probe")
            
        print("\n=== LINEAR PROBE COMPLETED ===")
        return
    print(f"Training configuration:")
    print(f"  S1 directory: {s1_dir} ({'exists' if s1_exists else 'NOT FOUND'})")
    print(f"  S2 directory: {s2_dir} ({'exists' if s2_exists else 'NOT FOUND'})")
    print(f"  Debug: {DEBUG}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LR}")
    print(f"  Head learning rate: {LR_HEAD}")
    print(f"  Bidirectional GRU: {args.bidirectional}")
    mode = args.mode
    skip_probes = args.skip_probes
    def maybe_run_probes(which: str):
        if skip_probes:
            print(f"Skipping probes (--skip-probes) for {which} mode")
            return
        probe_type = "sequence" if SEQUENCE_PROBE else "frame"
        if which in ("s1","full") and s1_exists:
            try:
                print(f"\n=== Running S1 {probe_type}-level probe ===")
                run_linear_probe(s1_dir, "s1", probe_type)
            except Exception as e:
                print(f"S1 probe failed: {e}")
        if which in ("s2","full") and s2_exists:
            try:
                print(f"\n=== Running S2 {probe_type}-level probe ===")
                run_linear_probe(s2_dir, "s2", probe_type)
            except Exception as e:
                print(f"S2 probe failed: {e}")
    global early_stop_requested
    if mode == 'dual':
        if not (s1_exists and s2_exists):
            print("ERROR: dual mode requires both S1 and S2 directories present")
            return
        print("\n=== DUAL-ONLY MODE ===")
        maybe_run_probes("full")  
        try:
            run_training(s1_dir, s2_dir, "dual", epochs=NUM_EPOCHS, bidirectional=args.bidirectional)
        except Exception as e:
            print(f"Dual training failed: {e}")
        return
    elif mode == 's1':
        if not s1_exists:
            print("ERROR: s1 mode selected but S1 directory missing")
            return
        print("\n=== S1-ONLY MODE ===")
        maybe_run_probes("s1")
        try:
            run_training(s1_dir, s2_dir, "s1", epochs=NUM_EPOCHS, bidirectional=args.bidirectional)
        except Exception as e:
            print(f"S1 training failed: {e}")
        return
    elif mode == 's2':
        if not s2_exists:
            print("ERROR: s2 mode selected but S2 directory missing")
            return
        print("\n=== S2-ONLY MODE ===")
        maybe_run_probes("s2")
        try:
            run_training(s1_dir, s2_dir, "s2", epochs=NUM_EPOCHS, bidirectional=args.bidirectional)
        except Exception as e:
            print(f"S2 training failed: {e}")
        return
    if s1_exists and s2_exists:
        print("\n=== FULL MODE: S1 + S2 + DUAL ===")
        maybe_run_probes("full")
        print("\n=== Training S1 model ===")
        try:
            run_training(s1_dir, s2_dir, "s1", epochs=NUM_EPOCHS, bidirectional=args.bidirectional)
        except Exception as e:
            print(f"S1 training failed: {e}")
        early_stop_requested = False
        print("Reset early_stop_requested flag for S2 training")
        print("\n=== Training S2 model ===")
        try:
            run_training(s1_dir, s2_dir, "s2", epochs=NUM_EPOCHS, bidirectional=args.bidirectional)
        except Exception as e:
            print(f"S2 training failed: {e}")
        early_stop_requested = False
        print("Reset early_stop_requested flag for dual training")
        print("\n=== Training Dual model ===")
        try:
            run_training(s1_dir, s2_dir, "dual", epochs=NUM_EPOCHS, bidirectional=args.bidirectional)
        except Exception as e:
            print(f"Dual training failed: {e}")
        print("\n=== FULL PIPELINE COMPLETED ===")
    elif s1_exists:
        print("\n=== ONLY S1 AVAILABLE (full mode degrades to s1) ===")
        maybe_run_probes("s1")
        try:
            run_training(s1_dir, s2_dir, "s1", epochs=NUM_EPOCHS, bidirectional=args.bidirectional)
        except Exception as e:
            print(f"S1 training failed: {e}")
    elif s2_exists:
        print("\n=== ONLY S2 AVAILABLE (full mode degrades to s2) ===")
        maybe_run_probes("s2")
        try:
            run_training(s1_dir, s2_dir, "s2", epochs=NUM_EPOCHS, bidirectional=args.bidirectional)
        except Exception as e:
            print(f"S2 training failed: {e}")

if __name__ == "__main__":
    main()