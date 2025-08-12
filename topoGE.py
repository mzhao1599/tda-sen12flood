from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.__stdout__.reconfigure(encoding='utf-8')
sys.__stderr__.reconfigure(encoding='utf-8')
import os, re, glob, argparse, math, atexit, signal, random
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
HIDDEN_SIZE   = 256
BATCH_SIZE    = 8     
NUM_EPOCHS    = 400
LR            = 0.001
LR_HEAD       = 0.001
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
DEBUG         = False
DEBUG_EVERY   = 50
TOP_K_PD      = 200
BETA_F        = math.sqrt(2) 
GRID_SIZE     = 10   
FEAT_DIM      = 2 * (GRID_SIZE * GRID_SIZE)  
DEFAULT_SEED  = 1599
GLOBAL_SEED   = DEFAULT_SEED
MOD_EMBED_DIM = 8
QP_CENTERS = {0: {0: None, 1: None}, 1: {0: None, 1: None}} 
QP_SIGMA2  = {0: {0: None, 1: None}, 1: {0: None, 1: None}}  

early_stop_requested = False
log_file = None

def signal_handler(signum, frame):
    global early_stop_requested
    print("\nCtrl+C detected: will finish epoch then stop...")
    early_stop_requested = True

def set_global_determinism(seed: int):
    """Set seeds and deterministic flags across libraries."""
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

def _seed_worker(worker_id: int):
    """Ensure each DataLoader worker has a distinct but deterministic seed."""
    base = GLOBAL_SEED
    worker_seed = base + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class Tee:
    def write(self, x):
        sys.__stdout__.write(x)
        if log_file: log_file.write(x)
    def flush(self):
        if log_file and not log_file.closed:
            try: log_file.flush()
            except: pass

NAME_RE = re.compile(
    r"(?P<seq>\d+)_(?P<idx_c>\d+)_([A-Za-z0-9]+)_(?P<idx_m>\d+)_(?P<date>\d{8})_(?P<label>[01])\.npy$"
)

def parse_pd_name(path: str):
    m = NAME_RE.match(os.path.basename(path)); assert m, f"Bad filename: {path}"
    return m['seq'], int(m['idx_c']), int(m['label'])

def load_pd(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if arr.size==0 or arr.ndim!=2 or arr.shape[1]<3:
        raise ValueError(f"Invalid PD contents: {path}")
    return arr[:,:3]

def filter_top_k_per_dim(pd: np.ndarray, k: int) -> np.ndarray:
    if pd.size==0: return pd
    out=[]
    for dim in [0,1]:
        sub = pd[pd[:,0]==dim]
        if sub.size==0: continue
        life = sub[:,2]-sub[:,1]
        if len(sub)>k:
            sub = sub[np.argsort(life)[-k:]]
        out.append(sub)
    return np.concatenate(out, axis=0) if out else pd

def init_quartile_qpersemb_centers(dataset, train_indices, is_dual: bool, single_mod_id: int|None=None):
    """Compute separate quantile grid centers per modality (and per homology dim)."""
    global QP_CENTERS, QP_SIGMA2
    target_mods = [0,1] if is_dual else [single_mod_id]
    quantiles = list(np.linspace(0,1,GRID_SIZE))
    births = {m:{0:[],1:[]} for m in target_mods}
    deaths = {m:{0:[],1:[]} for m in target_mods}
    for idx in train_indices:
        seq = dataset._seqs[idx]
        for item in seq:
            if isinstance(item, tuple): 
                path = item[1]; mod = single_mod_id if single_mod_id is not None else 0
            else:  
                path = item['path']; mod = 0 if item.get('sat')=='s1' else 1
            if mod not in target_mods: continue
            pd = load_pd(path)
            if pd.size==0: continue
            mask = (pd[:,0]<=1) & np.isfinite(pd[:,1]) & np.isfinite(pd[:,2])
            if not mask.any(): continue
            pd = filter_top_k_per_dim(pd[mask], TOP_K_PD)
            for dim in [0,1]:
                sub = pd[pd[:,0]==dim]
                if sub.size==0: continue
                births[mod][dim].append(sub[:,1]); deaths[mod][dim].append(sub[:,2])
    for mod in target_mods:
        for dim in [0,1]:
            if len(births[mod][dim])==0:
                raise ValueError(f"No valid bars collected to initialize centers for modality {mod} H{dim}")
            all_births = np.concatenate(births[mod][dim]); all_deaths = np.concatenate(deaths[mod][dim])
            all_births = all_births[np.isfinite(all_births)]; all_deaths = all_deaths[np.isfinite(all_deaths)]
            if all_births.size==0 or all_deaths.size==0:
                raise ValueError(f"Non-finite births/deaths for modality {mod} H{dim}")
            qb = np.quantile(all_births, quantiles); qd = np.quantile(all_deaths, quantiles)
            std_b = np.std(all_births); std_d = np.std(all_deaths); scale=(std_b+std_d)/2
            if (not np.isfinite(scale)) or scale<=1e-8:
                raise ValueError(f"Degenerate scale for modality {mod} H{dim} (scale={scale})")
            sigma = 0.1*scale; sigma2=float(sigma**2)
            gb,gd = np.meshgrid(qb,qd, indexing='ij'); centers = np.stack([gb.ravel(), gd.ravel()],axis=1)
            QP_CENTERS[mod][dim] = torch.tensor(centers, dtype=torch.float32)
            QP_SIGMA2[mod][dim] = sigma2
    print("Initialized quartile qpersemb centers (separate per modality):")
    for mod in target_mods:
        for dim in [0,1]:
            c = QP_CENTERS[mod][dim].numpy()
            print(f"  Mod{mod} H{dim}: births[{c[:,0].min():.3f},{c[:,0].max():.3f}] deaths[{c[:,1].min():.3f},{c[:,1].max():.3f}] sigma2={QP_SIGMA2[mod][dim]:.3e}")

def quartile_qpersemb_feature(pd: np.ndarray, modality_id: int) -> np.ndarray:
    """Compute feature vector (2*GRID_SIZE^2) for a single PD using modality-specific centers."""
    feats=[]
    for dim in [0,1]:
        centers=QP_CENTERS[modality_id][dim]; sigma2=QP_SIGMA2[modality_id][dim]
        if centers is None:
            raise RuntimeError(f'Quartile centers not initialized for modality {modality_id} H{dim}')
        sub = pd[pd[:,0]==dim]
        if sub.size==0:
            raise ValueError(f"No bars for modality {modality_id} dimension H{dim} in PD; fallback disabled")
        births=torch.from_numpy(sub[:,1].astype(np.float32))
        deaths=torch.from_numpy(sub[:,2].astype(np.float32))
        life=(deaths-births).clamp(min=0.)
        pts=torch.stack([births,deaths], dim=1)
        diff=pts.unsqueeze(1)-centers.unsqueeze(0)
        dist2=(diff*diff).sum(-1)
        rbf=torch.exp(-0.5*dist2/sigma2)
        weighted=(rbf*life.unsqueeze(1)).sum(0)
        feats.append(weighted)
    return torch.cat(feats, dim=0).numpy()

class PDSeqDataset(Dataset):
    def __init__(self, root_dir: str, modality_id: int):
        paths=glob.glob(os.path.join(root_dir,'*.npy')); assert paths, f"No .npy found in {root_dir}"
        buckets: Dict[str,List[Tuple[int,str,int]]]={}
        for p in paths:
            seq, idx_c, lbl = parse_pd_name(p)
            buckets.setdefault(seq,[]).append((idx_c,p,lbl))
        self._seqs=[]
        self.modality_id = modality_id
        for seq_id, items in buckets.items():
            items.sort(key=lambda t: t[0]); self._seqs.append(items)
    def __len__(self): return len(self._seqs)
    def __getitem__(self, idx):
        items=self._seqs[idx]; feats=[]; labels=[]
        for _, path, lbl in items:
            pd=filter_top_k_per_dim(load_pd(path), TOP_K_PD)
            feats.append(quartile_qpersemb_feature(pd, self.modality_id)); labels.append(lbl)
        seq_id = items[0][1].split(os.sep)[-1].split('_')[0]
        return torch.from_numpy(np.stack(feats)), torch.tensor(labels, dtype=torch.long), seq_id

def collate_pd_single(batch):
    feat_seqs,lbl_seqs,seq_ids = zip(*batch)
    lengths=[s.shape[0] for s in feat_seqs]; T_max=max(lengths); B=len(batch); Fdim=feat_seqs[0].shape[1]
    feats=torch.zeros(B,T_max,Fdim); lbls=torch.full((B,T_max), -100, dtype=torch.long)
    for i,(fs,ls) in enumerate(zip(feat_seqs,lbl_seqs)):
        T=fs.shape[0]; feats[i,:T]=fs; lbls[i,:T]=ls
    return feats,lbls,torch.tensor(lengths), list(seq_ids)

class CombinedPDDataset(Dataset):
    def __init__(self, s1_dir: str, s2_dir: str):
        all_items=[]
        for root in [s1_dir, s2_dir]:
            if not (root and os.path.isdir(root)): continue
            for fname in os.listdir(root):
                if not fname.endswith('.npy'): continue
                if not NAME_RE.match(fname): continue
                seq, idx_c, lbl = parse_pd_name(fname); sat=fname.split('_')[2]
                all_items.append({'seq_id':seq,'idx_c':idx_c,'lbl':lbl,'sat':sat,'path':os.path.join(root,fname)})
        seqs: Dict[str,List[Dict]]={}
        for it in all_items: seqs.setdefault(it['seq_id'], []).append(it)
        self._seqs=[]
        for seq_id, items in seqs.items(): items.sort(key=lambda x:x['idx_c']); self._seqs.append(items)
    def __len__(self): return len(self._seqs)
    def __getitem__(self, idx):
        items=self._seqs[idx]; feats=[]; labels=[]; mods=[]
        for it in items:
            pd=filter_top_k_per_dim(load_pd(it['path']), TOP_K_PD)
            mod_id = 0 if it['sat']=='s1' else 1
            feats.append(quartile_qpersemb_feature(pd, mod_id)); labels.append(it['lbl']); mods.append(mod_id)
        seq_id=items[0]['seq_id']
        return torch.from_numpy(np.stack(feats)), torch.tensor(labels), torch.tensor(mods), seq_id

def collate_pd_dual(batch):
    feat_seqs,lbl_seqs,mod_seqs,seq_ids = zip(*batch)
    lengths=[s.shape[0] for s in feat_seqs]; T_max=max(lengths); B=len(batch); Fdim=feat_seqs[0].shape[1]
    feats=torch.zeros(B,T_max,Fdim); lbls=torch.full((B,T_max), -100, dtype=torch.long); mods=torch.full((B,T_max), -1, dtype=torch.long)
    for i,(fs,ls,ms) in enumerate(zip(feat_seqs,lbl_seqs,mod_seqs)):
        T=fs.shape[0]; feats[i,:T]=fs; lbls[i,:T]=ls; mods[i,:T]=ms
    return feats,lbls,mods,torch.tensor(lengths), list(seq_ids)

class PDGRU(nn.Module):
    def __init__(self, input_dim=None, hidden=HIDDEN_SIZE, bidirectional: bool=True):
        if input_dim is None: input_dim=FEAT_DIM
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(input_dim, hidden, batch_first=True, bidirectional=bidirectional)
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, 1)
    def forward(self, feats, lengths, return_padded: bool=False):
        packed=pack_padded_sequence(feats, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out,_=self.gru(packed); out,_=pad_packed_sequence(out, batch_first=True); logits=self.head(out).squeeze(-1)
        if return_padded: return logits
        packed_logits=pack_padded_sequence(logits, lengths.cpu(), batch_first=True, enforce_sorted=False); return packed_logits.data, None

class DualPDGRU(nn.Module):
    def __init__(self, input_dim=None, hidden=HIDDEN_SIZE, mod_embed_dim: int=MOD_EMBED_DIM, bidirectional: bool=True):
        if input_dim is None: input_dim=FEAT_DIM
        super().__init__()
        self.bidirectional = bidirectional
        self.use_mod_embed = mod_embed_dim > 0
        self.mod_embed = nn.Embedding(2, mod_embed_dim) if self.use_mod_embed else None
        total_in = input_dim + (mod_embed_dim if self.use_mod_embed else 0)
        self.gru = nn.GRU(total_in, hidden, batch_first=True, bidirectional=bidirectional)
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim,1)
    def forward(self, feats, lengths, mods, return_padded: bool=False):
        if self.use_mod_embed:
            feats = torch.cat([feats, self.mod_embed(mods.clamp(min=0))], dim=-1)
        packed = pack_padded_sequence(feats, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out,_ = self.gru(packed); out,_ = pad_packed_sequence(out, batch_first=True); logits = self.head(out).squeeze(-1)
        if return_padded: return logits
        packed_logits = pack_padded_sequence(logits, lengths.cpu(), batch_first=True, enforce_sorted=False); return packed_logits.data, None

def weighted_bce_loss(logits, packed_labels: PackedSequence, pos_weight: float):
    targets=packed_labels.data.float(); pos_w=torch.tensor([pos_weight], device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_w)

def compute_confusion(preds, labels):
    TP=((preds==1)&(labels==1)).sum().item(); FP=((preds==1)&(labels==0)).sum().item(); TN=((preds==0)&(labels==0)).sum().item(); FN=((preds==0)&(labels==1)).sum().item(); return TP,FP,TN,FN

def compute_f1_score(TP,FP,FN):
    precision=TP/(TP+FP) if (TP+FP)>0 else 0.0; recall=TP/(TP+FN) if (TP+FN)>0 else 0.0; f1=2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0; return f1,precision,recall

def compute_fbeta(TP, FP, FN, beta: float):
    precision=TP/(TP+FP) if (TP+FP)>0 else 0.0; recall=TP/(TP+FN) if (TP+FN)>0 else 0.0
    if precision==0 and recall==0: return 0.0, precision, recall
    b2=beta*beta
    fbeta=(1+b2)*precision*recall/(b2*precision+recall) if (b2*precision+recall)>0 else 0.0
    return fbeta, precision, recall

def calculate_class_weights_and_bias(dataset, indices):
    total=pos=0
    for idx in indices:
        seq=dataset._seqs[idx]
        for item in seq:
            lbl=item[2] if isinstance(item, tuple) else item['lbl']; total+=1; pos+=lbl
    if total==0: return 1.0,0.0,0.0
    pos_rate=pos/total; neg_rate=1-pos_rate; pos_rate=max(0.001,min(0.999,pos_rate)); neg_rate=max(0.001,min(0.999,neg_rate))
    pos_weight=neg_rate/pos_rate; log_odds=math.log(pos_rate/neg_rate); print(f"  Frames: {total} Pos: {pos} ({pos_rate:.3f}) Neg: {total-pos} ({neg_rate:.3f}) pos_weight={pos_weight:.3f} log_odds={log_odds:.3f}"); return pos_weight,log_odds,pos_rate

def init_classification_bias(model, log_odds_bias: float):
    with torch.no_grad(): model.head.bias.fill_(log_odds_bias)

def train_one_epoch(model, loader, optim, pos_weight, is_dual=False):
    model.train(); tot=hit=total_loss=0
    if is_dual:
        for batch in tqdm(loader, desc='Train', leave=False):
            if len(batch)==5: feats,lbls,mods,lengths,_=batch
            else: feats,lbls,mods,lengths=batch
            feats,mods=feats.to(DEVICE),mods.to(DEVICE); lengths=lengths.to(DEVICE)
            packed_lbls=pack_padded_sequence(lbls.to(DEVICE), lengths.cpu(), batch_first=True, enforce_sorted=False)
            logits,_=model(feats,lengths,mods); loss=weighted_bce_loss(logits, packed_lbls, pos_weight)
            optim.zero_grad(); loss.backward(); optim.step(); total_loss+=loss.item()*packed_lbls.data.numel()
            preds=(torch.sigmoid(logits)>.5).long(); hit+=(preds==packed_lbls.data.long()).sum().item(); tot+=packed_lbls.data.numel()
    else:
        for batch in tqdm(loader, desc='Train', leave=False):
            if len(batch)==4: feats,lbls,lengths,_=batch
            else: feats,lbls,lengths=batch
            feats=feats.to(DEVICE); lengths=lengths.to(DEVICE)
            packed_lbls=pack_padded_sequence(lbls.to(DEVICE), lengths.cpu(), batch_first=True, enforce_sorted=False)
            logits,_=model(feats,lengths); loss=weighted_bce_loss(logits, packed_lbls, pos_weight)
            optim.zero_grad(); loss.backward(); optim.step(); total_loss+=loss.item()*packed_lbls.data.numel()
            preds=(torch.sigmoid(logits)>.5).long(); hit+=(preds==packed_lbls.data.long()).sum().item(); tot+=packed_lbls.data.numel()
    return total_loss/max(1,tot), hit/max(1,tot)

@torch.no_grad()
def eval_epoch(model, loader, pos_weight, is_dual=False):
    model.eval(); tot=hit=loss_sum=0; TP=FP=TN=FN=0
    if is_dual:
        for batch in tqdm(loader, desc='Val', leave=False):
            if len(batch)==5: feats,lbls,mods,lengths,_=batch
            else: feats,lbls,mods,lengths=batch
            feats,mods=feats.to(DEVICE),mods.to(DEVICE); lengths=lengths.to(DEVICE)
            packed_lbls=pack_padded_sequence(lbls.to(DEVICE), lengths.cpu(), batch_first=True, enforce_sorted=False)
            logits,_=model(feats,lengths,mods); loss=weighted_bce_loss(logits, packed_lbls, pos_weight)
            probs=torch.sigmoid(logits); preds=(probs>.5).long(); loss_sum+=loss.item()*packed_lbls.data.numel(); tot+=packed_lbls.data.numel(); hit+=(preds==packed_lbls.data.long()).sum().item(); tP,fP,tN,fN=compute_confusion(preds, packed_lbls.data.long()); TP+=tP; FP+=fP; TN+=tN; FN+=fN
    else:
        for batch in tqdm(loader, desc='Val', leave=False):
            if len(batch)==4: feats,lbls,lengths,_=batch
            else: feats,lbls,lengths=batch
            feats=feats.to(DEVICE); lengths=lengths.to(DEVICE)
            packed_lbls=pack_padded_sequence(lbls.to(DEVICE), lengths.cpu(), batch_first=True, enforce_sorted=False)
            logits,_=model(feats,lengths); loss=weighted_bce_loss(logits, packed_lbls, pos_weight)
            probs=torch.sigmoid(logits); preds=(probs>.5).long(); loss_sum+=loss.item()*packed_lbls.data.numel(); tot+=packed_lbls.data.numel(); hit+=(preds==packed_lbls.data.long()).sum().item(); tP,fP,tN,fN=compute_confusion(preds, packed_lbls.data.long()); TP+=tP; FP+=fP; TN+=tN; FN+=fN
    return loss_sum/max(1,tot), hit/max(1,tot), (TP,FP,TN,FN), (TP,FP,TN,FN)

@torch.no_grad()
def validation_sweep(model, val_loader, is_dual=False, thresholds=None, modality: str|None=None):
    """Run threshold sweep and select deployment threshold.
    """
    if thresholds is None:
        thresholds = [i/100 for i in range(0, 101)]
    print(f"\n=== Validation Sweep ({len(thresholds)} thresholds) ===")
    all_probs = []
    all_labels = []
    sequence_details = []
    model.eval()
    if is_dual:
        for batch in tqdm(val_loader, desc='Collecting predictions'):
            if len(batch) == 5:
                feats, lbls, mods, lengths, seq_ids = batch
            else:
                feats, lbls, mods, lengths = batch; seq_ids = ['unknown'] * feats.size(0)
            feats, mods = feats.to(DEVICE), mods.to(DEVICE); lengths = lengths.to(DEVICE)
            logits_full = model(feats, lengths, mods, return_padded=True); probs_full = torch.sigmoid(logits_full)
            packed_lbls = pack_padded_sequence(lbls.to(DEVICE), lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_probs = pack_padded_sequence(probs_full, lengths.cpu(), batch_first=True, enforce_sorted=False)
            all_probs.append(packed_probs.data.cpu().numpy()); all_labels.append(packed_lbls.data.cpu().numpy())
            for i, sid in enumerate(seq_ids):
                L = lengths[i].item()
                sequence_details.append({'real_seq_id': sid, 'length': L, 'labels': lbls[i, :L].tolist(), 'probs': probs_full[i, :L].detach().cpu().tolist()})
    else:
        for batch in tqdm(val_loader, desc='Collecting predictions'):
            if len(batch) == 4:
                feats, lbls, lengths, seq_ids = batch
            else:
                feats, lbls, lengths = batch; seq_ids = ['unknown'] * feats.size(0)
            feats = feats.to(DEVICE); lengths = lengths.to(DEVICE)
            logits_full = model(feats, lengths, return_padded=True); probs_full = torch.sigmoid(logits_full)
            packed_lbls = pack_padded_sequence(lbls.to(DEVICE), lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_probs = pack_padded_sequence(probs_full, lengths.cpu(), batch_first=True, enforce_sorted=False)
            all_probs.append(packed_probs.data.cpu().numpy()); all_labels.append(packed_lbls.data.cpu().numpy())
            for i, sid in enumerate(seq_ids):
                L = lengths[i].item()
                sequence_details.append({'real_seq_id': sid, 'length': L, 'labels': lbls[i, :L].tolist(), 'probs': probs_full[i, :L].detach().cpu().tolist()})
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    results = []
    for th in thresholds:
        preds = (all_probs >= th).astype(int)
        TP = ((preds == 1) & (all_labels == 1)).sum(); FP = ((preds == 1) & (all_labels == 0)).sum(); TN = ((preds == 0) & (all_labels == 0)).sum(); FN = ((preds == 0) & (all_labels == 1)).sum()
        precision = TP/(TP+FP) if (TP+FP) > 0 else 0.0
        recall = TP/(TP+FN) if (TP+FN) > 0 else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
        acc = (TP+TN)/(TP+FP+TN+FN)
        results.append({'threshold': th, 'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': acc, 'tp': TP, 'fp': FP, 'tn': TN, 'fn': FN})
    best_f1 = max(results, key=lambda x: x['f1'])
    high_p = [r for r in results if r['precision'] >= 0.90]
    fallback_used = False
    if not high_p:
        if modality == 's1':
            best_hp = max(results, key=lambda x: (x['precision'], x['recall'], -x['threshold']))
            optimal_threshold = best_hp['threshold']; deploy_metrics = best_hp; source = 'fallback_highest_precision'
            fallback_used = True
            print(f"WARNING: No threshold reached >=90% precision for S1; fallback to highest precision: P={best_hp['precision']:.3f} R={best_hp['recall']:.3f} @ th={best_hp['threshold']:.2f}")
        else:
            raise RuntimeError("No threshold reached >=90% precision; fallback only allowed for S1")
    else:
        best_r_90 = max(high_p, key=lambda x: x['recall'])
        optimal_threshold = best_r_90['threshold']; deploy_metrics = best_r_90; source = 'recall_at_90p'
        print(f"Best Recall@90%P R={best_r_90['recall']:.3f} @ th={best_r_90['threshold']:.2f} (P={best_r_90['precision']:.3f})")
    print(f"(Best F1 {best_f1['f1']:.4f} @ {best_f1['threshold']:.2f} shown for reference only; not used)")
    if fallback_used:
        print("Threshold source: fallback_highest_precision (S1 only)")
    return {'all_results': results, 'best_f1': best_f1, 'best_accuracy': None, 'best_recall_at_90p': (None if fallback_used else deploy_metrics), 'optimal_threshold': optimal_threshold, 'deploy_metrics': deploy_metrics, 'threshold_source': source, 'sequence_details': sequence_details, 'fallback_used': fallback_used}

def save_detailed_predictions_to_file(sweep_results, checkpoint_dir, modality, direction_tag: str|None=None):
    if direction_tag:
        predictions_file=os.path.join(checkpoint_dir, f"{modality}_topoGE_{direction_tag}_detailed_predictions.txt")
    else:
        predictions_file=os.path.join(checkpoint_dir, f"{modality}_topoGE_detailed_predictions.txt")
    seq_details=sweep_results.get('sequence_details', []); opt_th=sweep_results.get('optimal_threshold',0.5)
    with open(predictions_file,'w', encoding = 'utf-8', errors='replace') as f:
        f.write(f"Optimal threshold: {opt_th:.4f}\n")
        f.write("Format: frame_idx prob label pred@0.50 pred@opt\n\n")
        def _seq_key(d):
            try: return int(d['real_seq_id'])
            except: return 10**9
        for sd in sorted(seq_details, key=_seq_key):
            probs=sd['probs']; labels=sd['labels']; sid=sd['real_seq_id']
            f.write(f"SEQ {sid} len={sd['length']} positives={sum(labels)}\n")
            for i,(p,l) in enumerate(zip(probs,labels)):
                pred05=1 if p>=0.5 else 0; predOpt=1 if p>=opt_th else 0
                f.write(f"  {i:03d} {p:.4f} {l} {pred05} {predOpt}\n")
            f.write("\n")
    print(f"Detailed predictions saved to {predictions_file} ({len(seq_details)} sequences)")


def run_training(s1_dir: str, s2_dir: str, modality: str='s1', epochs: int=NUM_EPOCHS, seed: int=DEFAULT_SEED, bidirectional: bool=True):
    print(f"\n=== {modality.upper()} Training (topoGE) ==="); checkpoint_dir='.'; print(f"  Checkpoint dir: {checkpoint_dir}")
    if modality=='dual': ds_full=CombinedPDDataset(s1_dir,s2_dir)
    else:
        root_dir=s1_dir if modality=='s1' else s2_dir
        single_mod_id = 0 if modality=='s1' else 1
        ds_full=PDSeqDataset(root_dir, modality_id=single_mod_id)
    val_idx,train_idx=[],[]
    for i, seq in enumerate(ds_full._seqs):
        if modality=='dual': seq_id=seq[0]['seq_id']
        else: seq_id=os.path.basename(seq[0][1]).split('_')[0]
        is_val=not (len(seq_id)==4 and seq_id.isdigit()); (val_idx if is_val else train_idx).append(i)
    print(f"  Split: {len(train_idx)} train, {len(val_idx)} val sequences")
    if modality=='dual':
        init_quartile_qpersemb_centers(ds_full, train_idx, is_dual=True)
    else:
        init_quartile_qpersemb_centers(ds_full, train_idx, is_dual=False, single_mod_id=(0 if modality=='s1' else 1))
    g = torch.Generator(); g.manual_seed(seed)
    if modality=='dual':
        train_dl=DataLoader(torch.utils.data.Subset(ds_full, train_idx), BATCH_SIZE, shuffle=True, collate_fn=collate_pd_dual, num_workers=0, worker_init_fn=_seed_worker, generator=g)
        val_dl  =DataLoader(torch.utils.data.Subset(ds_full, val_idx), BATCH_SIZE, shuffle=False, collate_fn=collate_pd_dual, num_workers=0, worker_init_fn=_seed_worker, generator=g); is_dual=True
    else:
        train_dl=DataLoader(torch.utils.data.Subset(ds_full, train_idx), BATCH_SIZE, shuffle=True, collate_fn=collate_pd_single, num_workers=0, worker_init_fn=_seed_worker, generator=g)
        val_dl  =DataLoader(torch.utils.data.Subset(ds_full, val_idx), BATCH_SIZE, shuffle=False, collate_fn=collate_pd_single, num_workers=0, worker_init_fn=_seed_worker, generator=g); is_dual=False
    pos_weight,log_odds,pos_rate=calculate_class_weights_and_bias(ds_full, train_idx)
    model=(DualPDGRU(bidirectional=bidirectional).to(DEVICE) if is_dual else PDGRU(bidirectional=bidirectional).to(DEVICE))
    init_classification_bias(model, log_odds)
    optim=torch.optim.Adam(model.parameters(), lr=LR)
    best_score=0.0; best_epoch=0
    direction_tag = 'bi' if bidirectional else 'uni'
    ckpt_name = ('dual_topoGE_gru_best_' + direction_tag + '.pt') if is_dual else f'topoGE_gru_{modality}_{direction_tag}.pt'
    patience=20
    for epoch in range(1, epochs+1):
        tr_loss,tr_acc=train_one_epoch(model, train_dl, optim, pos_weight, is_dual)
        va_loss,va_acc,(TP_f,FP_f,TN_f,FN_f),_=eval_epoch(model, val_dl, pos_weight, is_dual)
        f1,prec,rec=compute_f1_score(TP_f,FP_f,FN_f)
        fbeta,_,_=compute_fbeta(TP_f,FP_f,FN_f,BETA_F)
        if fbeta>best_score:
            best_score=fbeta; best_epoch=epoch; torch.save(model.state_dict(), ckpt_name)
        prefix='DUAL' if is_dual else modality.upper()
        print(f"{prefix} E{epoch:03d} train-loss={tr_loss:.4f} acc={tr_acc:.3f} | val-loss={va_loss:.4f} acc={va_acc:.3f} f1={f1:.3f} fβ={fbeta:.3f} (β={BETA_F:.2f}) P={prec:.3f} R={rec:.3f} | val-cm TP={TP_f} FP={FP_f} TN={TN_f} FN={FN_f}")
        if early_stop_requested or (epoch-best_epoch)>=patience: print('Early stopping'); break
    print(f"Loading best epoch {best_epoch} model (best Fβ={best_score:.3f}) for sweep...")
    model.load_state_dict(torch.load(ckpt_name, map_location=DEVICE))
    sweep_results=validation_sweep(model, val_dl, is_dual=is_dual, modality=modality)
    save_detailed_predictions_to_file(sweep_results, '.', modality, direction_tag)
    print(f"Best {prefix} val-Fβ (β={BETA_F:.2f}): {best_score:.3f} @ epoch {best_epoch}")
    print(f"Deploy threshold ({sweep_results['threshold_source']}): {sweep_results['optimal_threshold']:.3f}")
    return sweep_results

def parse_args():
    p=argparse.ArgumentParser(description='topoGE GRU flood sequence classifier')
    p.add_argument('--s1-dir', default='s1/gray/cubical', help='Directory containing S1 PD .npy files')
    p.add_argument('--s2-dir', default='s2/gray/cubical', help='Directory containing S2 PD .npy files')
    p.add_argument('--debug', action='store_true', default=False, help='Enable debug output')
    p.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Training epochs')
    p.add_argument('--lr', type=float, default=LR, help='Learning rate')
    p.add_argument('--top-k-pd', type=int, default=TOP_K_PD, help='Top-k bars per dim retained before embedding')
    p.add_argument('--grid-size', type=int, default=GRID_SIZE, help='Quantile grid size per axis (e.g. 5->25; 6->36)')
    p.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed for deterministic training')
    p.add_argument('--bidirectional', action='store_true', default=False, help='Enable bidirectional GRU (default: disabled)')
    return p.parse_args()

def main():
    global DEBUG, NUM_EPOCHS, LR, TOP_K_PD, log_file
    signal.signal(signal.SIGINT, signal_handler)
    args=parse_args()
    DEBUG=args.debug; NUM_EPOCHS=args.epochs; LR=args.lr; TOP_K_PD=args.top_k_pd
    set_global_determinism(args.seed)
    globals()['GLOBAL_SEED']=args.seed
    globals()['GRID_SIZE']=args.grid_size
    globals()['FEAT_DIM']=2*(GRID_SIZE*GRID_SIZE)
    s1_dir=args.s1_dir; s2_dir=args.s2_dir
    s1_exists=os.path.isdir(s1_dir); s2_exists=os.path.isdir(s2_dir)
    log_file_path='topoGE_gru.txt'; log_fh=open(log_file_path,'a', buffering=1, encoding='utf-8', errors='replace'); atexit.register(log_fh.close); globals()['log_file']=log_fh; sys.stdout=Tee()
    print(f"Log file: {log_file_path}")
    print(f"Configuration: S1={s1_dir} ({'exists' if s1_exists else 'MISSING'}) S2={s2_dir} ({'exists' if s2_exists else 'MISSING'}) Batch={BATCH_SIZE} Epochs={NUM_EPOCHS} LR={LR} TopK={TOP_K_PD} Grid={GRID_SIZE} FeatDim={FEAT_DIM} Seed={args.seed} BiGRU={args.bidirectional}")
    if s1_exists and s2_exists:
        print('\n=== FULL topoGE PIPELINE (S1+S2) ===')
        try:
            run_training(s1_dir, s2_dir, 's1', NUM_EPOCHS, seed=args.seed, bidirectional=args.bidirectional)
        except Exception as e:
            print(f'S1 training failed: {e}')
            globals()['early_stop_requested']=False
        globals()['early_stop_requested'] = False
        try:
            run_training(s1_dir, s2_dir, 's2', NUM_EPOCHS, seed=args.seed, bidirectional=args.bidirectional)
        except Exception as e:
            print(f'S2 training failed: {e}')
        globals()['early_stop_requested'] = False
        try:
            run_training(s1_dir, s2_dir, 'dual', NUM_EPOCHS, seed=args.seed, bidirectional=args.bidirectional)
        except Exception as e:
            print(f'Dual training failed: {e}')
    elif s1_exists:
        run_training(s1_dir, s2_dir, 's1', NUM_EPOCHS, seed=args.seed, bidirectional=args.bidirectional)
    elif s2_exists:
        run_training(s1_dir, s2_dir, 's2', NUM_EPOCHS, seed=args.seed, bidirectional=args.bidirectional)
    else:
        print('ERROR: no PD directories found.')

if __name__=='__main__':
    main()
