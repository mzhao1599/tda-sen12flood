diff --git a/fusion_runner.py b/fusion_runner.py
index 4829412c210508bf678760eae850780f68c1c4d3..40ae9215b73d60f257c5d8e5e25e0629fe5b7acf 100644
--- a/fusion_runner.py
+++ b/fusion_runner.py
@@ -2,50 +2,61 @@ import os, sys, re, glob, math, atexit, argparse, random
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
 
+
+def seed_worker(worker_id: int):
+    """Set NumPy's seed for DataLoader workers.
+
+    On Windows, DataLoader workers are launched with the ``spawn`` start method,
+    which requires worker initialization functions to be pickleable. Defining
+    this helper at module scope avoids pickling errors that arise when using
+    lambdas or nested functions.
+    """
+    np.random.seed(DEFAULT_SEED + worker_id)
+
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
diff --git a/fusion_runner.py b/fusion_runner.py
index 4829412c210508bf678760eae850780f68c1c4d3..40ae9215b73d60f257c5d8e5e25e0629fe5b7acf 100644
--- a/fusion_runner.py
+++ b/fusion_runner.py
@@ -552,97 +563,129 @@ def run_late_fusion_single(modality: str, stacked_dir: str, cubical_dir: str,
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
-    train_dl = DataLoader(torch.utils.data.Subset(ds, train_idx), BATCH_SIZE, shuffle=True,  collate_fn=collate_fusion_single, num_workers=2, worker_init_fn=lambda wid: np.random.seed(DEFAULT_SEED+wid), generator=g)
-    val_dl   = DataLoader(torch.utils.data.Subset(ds, val_idx),   BATCH_SIZE, shuffle=False, collate_fn=collate_fusion_single, num_workers=2, worker_init_fn=lambda wid: np.random.seed(DEFAULT_SEED+wid), generator=g)
+    train_dl = DataLoader(
+        torch.utils.data.Subset(ds, train_idx),
+        BATCH_SIZE,
+        shuffle=True,
+        collate_fn=collate_fusion_single,
+        num_workers=2,
+        worker_init_fn=seed_worker,
+        generator=g,
+    )
+    val_dl = DataLoader(
+        torch.utils.data.Subset(ds, val_idx),
+        BATCH_SIZE,
+        shuffle=False,
+        collate_fn=collate_fusion_single,
+        num_workers=2,
+        worker_init_fn=seed_worker,
+        generator=g,
+    )
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
-    train_dl = DataLoader(torch.utils.data.Subset(ds, train_idx), BATCH_SIZE, shuffle=True,  collate_fn=collate_fusion_dual, num_workers=2, worker_init_fn=lambda wid: np.random.seed(DEFAULT_SEED+wid), generator=g)
-    val_dl   = DataLoader(torch.utils.data.Subset(ds, val_idx),   BATCH_SIZE, shuffle=False, collate_fn=collate_fusion_dual, num_workers=2, worker_init_fn=lambda wid: np.random.seed(DEFAULT_SEED+wid), generator=g)
+    train_dl = DataLoader(
+        torch.utils.data.Subset(ds, train_idx),
+        BATCH_SIZE,
+        shuffle=True,
+        collate_fn=collate_fusion_dual,
+        num_workers=2,
+        worker_init_fn=seed_worker,
+        generator=g,
+    )
+    val_dl = DataLoader(
+        torch.utils.data.Subset(ds, val_idx),
+        BATCH_SIZE,
+        shuffle=False,
+        collate_fn=collate_fusion_dual,
+        num_workers=2,
+        worker_init_fn=seed_worker,
+        generator=g,
+    )
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
diff --git a/fusion_runner.py b/fusion_runner.py
index 4829412c210508bf678760eae850780f68c1c4d3..40ae9215b73d60f257c5d8e5e25e0629fe5b7acf 100644
--- a/fusion_runner.py
+++ b/fusion_runner.py
@@ -671,26 +714,31 @@ def main():
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
+    # ``freeze_support`` is required for compatibility with Windows multiprocessing
+    # when this script is executed via spawn (the default on Windows).
+    import multiprocessing
+
+    multiprocessing.freeze_support()
     main()
