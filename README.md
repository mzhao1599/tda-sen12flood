## Pipeline for Topological Data Analysis on [SEN12FLOOD Dataset](https://clmrmb.github.io/SEN12-FLOOD/)

Python version: 3.9.12

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
Download SEN12FLOOD from [Source Cooperative](https://source.coop/esa/sen12flood) or [IEEE Dataport](https://ieee-dataport.org/open-access/sen12-flood-sar-and-multispectral-dataset-flood-detection).

### 3. Preprocess Dataset
Creates new `s1` and `s2` directories with processed data.
```bash
python preprocess.py --s1-dir s1 --s2-dir s2 --sen12-root .
```

### 4. Compute Cubical Persistence (Grayscale Maps)
```bash
python compute_cubical.py s1/gray
python compute_cubical.py s2/gray
```

### 5. Download BIFOLD BigEarthNet v2.0 Pretrained ResNet50 Weights
```bash
python download.py
```

### 6. Fine-tune ResNet50
Determinism: export `CUBLAS_WORKSPACE_CONFIG=":4096:8"` before running.
```bash
python resnet_GRU.py
python resnet_GRU.py --bidirectional
```

### 7. Train GRU on Gaussian RBF Embeddings of Persistence Diagrams
```bash
python topoGE.py
python topoGE.py --bidirectional
```

### 8. Late Fusion Sweep (ResNet + topoGE detailed prediction logs)
```bash
python latefusion.py
```

### 9. Cached Feature Fusion + GRU Training
```bash
python fusion_runner.py
python fusion_runner.py --bidirectional
```

### 10. Sequence Location Map (GeoJSON labels from Source Cooperative)
```bash
python sequence_map.py
```

### 11. Persistence Diagram Visualization
```bash
python visualize_pd.py s1/gray/cubical
```

### Hardware
Experiments were run on an NVIDIA Titan XP with CUDA 11.7.
