import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import List, Optional
import re
import ot 

def find_sequence_files(folder: str, sequence_id: str) -> List[str]:
    """Find all .npy files for a specific sequence ID."""
    pattern = os.path.join(folder, f"{sequence_id}_*.npy")
    files = glob.glob(pattern)
    def extract_index(filepath):
        basename = os.path.basename(filepath)
        match = re.search(r'_(\d+)_[A-Za-z0-9]+_(\d+)_(\d{8})_([01])\.npy$', basename)
        if match:
            return int(match.group(2))  
        return 0
    
    files.sort(key=extract_index)
    return files

def load_persistence_diagram(filepath: str) -> Optional[np.ndarray]:
    """Load persistence diagram from .npy file."""
    try:
        pd = np.load(filepath, allow_pickle=False)
        if pd.size == 0:
            return np.zeros((0, 3))
        if pd.ndim != 2 or pd.shape[1] < 3:
            print(f"Warning: Invalid PD shape in {filepath}: {pd.shape}")
            return None
        return pd[:, :3]  
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_date_from_filename(filepath: str) -> str:
    """Extract date from filename pattern."""
    basename = os.path.basename(filepath)
    match = re.search(r'_(\d{8})_', basename)
    if match:
        date_str = match.group(1)
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    return "Unknown"

def extract_flood_state_from_filename(filepath: str) -> int:
    """Extract flood state from filename pattern (0 = no flood, 1 = flood)."""
    basename = os.path.basename(filepath)
    match = re.search(r'_(\d{8})_([01])\.npy$', basename)
    if match:
        return int(match.group(2))
    return 0  

def wasserstein_distance_pds(pd1: np.ndarray, pd2: np.ndarray, dim: int = 0) -> float:
    """Compute Wasserstein distance between two persistence diagrams for a given dimension (requires POT)."""
    pd1_dim = pd1[pd1[:, 0] == dim]
    pd2_dim = pd2[pd2[:, 0] == dim]
    if len(pd1_dim) == 0 and len(pd2_dim) == 0:
        return 0.0
    points1 = pd1_dim[:, 1:3] if len(pd1_dim) > 0 else np.zeros((0, 2))
    points2 = pd2_dim[:, 1:3] if len(pd2_dim) > 0 else np.zeros((0, 2))
    if len(points1) == 0:
        return np.sum(points2[:, 1] - points2[:, 0]) if len(points2) > 0 else 0.0
    if len(points2) == 0:
        return np.sum(points1[:, 1] - points1[:, 0])
    diag1 = np.column_stack([points1.mean(axis=1), points1.mean(axis=1)])
    diag2 = np.column_stack([points2.mean(axis=1), points2.mean(axis=1)])
    all_points1 = np.vstack([points1, diag2])
    all_points2 = np.vstack([points2, diag1])
    M = ot.dist(all_points1, all_points2, metric='euclidean')
    a = np.ones(len(all_points1)) / len(all_points1)
    b = np.ones(len(all_points2)) / len(all_points2)
    try:
        W = ot.emd2(a, b, M)
        return W
    except:
        return np.mean(M)

def filter_top_k_per_dim(pd: np.ndarray, k: int) -> np.ndarray:
    """Keep only the top k features per dimension (H0, H1) by lifetime."""
    if pd.size == 0:
        return pd
    filtered = []
    for dim in [0, 1]:
        mask = pd[:, 0] == dim
        pd_dim = pd[mask]
        if len(pd_dim) > 0:
            lifetimes = pd_dim[:, 2] - pd_dim[:, 1]
            if len(pd_dim) > k:
                topk_idx = np.argsort(lifetimes)[-k:]
                pd_dim = pd_dim[topk_idx]
        filtered.append(pd_dim)
    if filtered:
        return np.concatenate(filtered, axis=0)
    return pd

def plot_persistence_diagram(pd: np.ndarray, title: str, ax: plt.Axes, top_k: int = 50, xlim=None, ylim=None):
    """Plot a single persistence diagram on given axes, keeping top_k per dim."""
    pd = filter_top_k_per_dim(pd, top_k)
    if pd.size == 0:
        ax.text(0.5, 0.5, 'No features', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
        return
    h0_mask = pd[:, 0] == 0
    h1_mask = pd[:, 0] == 1
    h0_features = pd[h0_mask]
    h1_features = pd[h1_mask]
    if len(h0_features) > 0:
        ax.scatter(h0_features[:, 1], h0_features[:, 2], 
                  c='blue', alpha=0.7, s=20, label=f'H₀ ({len(h0_features)})')
    if len(h1_features) > 0:
        ax.scatter(h1_features[:, 1], h1_features[:, 2], 
                  c='red', alpha=0.7, s=20, label=f'H₁ ({len(h1_features)})')
    if xlim is not None and ylim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], 'k--', alpha=0.5, linewidth=1)
    else:
        if len(pd) > 0:
            min_val = min(pd[:, 1].min(), pd[:, 2].min())
            max_val = max(pd[:, 1].max(), pd[:, 2].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

def create_sequence_summary(pd_files: List[str], output_path: str, sequence_id: str, top_k: int = 50):
    """Create a summary plot showing persistence statistics over time."""
    dates = []
    flood_states = []
    h0_amplitudes = []
    h1_amplitudes = []
    h0_entropies = []
    h1_entropies = []
    h0_wasserstein = []
    h1_wasserstein = []
    h0_consecutive_wasserstein = []
    h1_consecutive_wasserstein = []
    reference_pd = None
    previous_pd = None
    for filepath in pd_files:
        pd = load_persistence_diagram(filepath)
        if pd is not None:
            reference_pd = filter_top_k_per_dim(pd, top_k)
            break
    for i, filepath in enumerate(pd_files):
        pd = load_persistence_diagram(filepath)
        if pd is None:
            continue
        pd = filter_top_k_per_dim(pd, top_k)
            
        date = extract_date_from_filename(filepath)
        flood_state = extract_flood_state_from_filename(filepath)
        dates.append(date)
        flood_states.append(flood_state)
        for dim, dim_amplitudes, dim_entropies, dim_wasserstein, dim_consecutive in [
            (0, h0_amplitudes, h0_entropies, h0_wasserstein, h0_consecutive_wasserstein), 
            (1, h1_amplitudes, h1_entropies, h1_wasserstein, h1_consecutive_wasserstein)
        ]:
            dim_features = pd[pd[:, 0] == dim]
            if len(dim_features) > 0:
                lifetimes = dim_features[:, 2] - dim_features[:, 1]
                amplitude = np.max(lifetimes)
                if len(lifetimes) > 1:
                    total_persistence = np.sum(lifetimes)
                    probs = lifetimes / total_persistence
                    entropy = -np.sum(probs * np.log(probs + 1e-12))
                else:
                    entropy = 0.0
            else:
                amplitude = 0.0
                entropy = 0.0
            if reference_pd is not None:
                wasserstein = wasserstein_distance_pds(reference_pd, pd, dim)
            else:
                wasserstein = 0.0
            if previous_pd is not None:
                consecutive_wasserstein = wasserstein_distance_pds(previous_pd, pd, dim)
            else:
                consecutive_wasserstein = 0.0
            
            dim_amplitudes.append(amplitude)
            dim_entropies.append(entropy)
            dim_wasserstein.append(wasserstein)
            dim_consecutive.append(consecutive_wasserstein)
        previous_pd = pd
    if not dates:
        print("No valid persistence diagrams found for summary")
        return
    flood_start_indices = []
    for i in range(1, len(flood_states)):
        if flood_states[i] == 1 and flood_states[i-1] == 0:
            flood_start_indices.append(i)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    x_pos = range(len(dates))
    ax1.plot(x_pos, h0_amplitudes, 'b-o', label='H₀ Amplitude', linewidth=2, markersize=4)
    ax1.plot(x_pos, h1_amplitudes, 'r-o', label='H₁ Amplitude', linewidth=2, markersize=4)
    for flood_idx in flood_start_indices:
        ax1.axvline(x=flood_idx, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Flood Start' if flood_idx == flood_start_indices[0] else "")
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Top K Persistence Amplitude')
    ax1.set_title(f'Persistence Amplitudes Over Time - Sequence {sequence_id}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x_pos[::max(1, len(x_pos)//10)])  
    ax1.set_xticklabels([dates[i] for i in x_pos[::max(1, len(x_pos)//10)]], rotation=45)
    ax2.plot(x_pos, h0_entropies, 'b-o', label='H₀ Entropy', linewidth=2, markersize=4)
    ax2.plot(x_pos, h1_entropies, 'r-o', label='H₁ Entropy', linewidth=2, markersize=4)
    for flood_idx in flood_start_indices:
        ax2.axvline(x=flood_idx, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Flood Start' if flood_idx == flood_start_indices[0] else "")
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Persistence Entropy')
    ax2.set_title(f'Persistence Entropies Over Time - Sequence {sequence_id}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(x_pos[::max(1, len(x_pos)//10)])
    ax2.set_xticklabels([dates[i] for i in x_pos[::max(1, len(x_pos)//10)]], rotation=45)
    ax3.plot(x_pos, h0_wasserstein, 'b-o', label='H₀ Wasserstein', linewidth=2, markersize=4)
    ax3.plot(x_pos, h1_wasserstein, 'r-o', label='H₁ Wasserstein', linewidth=2, markersize=4)
    for flood_idx in flood_start_indices:
        ax3.axvline(x=flood_idx, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Flood Start' if flood_idx == flood_start_indices[0] else "")
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Wasserstein Distance from First Diagram')
    ax3.set_title(f'Wasserstein Distances Over Time - Sequence {sequence_id}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(x_pos[::max(1, len(x_pos)//10)])
    ax3.set_xticklabels([dates[i] for i in x_pos[::max(1, len(x_pos)//10)]], rotation=45)
    ax4.plot(x_pos[1:], h0_consecutive_wasserstein[1:], 'b-o', label='H₀ Consecutive', linewidth=2, markersize=4)
    ax4.plot(x_pos[1:], h1_consecutive_wasserstein[1:], 'r-o', label='H₁ Consecutive', linewidth=2, markersize=4)
    for flood_idx in flood_start_indices:
        if flood_idx > 0:  
            ax4.axvline(x=flood_idx, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Flood Start' if flood_idx == flood_start_indices[0] else "")
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Consecutive Wasserstein Distance')
    ax4.set_title(f'Consecutive Wasserstein Distances - Sequence {sequence_id}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(x_pos[1:][::max(1, len(x_pos[1:])//10)])
    ax4.set_xticklabels([dates[i] for i in x_pos[1:][::max(1, len(x_pos[1:])//10)]], rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Summary plot saved to: {output_path}")
    if flood_start_indices:
        print(f"Flood events detected in sequence {sequence_id}:")
        for idx in flood_start_indices:
            print(f"  - Flood starts at index {idx}: {dates[idx]}")
    else:
        print(f"No flood events detected in sequence {sequence_id}")
    if len(h0_consecutive_wasserstein) > 1:
        avg_h0_change = np.mean(h0_consecutive_wasserstein[1:])
        max_h0_change = np.max(h0_consecutive_wasserstein[1:])
        print(f"H₀ consecutive changes - Average: {avg_h0_change:.4f}, Max: {max_h0_change:.4f}")
    
    if len(h1_consecutive_wasserstein) > 1:
        avg_h1_change = np.mean(h1_consecutive_wasserstein[1:])
        max_h1_change = np.max(h1_consecutive_wasserstein[1:])
        print(f"H₁ consecutive changes - Average: {avg_h1_change:.4f}, Max: {max_h1_change:.4f}")

def process_sequence(pd_folder: str, output_dir: str, sequence_id: str, top_k: int = 50):
    """Process all persistence diagrams for a sequence (simplified: no modality/filtration subdirs).
    """
    print(f"Processing sequence {sequence_id} from {pd_folder}")
    seq_root = os.path.join(output_dir, f"seq_{sequence_id}")
    individual_dir = os.path.join(seq_root, "individual_pds")
    os.makedirs(individual_dir, exist_ok=True)
    os.makedirs(seq_root, exist_ok=True)
    pd_files = find_sequence_files(pd_folder, sequence_id)
    if not pd_files:
        print(f"No persistence diagram files found for sequence {sequence_id}")
        return
    print(f"Found {len(pd_files)} persistence diagrams for sequence {sequence_id}")
    all_births = []
    all_deaths = []
    for filepath in pd_files:
        pd = load_persistence_diagram(filepath)
        if pd is None:
            continue
        pd = filter_top_k_per_dim(pd, top_k)
        if pd.size > 0:
            all_births.extend(pd[:, 1])
            all_deaths.extend(pd[:, 2])
    if all_births and all_deaths:
        global_min = min(min(all_births), min(all_deaths))
        global_max = max(max(all_births), max(all_deaths))
        xlim = (global_min, global_max)
        ylim = (global_min, global_max)
    else:
        xlim = ylim = None
    for i, filepath in enumerate(pd_files):
        print(f"Processing {i+1}/{len(pd_files)}: {os.path.basename(filepath)}")
        
        pd = load_persistence_diagram(filepath)
        if pd is None:
            continue
        pd = filter_top_k_per_dim(pd, top_k)
        date = extract_date_from_filename(filepath)
        flood_state = extract_flood_state_from_filename(filepath)
        flood_label = "FLOOD" if flood_state == 1 else "NO FLOOD"
        basename = os.path.basename(filepath)
        title = f"PD - {basename}\nDate: {date} ({flood_label})"
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plot_persistence_diagram(pd, title, ax, top_k=top_k, xlim=xlim, ylim=ylim)
        output_filename = f"{os.path.splitext(basename)[0]}_pd.png"
        output_path = os.path.join(individual_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        if pd.size > 0:
            h0_count = np.sum(pd[:, 0] == 0)
            h1_count = np.sum(pd[:, 0] == 1)
            lifetimes = pd[:, 2] - pd[:, 1]
            total_persistence = np.sum(lifetimes)
            max_lifetime = np.max(lifetimes) if len(lifetimes) > 0 else 0
            
            print(f"  H₀: {h0_count}, H₁: {h1_count}, Total persistence: {total_persistence:.3f}, Max lifetime: {max_lifetime:.3f}, Flood: {flood_label}")
        else:
            print(f"  Empty persistence diagram, Flood: {flood_label}")
    summary_path = os.path.join(seq_root, f"seq_{sequence_id}_summary.png")
    create_sequence_summary(pd_files, summary_path, sequence_id, top_k=top_k)
    print(f"Completed sequence {sequence_id}")
    print(f"Individual plots saved to: {individual_dir}")
    print(f"Summary plot saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize persistence diagrams from .npy files')
    parser.add_argument('pd_folder', 
                       help='Folder containing .npy persistence diagram files')
    parser.add_argument('--sequence', default='1', 
                       help='Sequence ID to process (default: 1)')
    parser.add_argument('--output-dir', default='pd_visualizations', 
                       help='Output directory for visualizations (default: pd_visualizations)')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top k features per dimension to plot (default: 50)')
    args = parser.parse_args()
    if not os.path.exists(args.pd_folder):
        print(f"Error: Persistence diagram folder {args.pd_folder} does not exist")
        return
    os.makedirs(args.output_dir, exist_ok=True)
    process_sequence(args.pd_folder, args.output_dir, args.sequence, top_k=args.top_k)

if __name__ == "__main__":
    main()
