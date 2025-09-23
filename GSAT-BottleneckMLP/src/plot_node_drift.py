from calendar import EPOCH
import pickle
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from networkx.algorithms.bipartite import latapy_clustering
import torch
import os
import glob

def calc_node_drift_per_epoch(all_embs, num_epochs):
    """Convert embeddings list to epoch dict and calculate drift per epoch"""
    # Group embeddings by epoch
    embeddings_dict_epoch = defaultdict(list)
    print(f"All embs: {len(all_embs)}")
    batches_per_epoch = len(all_embs) // num_epochs
    print(f"Batches per epoch: {batches_per_epoch}")
    
    for i in range(len(all_embs)):
        emb = all_embs[i][0]
        data = all_embs[i][1]
        att = None
        if len(all_embs[i]) == 3:
            att = all_embs[i][2]
            if i == 0:
                print('ATT LA YAPIYORUZ')
        epoch = i // batches_per_epoch
        embeddings_dict_epoch[epoch].append((emb, data, att))
    
    prev_imp_centroid = None
    prev_unimp_centroid = None
    imp_drift_per_epoch = []
    unimp_drift_per_epoch = []

    for epoch in range(num_epochs):
        # Collect all embeddings and masks for this epoch
        all_embs_epoch = []
        all_imp_mask = []

        for emb, data, att in embeddings_dict_epoch[epoch]:
            all_embs_epoch.append(emb)
            
            if att is not None:
                # Use node-level attention values above 0.5 to find important nodes
                important_nodes = torch.where(att > 0.5)[0]
            else:
                # Fallback to original method using edge labels
                expl_edge_mask = data.edge_label == 1
                important_edges = data.edge_index[:, expl_edge_mask]
                important_nodes = torch.unique(important_edges)
            
            num_nodes = emb.size(0)
            mask = torch.zeros(num_nodes, dtype=torch.bool, device=emb.device)
            if len(important_nodes) > 0:
                mask[important_nodes] = True
            all_imp_mask.append(mask)

        # Concatenate all embeddings and masks
        emb_epoch = torch.cat(all_embs_epoch, dim=0)
        imp_mask_epoch = torch.cat(all_imp_mask, dim=0)
        unimp_mask_epoch = ~imp_mask_epoch

        # Compute centroids for this epoch
        cur_imp_centroid = emb_epoch[imp_mask_epoch].mean(dim=0) if imp_mask_epoch.any() else None
        cur_unimp_centroid = emb_epoch[unimp_mask_epoch].mean(dim=0) if unimp_mask_epoch.any() else None

        # Compute drift relative to previous epoch
        imp_drift = 0.0
        unimp_drift = 0.0
        
        if prev_imp_centroid is not None and cur_imp_centroid is not None:
            imp_drift = torch.norm(cur_imp_centroid - prev_imp_centroid).item()
            
        if prev_unimp_centroid is not None and cur_unimp_centroid is not None:
            unimp_drift = torch.norm(cur_unimp_centroid - prev_unimp_centroid).item()

        # Store drift for this epoch (skip epoch 0 as there's no previous epoch)
        if epoch > 0:
            imp_drift_per_epoch.append(imp_drift)
            unimp_drift_per_epoch.append(unimp_drift)

        # Update previous centroids
        prev_imp_centroid = cur_imp_centroid
        prev_unimp_centroid = cur_unimp_centroid

    return imp_drift_per_epoch, unimp_drift_per_epoch

def main():
    parser = argparse.ArgumentParser(description='Plot node drift per epoch across seeds')
    parser.add_argument('--folder_path', type=str, required=True, help='Path to folder containing embedding pickle files')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--layer', type=str, required=True, help='Layer to analyze')
    parser.add_argument('--output_path', type=str, default='node_drift_plot.png', help='Output path for the plot')
    args = parser.parse_args()
    print(f'Folder path: {args.folder_path}')
    print(f'Number of epochs: {args.num_epochs}')
    print(f'Layer: {args.layer}')
    layer = args.layer
    
    # Check if folder exists
    if not os.path.exists(args.folder_path):
        print(f"[ERROR] Folder not found: {args.folder_path}")
        return
    
    # Find all pickle files in the folder
    pickle_files = glob.glob(os.path.join(args.folder_path, "*.pkl"))
    if not pickle_files:
        print(f"[ERROR] No pickle files found in folder: {args.folder_path}")
        return
    
    print(f"[INFO] Found {len(pickle_files)} pickle files in folder")
    
    # Process each file and collect drift per epoch
    all_imp_drift_per_epoch = []
    all_unimp_drift_per_epoch = []
    
    for pkl_file in pickle_files:
        filename = os.path.basename(pkl_file)
        print(f"\n[INFO] Processing file: {filename}")
        
        with open(pkl_file, 'rb') as f:
            all_embs = pickle.load(f)
        
        print(f"Available layers: {list(all_embs.keys())}")
        
        imp_drift_per_epoch, unimp_drift_per_epoch = calc_node_drift_per_epoch(all_embs[layer], args.num_epochs)
        all_imp_drift_per_epoch.append(imp_drift_per_epoch)
        all_unimp_drift_per_epoch.append(unimp_drift_per_epoch)
    
    # Convert to numpy arrays for easier computation
    all_imp_drift_per_epoch = np.array(all_imp_drift_per_epoch)
    all_unimp_drift_per_epoch = np.array(all_unimp_drift_per_epoch)
    
    # Calculate average drift per epoch across all seeds
    avg_imp_drift_per_epoch = np.mean(all_imp_drift_per_epoch, axis=0)
    avg_unimp_drift_per_epoch = np.mean(all_unimp_drift_per_epoch, axis=0)
    
    # Calculate standard deviation for error bars
    std_imp_drift_per_epoch = np.std(all_imp_drift_per_epoch, axis=0)
    std_unimp_drift_per_epoch = np.std(all_unimp_drift_per_epoch, axis=0)
    
    # Create epochs array (starting from epoch 1 since we skip epoch 0)
    epochs = np.arange(1, args.num_epochs)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    plt.errorbar(epochs, avg_imp_drift_per_epoch, yerr=std_imp_drift_per_epoch, 
                label='Important Nodes', marker='o', markersize=3, capsize=2, capthick=1, alpha=0.7)
    plt.errorbar(epochs, avg_unimp_drift_per_epoch, yerr=std_unimp_drift_per_epoch, 
                label='Unimportant Nodes', marker='s', markersize=3, capsize=2, capthick=1, alpha=0.7)
    
    plt.xlabel('Epoch')
    plt.ylabel('Average Node Centroid Drift')
    plt.title(f'GSAT Node Centroid Drift per Epoch (Layer: {layer})\nAveraged across {len(pickle_files)} seeds')
    plt.legend()
    plt.grid(True, alpha=0.3)
        # --- Save the plot in node_drift_plots folder ---
    # Ensure save directory exists
    save_dir = "node_drift_plots"
    os.makedirs(save_dir, exist_ok=True)

    # Clean folder name (in case it has slashes)
    folder_name = os.path.basename(os.path.normpath(args.folder_path))

    # Construct filename
    save_path = os.path.join(save_dir, f"node_drift_{folder_name}.png")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[INFO] Plot saved to: {save_path}")

    # Save the plot
    # plt.tight_layout()
    # plt.savefig(args.output_path, dpi=300, bbox_inches='tight')
    # print(f"\n[INFO] Plot saved to: {args.output_path}")
    
    # Print summary statistics
    print(f"\n[SUMMARY]")
    print(f"Number of seeds: {len(pickle_files)}")
    print(f"Epochs analyzed: {len(epochs)} (epochs 1 to {args.num_epochs-1})")
    print(f"Average important node drift per epoch: {np.mean(avg_imp_drift_per_epoch):.6f}")
    print(f"Average unimportant node drift per epoch: {np.mean(avg_unimp_drift_per_epoch):.6f}")

if __name__ == '__main__':
    main()
