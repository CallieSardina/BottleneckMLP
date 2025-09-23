import pickle
import argparse
from collections import defaultdict

import torch
import os
import glob
import matplotlib.pyplot as plt


def calc_linkage_distance(all_embs, num_epochs):
    """Calculate avg linkage distance (important vs unimportant centroids) over epochs"""
    # Group embeddings by epoch
    embeddings_dict_epoch = defaultdict(list)
    batches_per_epoch = len(all_embs) // num_epochs
    
    for i, emb_data in enumerate(all_embs):
        emb = emb_data[0]
        data = emb_data[1]
        att = None
        if len(emb_data) == 3:
            att = emb_data[2]
        epoch = i // batches_per_epoch
        embeddings_dict_epoch[epoch].append((emb, data, att))
    
    epoch_linkage = []

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

        # Concatenate embeddings & masks
        emb_epoch = torch.cat(all_embs_epoch, dim=0)
        imp_mask_epoch = torch.cat(all_imp_mask, dim=0)
        unimp_mask_epoch = ~imp_mask_epoch

        if imp_mask_epoch.any() and unimp_mask_epoch.any():
            imp_centroid = emb_epoch[imp_mask_epoch].mean(dim=0)
            unimp_centroid = emb_epoch[unimp_mask_epoch].mean(dim=0)
            linkage_dist = torch.norm(imp_centroid - unimp_centroid).item()
        else:
            linkage_dist = float('nan')  # skip if no nodes
        epoch_linkage.append(linkage_dist)

    return epoch_linkage


def main():
    parser = argparse.ArgumentParser(description='Analyze node centroid linkage across epochs')
    parser.add_argument('--folder_path', type=str, required=True, help='Path to folder containing embedding pickle files')
    args = parser.parse_args()
    args.num_epochs = 300

    # Check folder
    if not os.path.exists(args.folder_path):
        print(f"[ERROR] Folder not found: {args.folder_path}")
        return
    
    pickle_files = glob.glob(os.path.join(args.folder_path, "*.pkl"))
    if not pickle_files:
        print(f"[ERROR] No pickle files found in folder: {args.folder_path}")
        return
    
    print(f"[INFO] Found {len(pickle_files)} pickle files in folder")

    # For storing results per layer across seeds
    layer_distances_all_seeds = defaultdict(list)

    for pkl_file in pickle_files:
        filename = os.path.basename(pkl_file)
        print(f"\n[INFO] Processing file: {filename}")
        
        with open(pkl_file, 'rb') as f:
            all_embs = pickle.load(f)
        
        # Process all layers that start with "layer_"
        for layer_name, layer_embs in all_embs.items():
            if not layer_name in ["layer_0", "layer_1", "layer_2", "layer_3"]:
                continue
            print(f"  -> Processing layer: {layer_name}")
            linkage_curve = calc_linkage_distance(layer_embs, args.num_epochs)
            layer_distances_all_seeds[layer_name].append(linkage_curve)

    # Average across seeds
    avg_layer_curves = {}
    for layer_name, curves in layer_distances_all_seeds.items():
        stacked = torch.tensor(curves)  # shape: [num_seeds, num_epochs]
        avg_curve = stacked.nanmean(dim=0).tolist()
        avg_layer_curves[layer_name] = avg_curve

    # Plot
    plt.figure(figsize=(8, 5))
    for layer_name, avg_curve in avg_layer_curves.items():
        plt.plot(range(args.num_epochs), avg_curve, label=layer_name)
    
    plt.xlabel("Epoch")
    plt.ylabel("Avg Linkage Distance (||imp - unimp||)")
    plt.title("GSAT Important vs Unimportant Node Centroid Distance Over Epochs")
    plt.legend()
    plt.tight_layout()
    
    # Save plot instead of showing it
    plot_filename = os.path.join(args.folder_path, "node_linkage_plot.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"[INFO] Plot saved to: {plot_filename}")
    plt.close()


if __name__ == '__main__':
    main()
