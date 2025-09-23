import pickle
import argparse
import numpy as np
import torch
import os
import glob
from collections import defaultdict
from scipy.stats import ttest_ind

def gaussian_entropy(cov_matrix):
    """Compute Gaussian entropy given covariance matrix"""
    d = cov_matrix.shape[0]
    det = np.linalg.det(cov_matrix + 1e-6 * np.eye(d))
    return 0.5 * np.log((2 * np.pi * np.e) ** d * det)

def extract_epoch_embeddings(all_embs, num_epochs, epoch_id):
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

    all_embs_epoch = []
    all_imp_mask = []
    for emb, data, att in embeddings_dict_epoch[epoch_id]:
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

    emb_epoch = torch.cat(all_embs_epoch, dim=0).cpu().numpy()
    imp_mask_epoch = torch.cat(all_imp_mask, dim=0).cpu().numpy()
    unimp_mask_epoch = ~imp_mask_epoch
    return emb_epoch, imp_mask_epoch, unimp_mask_epoch

def analyze_embeddings(emb, imp_mask, unimp_mask):
    # Check if we have any important or unimportant nodes
    if not imp_mask.any():
        print("[WARNING] No important nodes found, returning zeros for important metrics")
        return 0.0, 0.0, 0.0, 0.0, 1.0, None, None
    
    if not unimp_mask.any():
        print("[WARNING] No unimportant nodes found, returning zeros for unimportant metrics")
        return 0.0, 0.0, 0.0, 0.0, 1.0, None, None
    
    imp_emb = emb[imp_mask]
    unimp_emb = emb[unimp_mask]

    # Additional safety check for minimum number of samples needed for covariance
    if imp_emb.shape[0] < 2:
        print("[WARNING] Not enough important nodes for covariance calculation")
        return 0.0, 0.0, 0.0, 0.0, 1.0, None, None
        
    if unimp_emb.shape[0] < 2:
        print("[WARNING] Not enough unimportant nodes for covariance calculation")
        return 0.0, 0.0, 0.0, 0.0, 1.0, None, None

    cov_imp = np.cov(imp_emb.T)
    cov_unimp = np.cov(unimp_emb.T)

    H_imp = gaussian_entropy(cov_imp)
    H_unimp = gaussian_entropy(cov_unimp)

    var_imp = np.trace(cov_imp)
    var_unimp = np.trace(cov_unimp)

    _, pval = ttest_ind(np.diag(cov_unimp), np.diag(cov_imp), equal_var=False)

    return H_imp, H_unimp, var_imp, var_unimp, pval, cov_imp, cov_unimp


def main():
    parser = argparse.ArgumentParser(description='Entropy analysis of node embeddings')
    parser.add_argument('--folder_path', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, required=True)
    parser.add_argument('--layer', type=str, required=True)
    parser.add_argument('--output_prefix', type=str, default=None)
    args = parser.parse_args()

    if args.output_prefix is None:
        folder_name = os.path.basename(args.folder_path.rstrip('/'))
        if folder_name.startswith('embeddings_'):
            folder_name = folder_name[len('embeddings_'):]
        args.output_prefix = folder_name

    print(f"Folder path: {args.folder_path}")

    pickle_files = glob.glob(os.path.join(args.folder_path, "*.pkl"))
    if not pickle_files:
        print("[ERROR] No pickle files found")
        return

    epochs_to_check = [0, args.num_epochs // 4, args.num_epochs - 1]

    for epoch in epochs_to_check:
        print(f"\n[INFO] Analyzing epoch {epoch}")
        results_epoch = { "imp_entropy": [], "unimp_entropy": [], "imp_var": [], "unimp_var": [], "pval": [] }

        for seed_id, pkl_file in enumerate(pickle_files):
            with open(pkl_file, 'rb') as f:
                all_embs = pickle.load(f)
            layer_embs = all_embs[args.layer]

            emb_epoch, imp_mask, unimp_mask = extract_epoch_embeddings(layer_embs, args.num_epochs, epoch)
            H_imp, H_unimp, var_imp, var_unimp, pval, cov_imp, cov_unimp = analyze_embeddings(emb_epoch, imp_mask, unimp_mask)

            results_epoch["imp_entropy"].append(H_imp)
            results_epoch["unimp_entropy"].append(H_unimp)
            results_epoch["imp_var"].append(var_imp)
            results_epoch["unimp_var"].append(var_unimp)
            results_epoch["pval"].append(pval)

        print(f"[SUMMARY] Epoch {epoch}")
        for key in results_epoch:
            mean_val = np.mean(results_epoch[key])
            std_val = np.std(results_epoch[key])
            print(f"{key}: {mean_val:.4f} ± {std_val:.4f}")

if __name__ == "__main__":
    main()
