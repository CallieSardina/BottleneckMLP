import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from scipy.linalg import pinv
from collections import defaultdict
from torch_geometric.data import Data
import matplotlib.pyplot as plt

# Function to compute the commute time matrix
def compute_commute_time_matrix(G):
    L = nx.laplacian_matrix(G).todense()
    L_plus = pinv(L)
    vol_G = G.number_of_edges() * 2
    n = G.number_of_nodes()
    CT = np.zeros((n, n))
    for u in range(n):
        for v in range(n):
            CT[u, v] = vol_G * (L_plus[u, u] + L_plus[v, v] - 2 * L_plus[u, v])
    return CT

# Function to assign edges to categories based on node indices
def get_edge_categories(data, cat1, cat2, cat3):
    edge_index = data['edge_index'].cpu().numpy()
    edge_list = [(int(u), int(v)) for u, v in edge_index.T]

    def in_cat(u, v, cat_nodes):
        return u in cat_nodes and v in cat_nodes

    edge_cats = defaultdict(list)
    for u, v in edge_list:
        if in_cat(u, v, cat1):
            edge_cats["cat1"].append((u, v))
        elif in_cat(u, v, cat2):
            edge_cats["cat2"].append((u, v))
        elif in_cat(u, v, cat3):
            edge_cats["cat3"].append((u, v))
    return edge_cats

# Function to compute drift scores based on embeddings and edge categories
def compute_drift_scores(embeddings, edge_categories, CT):
    emb_np = embeddings.cpu().detach().numpy()
    drift_dict = {}
    for cat, edge_list in edge_categories.items():
        scores = []
        for u, v in edge_list:
            drift = np.linalg.norm(emb_np[u] - emb_np[v]) / (CT[u, v] + 1e-6)
            scores.append(drift)
        drift_dict[cat] = scores
    return drift_dict



spaces = ['layer_0', 'layer_1', 'layer_2']
drift_records = []

for epoch in range(300): 
    print(f"\nProcessing epoch {epoch}...")
    emb_dict = torch.load(f'./embeddings_for_drift_test/embeddings_epoch_{epoch}.pt')

    cumulative_nodes = 0

    for graph_id in range(8): 
        cat1 = torch.load(f'./indices_per_graph_for_drift_test/category1_graph_{graph_id}_epoch_{epoch}.pt')
        cat2 = torch.load(f'./indices_per_graph_for_drift_test/category2_graph_{graph_id}_epoch_{epoch}.pt')
        cat3 = torch.load(f'./indices_per_graph_for_drift_test/category3_graph_{graph_id}_epoch_{epoch}.pt')

        print(cat1)
        print(cat2)
        print(cat3)


        data = torch.load('./graphs/graph_0_epoch_0_batch_0.pt',  map_location='cpu')
     
        print(tmp)

        graph_path = f'./graphs/graph_{epoch}_{graph_id}.pt'
        if not os.path.exists(graph_path):
            print(f"Warning: {graph_path} not found.")
            continue
        data = torch.load(graph_path)

        # Get edge categories for the current graph
        edge_cats = get_edge_categories(data, cat1.tolist(), cat2.tolist(), cat3.tolist())
        data = Data(edge_index=data.edge_index, num_nodes=data.num_nodes)
        G = to_networkx(data, to_undirected=True)
        CT = compute_commute_time_matrix(G)

        num_nodes = data.num_nodes

        start_idx = cumulative_nodes 
        end_idx = start_idx + num_nodes 

        # Track drift scores for each layer
        for layer in spaces:
            # node_counts = [torch.load(f'./graphs/graph_{i}.pt')['num_nodes'] for i in range(24)]
            # assert sum(node_counts) == emb_dict[layer].shape[0]
            node_embs = emb_dict[layer][start_idx:end_idx]  
            print(emb_dict[layer].shape)
            print(start_idx, end_idx)
            drift_scores = compute_drift_scores(node_embs, edge_cats, CT)

            for cat, scores in drift_scores.items():
                for score in scores:
                    drift_records.append({
                        "epoch": epoch,
                        "layer": layer,
                        "category": cat,
                        "drift": score
                    })

        cumulative_nodes += num_nodes
        print("Expected embedding size:", emb_dict[layer].shape[0])
        print("Final cumulative_nodes:", cumulative_nodes)
    print(tmp)

df = pd.DataFrame(drift_records)
grouped = df.groupby(["epoch", "layer", "category"])["drift"].mean().reset_index()
categories = grouped["category"].unique()
num_categories = len(categories)

fig, axes = plt.subplots(nrows=1, ncols=num_categories, figsize=(6 * num_categories, 5), sharey=True)

if num_categories == 1:
    axes = [axes]

for ax, category in zip(axes, categories):
    category_data = grouped[grouped["category"] == category]

    for layer, group in category_data.groupby("layer"):
        ax.plot(group["epoch"].values, group["drift"].values, label=f"Layer {layer}")

    ax.set_title(f"Category: {category}")
    ax.set_xlabel("Epoch")
    ax.grid(True)
    ax.legend()

axes[0].set_ylabel("Mean Drift")
plt.suptitle("Drift Over Epochs by Layer (per Category)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.savefig("./similarity_logs/drift_by_category.png")
plt.show()


