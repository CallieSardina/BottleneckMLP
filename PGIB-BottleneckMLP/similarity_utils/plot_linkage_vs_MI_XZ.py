import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

# === Compute average linkage ===
def average_linkage_distance(embeddings):
    if len(embeddings) < 2:
        return 0
    pairwise_distances = pdist(embeddings)
    linkage_matrix = linkage(pairwise_distances, method='average')
    return linkage_matrix[-1, 2]

# === Extract linkage distances and record valid epochs ===
def compute_linkage_per_layer(embedding_dir, num_layers=4):
    distances = defaultdict(list)
    valid_epochs = []
    device = 'cpu'

    for filename in os.listdir(embedding_dir):
        if not filename.startswith("embeddings_epoch_"):
            continue
        epoch = int(filename.split("_")[-1].split(".")[0])
        try:
            emb_dict = torch.load(os.path.join(embedding_dir, filename), map_location=device)

            space1 = emb_dict['node_embs'].detach().to(device)
            space2 = emb_dict['layer_0'].detach().to(device)
            space3 = emb_dict['layer_1'].detach().to(device)
            space4 = emb_dict['layer_2'].detach().to(device)

            cat1_path = os.path.join(embedding_dir.replace("embeddings", "indices_per_batch"), f"category1_indices_epoch_{epoch}.pt")
            cat3_path = os.path.join(embedding_dir.replace("embeddings", "indices_per_batch"), f"category3_indices_epoch_{epoch}.pt")
            if not os.path.exists(cat1_path) or not os.path.exists(cat3_path):
                continue

            category1_indices = torch.load(cat1_path)
            category3_indices = torch.load(cat3_path)

            space1_combined = torch.cat([space1[category1_indices], space1[category3_indices]], dim=0).cpu().numpy()
            space2_combined = torch.cat([space2[category1_indices], space2[category3_indices]], dim=0).cpu().numpy()
            space3_combined = torch.cat([space3[category1_indices], space3[category3_indices]], dim=0).cpu().numpy()
            space4_combined = torch.cat([space4[category1_indices], space4[category3_indices]], dim=0).cpu().numpy()

            distances[0].append((epoch, average_linkage_distance(space1_combined)))
            distances[1].append((epoch, average_linkage_distance(space2_combined)))
            distances[2].append((epoch, average_linkage_distance(space3_combined)))
            distances[3].append((epoch, average_linkage_distance(space4_combined)))

            valid_epochs.append(epoch)
        except Exception as e:
            print(f"Error processing epoch {epoch}: {e}")

    return distances, set(valid_epochs)

# === Parse MI(X,Z) with epochs ===
def parse_mi_file_layerwise(filepath, expected_layers=4):
    layerwise_mi = defaultdict(list)
    valid_epochs = []

    with open(filepath, 'r') as f:
        for line in f:
            match = re.match(r"Epoch (\d+), MI_XZ: \[([^\]]+)\]", line)
            if match:
                epoch = int(match.group(1))
                values_str = match.group(2)
                values = [float(v.strip()) for v in values_str.split(",")]
                for i in range(min(expected_layers, len(values))):
                    layerwise_mi[i].append((epoch, values[i]))
                valid_epochs.append(epoch)
    return layerwise_mi, set(valid_epochs)

# === Plotting with aligned epochs ===
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

def plot_mi_vs_linkage(mi_data, linkage_data, output_path):
    num_layers = len(linkage_data)

    fig = plt.figure(figsize=(5 * num_layers + 1.5, 4), constrained_layout=True)
    gs = GridSpec(1, num_layers + 1, figure=fig, width_ratios=[1]*num_layers + [0.05])

    cmap = cm.get_cmap('plasma')
    max_epochs = 300
    norm = mcolors.Normalize(vmin=0, vmax=max_epochs - 1)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    axs = [fig.add_subplot(gs[0, i]) for i in range(num_layers)]

    for i in range(num_layers):
        mi_epoch_val = dict(mi_data[i])
        linkage_epoch_val = dict(linkage_data[i])
        common_epochs = sorted(set(mi_epoch_val.keys()) & set(linkage_epoch_val.keys()))

        if not common_epochs:
            print(f"No overlapping epochs for layer {i}")
            continue

        distances = [linkage_epoch_val[e] for e in common_epochs]
        mi_values = [mi_epoch_val[e] for e in common_epochs]
        colors = cmap(norm(range(len(distances))))

        axs[i].scatter(distances, mi_values, c=colors, alpha=0.8, s=20)
        axs[i].set_title(f'Layer {i+1}')
        axs[i].set_xlabel('Avg. Linkage Distance\n(Cat 1 & 3)')
        axs[i].set_ylabel('MI(X; Z)')

    # Add colorbar to last column
    cbar_ax = fig.add_subplot(gs[0, -1])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Epoch')

    fig.suptitle('MI(X; Z) vs Avg. Linkage Distance (Per Layer)', fontsize=14)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()


def plot_layerwise_avg_mi_vs_linkage(mi_data, linkage_data, output_path):
    num_layers = len(linkage_data)
    avg_linkage = []
    avg_mi = []

    for i in range(num_layers):
        linkage_vals = [val for (_, val) in linkage_data[i]]
        mi_vals = [val for (_, val) in mi_data[i]]

        # Align by common epochs
        linkage_epochs = {epoch for (epoch, _) in linkage_data[i]}
        mi_epochs = {epoch for (epoch, _) in mi_data[i]}
        common_epochs = linkage_epochs & mi_epochs

        linkage_dict = dict(linkage_data[i])
        mi_dict = dict(mi_data[i])

        aligned_linkage = [linkage_dict[e] for e in common_epochs]
        aligned_mi = [mi_dict[e] for e in common_epochs]

        if len(common_epochs) == 0:
            print(f"[Layer {i}] No common epochs — skipping.")
            avg_linkage.append(np.nan)
            avg_mi.append(np.nan)
        else:
            avg_linkage.append(np.mean(aligned_linkage))
            avg_mi.append(np.mean(aligned_mi))

    # === Plot ===
    plt.figure(figsize=(6, 5))
    plt.plot(avg_linkage, avg_mi, marker='o', linestyle='-', color='tab:red')

    for i, (x, y) in enumerate(zip(avg_linkage, avg_mi)):
        plt.text(x, y, f'Layer {i}', fontsize=9, ha='right')

    plt.xlabel('Average Linkage Distance\n(Categories 1 & 3)')
    plt.ylabel('Average MI(X; Z)')
    plt.title('Layerwise Avg. MI vs Linkage')
    plt.grid(True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()


# === Main ===
if __name__ == "__main__":
    embedding_dir = './embeddings_for_ht_mutag_128_32_1000_epochs'
    mi_file = './MI_logs/mutag_128_32_nsa.txt'
    output_plot_path = './similarity_logs/mi_vs_linkage_per_layer.png'

    print("Computing linkage distances...")
    linkage_per_layer, linkage_epochs = compute_linkage_per_layer(embedding_dir, num_layers=4)

    print("Parsing MI(X; Z) log...")
    mi_per_layer, mi_epochs = parse_mi_file_layerwise(mi_file, expected_layers=4)

    common_epochs = sorted(linkage_epochs & mi_epochs)
    print(f"Total common epochs: {len(common_epochs)}")

    print("Plotting...")
    plot_mi_vs_linkage(mi_per_layer, linkage_per_layer, output_plot_path)

    summary_plot_path = './similarity_logs/layerwise_avg_mi_vs_linkage.png'
    plot_layerwise_avg_mi_vs_linkage(mi_per_layer, linkage_per_layer, summary_plot_path)
