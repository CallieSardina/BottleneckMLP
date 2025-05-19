import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

device = 'cpu'
epoch = 150  # pick any epoch you want to visualize

embedding_dir = './embeddings'
indices_dir = './indices_per_batch'

# Load embeddings and category indices
emb_dict = torch.load(f'{embedding_dir}/embeddings_epoch_{epoch}.pt', map_location=device)
category1_indices = torch.load(f"{indices_dir}/category1_indices_epoch_{epoch}.pt")
category2_indices = torch.load(f"{indices_dir}/category2_indices_epoch_{epoch}.pt")
category3_indices = torch.load(f"{indices_dir}/category3_indices_epoch_{epoch}.pt")

spaces = ['node_embs', 'layer_0', 'layer_1', 'layer_2']

# Assign colors to each category
category_labels = {
    "cat1": category1_indices,
    "cat2": category2_indices,
    "cat3": category3_indices
}
colors = {
    "cat1": 'tab:red',
    "cat2": 'tab:orange',
    "cat3": 'tab:green'
}

# Create subplots
fig, axs = plt.subplots(1, len(spaces), figsize=(5 * len(spaces), 4))
fig.suptitle(f't-SNE Visualization at Epoch {epoch}', fontsize=16)

for i, space in enumerate(spaces):
    # Collect embeddings and labels
    all_embeddings = []
    all_labels = []

    for cat_name, indices in category_labels.items():
        embs = emb_dict[space][indices].detach().cpu().numpy()
        all_embeddings.append(embs)
        all_labels.extend([cat_name] * len(embs))

    all_embeddings = np.vstack(all_embeddings)

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca')
    reduced = tsne.fit_transform(all_embeddings)

    # Plot points by category
    ax = axs[i]
    for cat_name in category_labels:
        idxs = [j for j, label in enumerate(all_labels) if label == cat_name]
        ax.scatter(reduced[idxs, 0], reduced[idxs, 1], s=10, alpha=0.6, label=cat_name, color=colors[cat_name])

    ax.set_title(space)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'./similarity_logs/tsne_epoch_{epoch}.png')
plt.show()

