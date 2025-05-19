import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import torch
import matplotlib.pyplot as plt
import random
import os

def hsic_between(X, Y, sigma=1.0):
    """Compute HSIC between two [n_samples, dim] matrices."""
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()

    K = rbf_kernel(X, gamma=1 / (2 * sigma ** 2))
    L = rbf_kernel(Y, gamma=1 / (2 * sigma ** 2))

    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H

    return np.trace(Kc @ Lc) / (n ** 2)

# Config
epoch = 150
device = 'cpu'
sample_size = 15

# Load embeddings
emb_dict = torch.load(f'./embeddings_for_ht_mutag_128_32_1000_epochs/embeddings_epoch_{epoch}.pt')

# Layers
node_emb = emb_dict['node_embs'].detach().to(device)
layer_0 = emb_dict['layer_0'].detach().to(device)
layer_1 = emb_dict['layer_1'].detach().to(device)
layer_2 = emb_dict['layer_2'].detach().to(device)
last_layer = emb_dict['last_layer'].detach().to(device)

# Sample nodes
total_nodes = layer_0.shape[0]
indices = random.sample(range(total_nodes), min(sample_size, total_nodes))
num_batches = len(node_emb)
indices = list(range(num_batches - 15, num_batches))

# Subset embeddings
x = node_emb[indices]
z0 = layer_0[indices]
z1 = layer_1[indices]
z2 = layer_2[indices]
z_out = last_layer[indices]

# Compute HSIC values
hsic_vals = [
    hsic_between(x, z0),
    hsic_between(x, z1),
    hsic_between(x, z2),
    hsic_between(x, z_out),
]

# Plot
plt.figure(figsize=(7, 5))
layer_ids = [0, 1, 2, 3]
plt.plot(layer_ids, hsic_vals, marker='o', linestyle='-', linewidth=2, color='steelblue')
plt.title("HSIC Over Layers")
plt.xlabel("Layer")
plt.ylabel("HSIC Value")
plt.xticks(layer_ids)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save
os.makedirs("./similarity_logs", exist_ok=True)
plt.savefig("./similarity_logs/hsic_per_layer_last_epoch.png")
plt.show()


# import numpy as np
# from sklearn.metrics.pairwise import rbf_kernel
# import torch
# import os
# import matplotlib.pyplot as plt
# import random

# def hsic_between(X, Y, sigma=1.0):
#     """Compute HSIC between two [n_samples, dim] matrices."""
#     X = X.detach().cpu().numpy()
#     Y = Y.detach().cpu().numpy()

#     K = rbf_kernel(X, gamma=1 / (2 * sigma ** 2))
#     L = rbf_kernel(Y, gamma=1 / (2 * sigma ** 2))

#     n = K.shape[0]
#     H = np.eye(n) - np.ones((n, n)) / n
#     Kc = H @ K @ H
#     Lc = H @ L @ H

#     return np.trace(Kc @ Lc) / (n ** 2)

# # Setup
# epochs = 300
# device = 'cpu'
# sample_size = 100  # adjust based on your node count and memory

# hsic_l0 = []
# hsic_l1 = []
# hsic_l2 = []

# for epoch in range(epochs):
#     try:
#         emb_dict = torch.load(f'./embeddings_for_ht_mutag_128_32_1000_epochs/embeddings_epoch_{epoch}.pt')

#         # Get layers
#         layer_0 = emb_dict['layer_0'].detach().to(device)
#         layer_1 = emb_dict['layer_1'].detach().to(device)
#         layer_2 = emb_dict['layer_2'].detach().to(device)
#         last_layer = emb_dict['last_layer'].detach().to(device)

#         # Sample random indices (shared across all layers)
#         total_nodes = layer_0.shape[0]
#         indices = random.sample(range(total_nodes), min(sample_size, total_nodes))

#         # Subset
#         x0 = layer_0[indices]
#         x1 = layer_1[indices]
#         x2 = layer_2[indices]
#         z = last_layer[indices]

#         # Compute HSIC
#         hsic_l0.append(hsic_between(x0, z))
#         hsic_l1.append(hsic_between(x1, z))
#         hsic_l2.append(hsic_between(x2, z))

#     except Exception as e:
#         print(f"Error at epoch {epoch}: {e}")
#         hsic_l0.append(None)
#         hsic_l1.append(None)
#         hsic_l2.append(None)

# # Plot
# plt.figure(figsize=(10, 6))
# x_epochs = list(range(epochs))

# plt.plot(x_epochs, hsic_l0, label="HSIC: Layer 0 vs Last", linewidth=1.8)
# plt.plot(x_epochs, hsic_l1, label="HSIC: Layer 1 vs Last", linewidth=1.8)
# plt.plot(x_epochs, hsic_l2, label="HSIC: Layer 2 vs Last", linewidth=1.8)

# plt.title("HSIC Between Layers and Final Layer Over Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("HSIC")
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend()
# plt.tight_layout()

# # Save
# os.makedirs("./similarity_logs", exist_ok=True)
# plt.savefig("./similarity_logs/hsic_random_sample_layers_vs_last.png")
# plt.show()
