import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import torch
import matplotlib.pyplot as plt
import os

def average_linkage_distance(embeddings):
    if len(embeddings) < 2:
        return 0
    pairwise_distances = pdist(embeddings)
    linkage_matrix = linkage(pairwise_distances, method='average')
    return linkage_matrix[-1, 2]

# Store average linkage distances for each layer across epochs
epochs = 300
avg_distances_layer1 = []
avg_distances_layer2 = []
avg_distances_layer3 = []
avg_distances_layer4 = []
# avg_distances_layer5 = []

device = 'cpu'

for epoch in range(epochs):
    emb_dict = torch.load(f'./embeddings_for_ht_mutag_128_32_1000_epochs/embeddings_epoch_{epoch}.pt')

    space1 = emb_dict['node_embs'].detach().to(device)
    space2 = emb_dict['layer_0'].detach().to(device)
    space3 = emb_dict['layer_1'].detach().to(device)
    space4 = emb_dict['layer_2'].detach().to(device)
    # space5 = emb_dict['last_layer'].detach().to(device)

    category1_indices = torch.load(f"./indices_per_batch_for_ht_mutag_128_32_1000_epochs/category1_indices_epoch_{epoch}.pt")
    category2_indices = torch.load(f"./indices_per_batch_for_ht_mutag_128_32_1000_epochs/category2_indices_epoch_{epoch}.pt")
    category3_indices = torch.load(f"./indices_per_batch_for_ht_mutag_128_32_1000_epochs/category3_indices_epoch_{epoch}.pt")

    # Combine cat1 and cat3
    space1_combined = torch.cat([space1[category1_indices], space1[category3_indices]], dim=0).cpu().numpy()
    space2_combined = torch.cat([space2[category1_indices], space2[category3_indices]], dim=0).cpu().numpy()
    space3_combined = torch.cat([space3[category1_indices], space3[category3_indices]], dim=0).cpu().numpy()
    space4_combined = torch.cat([space4[category1_indices], space4[category3_indices]], dim=0).cpu().numpy()
    # space5_combined = torch.cat([space5[category1_indices], space5[category3_indices]], dim=0).cpu().numpy()

    # Compute and store distances
    avg_distances_layer1.append(average_linkage_distance(space1_combined))
    avg_distances_layer2.append(average_linkage_distance(space2_combined))
    avg_distances_layer3.append(average_linkage_distance(space3_combined))
    avg_distances_layer4.append(average_linkage_distance(space4_combined))
    # avg_distances_layer5.append(average_linkage_distance(space5_combined))

print(np.mean(avg_distances_layer1))
print(np.mean(avg_distances_layer2))
print(np.mean(avg_distances_layer3))
print(np.mean(avg_distances_layer4))

# Plot
plt.figure(figsize=(10, 6))
x_epochs = list(range(epochs))

plt.plot(x_epochs, avg_distances_layer1, label="Layer 1", linewidth=1.8)
plt.plot(x_epochs, avg_distances_layer2, label="Layer 2", linewidth=1.8)
plt.plot(x_epochs, avg_distances_layer3, label="Layer 3", linewidth=1.8)
plt.plot(x_epochs, avg_distances_layer4, label="Layer 4", linewidth=1.8)
# plt.plot(x_epochs, avg_distances_layer5, label="Layer 5", linewidth=1.8)

plt.title('Average Linkage Distance Between Category 1 and Category 3')
plt.xlabel('Epoch')
plt.ylabel('Average Linkage Distance')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

# Save plot
os.makedirs("./similarity_logs", exist_ok=True)
plt.savefig("./similarity_logs/avg_linkage_over_epochs_cat_1_3.png")
plt.show()
