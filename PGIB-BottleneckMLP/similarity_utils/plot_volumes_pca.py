import matplotlib.pyplot as plt
import numpy as np

# Categories and layers
categories = ["cat1", "cat2", "cat3"]
layer_labels = ["node_embs", "layer_0", "layer_1", "layer_2", "last_layer"]
num_layers = len(layer_labels)

# Function to parse a volume file
def load_volumes(filepath):
    volumes = [[] for _ in range(num_layers)]
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split("Vol:")[-1].split(",")
            for i, val in enumerate(parts):
                try:
                    volumes[i].append(float(val))
                except ValueError:
                    volumes[i].append(None)
    return volumes

# Initialize results dictionary
avg_volumes = {cat: [] for cat in categories}

# Compute average convex hull volumes for each category and layer
for cat in categories:
    path = f"./similarity_logs/{cat}_conv_hull_volume_pca.txt"
    volumes = load_volumes(path)

    # Compute averages, ignoring None and avoiding division by zero
    cat_avg = []
    for layer in range(num_layers):
        # Filter out None values from the volume list for the current layer
        valid_volumes = [v for v in volumes[layer] if v is not None]
        
        # Only compute the average if there are valid volumes, otherwise set it to None
        if valid_volumes:
            avg = sum(valid_volumes) / len(valid_volumes)
        else:
            avg = None  # Or you can use 0 or any other value that makes sense for your case
        
        cat_avg.append(avg)

    avg_volumes[cat] = cat_avg

# Plotting - Bar graph for average convex hull volumes
fig, axs = plt.subplots(figsize=(10, 6))

# X positions for each layer
x = np.arange(len(layer_labels))

# Width of the bars
width = 0.2

# Bar plots for each category
for i, cat in enumerate(categories):
    cat_avg = avg_volumes[cat]
    
    # Ensure to handle missing or None values gracefully
    # Replace None with 0 or another value if necessary
    cat_avg = [v if v is not None else 0 for v in cat_avg]
    
    axs.bar(x + i * width, cat_avg, width, label=cat)

# Adding labels, title, and legend
axs.set_xlabel('Layers')
axs.set_ylabel('Average Volume')
axs.set_title('Average Convex Hull Volumes per Layer and Category')
axs.set_xticks(x + width)
axs.set_xticklabels(layer_labels)
axs.legend()

# Show gridlines for readability
axs.grid(True, linestyle='--', alpha=0.5)

# Show the plot
plt.tight_layout()
plt.savefig('./similarity_logs/avg_volumes_bar_graph.png')
plt.show()
# import matplotlib.pyplot as plt

# # File paths
# categories = ["cat1", "cat2", "cat3"]
# layer_labels = ["node_embs", "layer_0", "layer_1", "layer_2", "last_layer"]
# num_layers = len(layer_labels)
# num_epochs = 300

# # Function to parse a volume file
# def load_volumes(filepath):
#     volumes = [[] for _ in range(num_layers)]
#     with open(filepath, "r") as f:
#         for line in f:
#             parts = line.strip().split("Vol:")[-1].split(",")
#             for i, val in enumerate(parts):
#                 try:
#                     volumes[i].append(float(val))
#                 except ValueError:
#                     volumes[i].append(None)
#     return volumes

# # Plotting
# fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# for i, cat in enumerate(categories):
#     path = f"./similarity_logs/{cat}_conv_hull_volume_pca.txt"
#     volumes = load_volumes(path)

#     for j in range(num_layers):
#         axs[i].plot(range(len(volumes[j])), volumes[j], label=layer_labels[j])
    
#     axs[i].set_title(f"Convex Hull Volumes - {cat}")
#     axs[i].set_ylabel("Volume")
#     axs[i].legend()
#     axs[i].grid(True)

# axs[2].set_xlabel("Epoch")

# plt.tight_layout()
# plt.savefig('./similarity_logs/volumes_pca.png')
