import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_kl_file(filepath):
    epoch_losses = defaultdict(list)
    
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(r"Epoch\s*(\d+),\s*KL Loss:\s*([\d.]+)", line, re.IGNORECASE)
            if match:
                epoch = int(match.group(1))
                kl_loss = float(match.group(2))
                epoch_losses[epoch].append(kl_loss)
    
    # Average KL loss per epoch
    avg_losses = {epoch: sum(losses) / len(losses) for epoch, losses in epoch_losses.items()}
    return avg_losses

def plot_kl_curves(directory):
    plt.figure(figsize=(10, 6))

    label_map = {
        "original_full_loss_mutag_orig_full_loss_for_KL.txt": "PGIB (baseline)",
        "original_NO_IB_loss_mutag_orig_no_ib_terms_for_KL_2.txt": "PGIB - IB Loss",
        "mutag_orig_no_ib_terms_for_KL_128_32.txt": "PGIB - IB loss + BottleneckMLP",
        # "with_MLP_mutag_GIN_with_fcs_for_KL_run2.txt": "PGIB - IB loss + GIN+MLP"
    }

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            avg_losses = parse_kl_file(filepath)
            epochs = sorted(avg_losses.keys())
            losses = [avg_losses[e] for e in epochs]
            label = label_map.get(filename, filename)
            plt.plot(epochs, losses, label=label)

    plt.xlabel("Epoch")
    plt.ylabel("KL Loss")
    plt.title("KL Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./for_KL_plot/KL_plot.png')

# Run the plotting
plot_kl_curves('./for_KL_plot')
