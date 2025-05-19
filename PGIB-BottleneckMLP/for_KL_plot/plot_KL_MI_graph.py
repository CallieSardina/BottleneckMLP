import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def parse_kl_file(filepath):
    epoch_data = defaultdict(list)
    with open(filepath, 'r') as f:
        for line in f:
            match = re.match(r"Epoch (\d+), KL Loss: ([\d\.eE+-]+)", line)
            if match:
                epoch = int(match.group(1))
                kl_loss = float(match.group(2))
                epoch_data[epoch].append(kl_loss)
    return {epoch: np.mean(values) for epoch, values in sorted(epoch_data.items())}

def parse_mi_file(filepath):
    epoch_data = defaultdict(list)
    with open(filepath, 'r') as f:
        for line in f:
            match = re.match(r"Epoch (\d+), MI_XZ: \[.*,\s*([\d\.eE+-]+)\], MI_ZY", line)
            if match:
                epoch = int(match.group(1))
                last_mi_xz = float(match.group(2))
                epoch_data[epoch].append(last_mi_xz)
    return {epoch: np.mean(values) for epoch, values in sorted(epoch_data.items())}

def plot_kl_mi(kl_data, mi_data, title="KL and I(X; Z)over Epochs"):
    epochs = sorted(set(kl_data.keys()).union(mi_data.keys()))
    kl_values = [kl_data.get(e, np.nan) for e in epochs]
    mi_values = [mi_data.get(e, np.nan) for e in epochs]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, kl_values, label='KL Loss', marker='o')
    plt.plot(epochs, mi_values, label='I(X; Z)', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./for_KL_plot/KL_MI_graph.png")

if __name__ == "__main__":
    kl_file = "./for_KL_plot/original_full_loss_mutag_orig_full_loss_for_KL.txt"  
    mi_file = "./MI_logs/mutag_with_MLP_for_KL_128_32_128.txt"      

    print("start")

    kl_data = parse_kl_file(kl_file)
    mi_data = parse_mi_file(mi_file)

    plot_kl_mi(kl_data, mi_data)
