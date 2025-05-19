import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def parse_mi_file(filepath):
    epoch_data = defaultdict(list)
    with open(filepath, 'r') as f:
        for line in f:
            match = re.match(r"Epoch (\d+), MI_XZ: \[.*,\s*([\d\.eE+-]+)\], MI_ZY: \[.*,\s*([\d\.eE+-]+)\]", line)
            if match:
                epoch = int(match.group(1))
                last_mi_zy = float(match.group(3))
                epoch_data[epoch].append(last_mi_zy)
    return {epoch: np.mean(values) for epoch, values in sorted(epoch_data.items())}

def plot_mi_multiple(files_and_labels, title="I(Z; Y) Over Epochs"):
    plt.figure(figsize=(10, 6))

    for filepath, label in files_and_labels:
        mi_data = parse_mi_file(filepath)
        epochs = sorted(mi_data.keys())
        mi_values = [mi_data[e] for e in epochs]
        plt.plot(epochs, mi_values, label=label, marker='x')

    plt.xlabel('Epoch')
    plt.ylabel('I(Z; Y)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("./for_KL_plot", exist_ok=True)
    plt.savefig("./for_KL_plot/MI_ZY_plot.png")

if __name__ == "__main__":
    files_and_labels = [
        ("./MI_logs/mutag_orig_full_loss_for_KL.txt", "PGIB"),
        ("./MI_logs/mutag_orig_no_ib_terms_for_KL.txt", "PGIB - IB terms"),
        ("./MI_logs/mutag_orig_no_ib_terms_for_KL_128_32.txt", "PGIB - IB terms + BottleneckMLP"),
    ]

    print("Plotting last layer I(Z; Y) curves...")
    plot_mi_multiple(files_and_labels)
