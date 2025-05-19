import os
import re
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_epoch_layer_trends(filepath):
    """
    Parses the log file and returns per-epoch NSA and LNSA trends per layer.
    Returns a tuple: (epoch_numbers, nsa_per_layer, lnsa_per_layer)
    """
    pattern = r"Epoch (\d+), NSA\+LNSA: ([\d\., ]+), LNSA: ([\d\., ]+)"
    raw_epoch_data = defaultdict(lambda: defaultdict(list))

    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                values_nsa_lnsa = [float(v.strip()) for v in match.group(2).split(',')]
                values_lnsa = [float(v.strip()) for v in match.group(3).split(',')]
                for layer, (v_total, v_lnsa) in enumerate(zip(values_nsa_lnsa, values_lnsa)):
                    if v_total != 0.0 and v_lnsa != 0.0 and v_total != 'nan' and v_lnsa != 'nan':
                        raw_epoch_data[epoch][layer].append((v_total, v_lnsa))
                    else:
                        raw_epoch_data[epoch][layer].append((None, None))

    all_epochs = sorted(raw_epoch_data.keys())
    layer_set = set()
    for epoch_layers in raw_epoch_data.values():
        layer_set.update(epoch_layers.keys())
    max_layer = max(layer_set)

    nsa_per_layer = {l: [None] * len(all_epochs) for l in range(max_layer + 1)}
    lnsa_per_layer = {l: [None] * len(all_epochs) for l in range(max_layer + 1)}

    for i, epoch in enumerate(all_epochs):
        for layer in range(max_layer + 1):
            if layer in raw_epoch_data[epoch]:
                nsa_vals, lnsa_vals = zip(*raw_epoch_data[epoch][layer])

                nsa_running_total = 0
                len_nsa_vals = 1
                for nsa_val in nsa_vals:
                    if nsa_val == None or nsa_val > 1:
                        continue
                    else:
                        nsa_running_total += nsa_val
                        len_nsa_vals += 1
                nsa_avg = nsa_running_total / len_nsa_vals
                nsa_per_layer[layer][i] = nsa_avg
                # nsa_per_layer[layer][i] = sum(nsa_vals) / len(nsa_vals)
                lnsa_running_total = 0
                len_lnsa_vals = 1
                for lnsa_val in lnsa_vals:
                    if lnsa_val == None or lnsa_val > 1:
                        continue
                    else:
                        lnsa_running_total += lnsa_val
                        len_lnsa_vals += 1
                lnsa_avg = lnsa_running_total / len_lnsa_vals
                lnsa_per_layer[layer][i] = lnsa_avg
                # lnsa_per_layer[layer][i] = sum(lnsa_vals) / len(lnsa_vals)
                # if nsa_per_layer[layer][i] > 1 or lnsa_per_layer[layer][i] > 1:
                #     nsa_per_layer[layer][i] = 0
                #     lnsa_per_layer[layer][i] = 0
                if nsa_per_layer[layer][i] == 0.0:
                    nsa_per_layer[layer][i] = nsa_per_layer[layer][i]
                else:
                    nsa_per_layer[layer][i] = nsa_per_layer[layer][i] - lnsa_per_layer[layer][i]
                if nsa_per_layer[layer][i] < 0: print("ERROR: neg nsa value", nsa_per_layer[layer][i], lnsa_per_layer[layer][i])
            # else:
            #     # Fill missing data with 0 (or interpolate as needed)
            #     nsa_per_layer[layer][i] = 0
            #     lnsa_per_layer[layer][i] = 0

    return all_epochs, nsa_per_layer, lnsa_per_layer


def plot_layer_trends_across_epochs(filepaths, mode="lnsa", save_as="epoch_trends"):
    """
    Plots NSA or LNSA trends over epochs. Each subplot is a layer, each line is a category.
    """
    assert mode in ("nsa", "lnsa")

    # First, parse all files to get all trends and common epochs
    all_data = []
    max_layer = -1
    for filepath in filepaths:
        epochs, nsa_layer_trends, lnsa_layer_trends = parse_epoch_layer_trends(filepath)
        trends = lnsa_layer_trends if mode == "lnsa" else nsa_layer_trends
        all_data.append((epochs, trends))
        max_layer = max(max_layer, max(trends.keys()))

    fig, axes = plt.subplots(1, max_layer + 1, figsize=(6 * (max_layer + 1), 5), sharey=True)

    for layer in range(max_layer + 1):
        ax = axes[layer]
        for i, (epochs, trends) in enumerate(all_data):
            label = f'Category {i + 1}'
            if layer in trends:
                values = trends[layer]
                ax.plot(epochs, values, label=label)

        ax.set_title(f'Layer {layer + 1}')
        ax.set_xlabel("Epoch")
        if layer == 0:
            ax.set_ylabel(f"{'LNSA' if mode == 'lnsa' else 'NSA'}")
        ax.grid(True)
        ax.legend()

    plt.suptitle(f"{'LNSA' if mode == 'lnsa' else 'NSA'} Trends per Category per Layer")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{save_as}_{mode}_trends_by_layer.png")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python script.py <file1.txt> <file2.txt> <file3.txt> <nsa|lnsa> <save_as>")
        sys.exit(1)

    files = sys.argv[1:4]
    mode = sys.argv[4]
    save_as = sys.argv[5]
    plot_layer_trends_across_epochs(files, mode, save_as)
