from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re


def parse_nsa(filepath):
    """
    Parses the log file and returns per-epoch NSA and LNSA trends per layer.
    Returns a tuple: (epoch_numbers, nsa_per_layer, lnsa_per_layer)
    """
    pattern = r"Epoch (\d+), NSA: ([\d\., ]+)"
    raw_epoch_data = defaultdict(lambda: defaultdict(list))

    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                values_nsa = [float(v.strip()) for v in match.group(2).split(',')]
                for layer, v_total in enumerate(values_nsa):
                    if v_total != 0.0 and v_total != 'nan':
                        raw_epoch_data[epoch][layer].append(v_total)
                    else:
                        raw_epoch_data[epoch][layer].append(None)

    all_epochs = sorted(raw_epoch_data.keys())
    layer_set = set()
    for epoch_layers in raw_epoch_data.values():
        layer_set.update(epoch_layers.keys())
    max_layer = max(layer_set)

    nsa_per_layer = {l: [None] * len(all_epochs) for l in range(max_layer + 1)}

    for i, epoch in enumerate(all_epochs):
        for layer in range(max_layer + 1):
            if layer in raw_epoch_data[epoch]:
                nsa_vals = raw_epoch_data[epoch][layer]

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

    return all_epochs, nsa_per_layer

def parse_lnsa(filepath):
    pattern = r"Epoch (\d+), NSA\+LNSA: ([\d\., ]+), LNSA: ([\d\., ]+)"
    raw_epoch_data = defaultdict(lambda: defaultdict(list))

    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                values_lnsa = [float(v.strip()) for v in match.group(3).split(',')]
                for layer, v_lnsa in enumerate(values_lnsa):
                    if v_lnsa != 0.0 and v_lnsa != 'nan':
                        raw_epoch_data[epoch][layer].append(v_lnsa)
                    else:
                        raw_epoch_data[epoch][layer].append(None)

    all_epochs = sorted(raw_epoch_data.keys())
    max_layer = max(k for epoch in raw_epoch_data for k in raw_epoch_data[epoch])

    lnsa_per_layer = {l: [None] * len(all_epochs) for l in range(max_layer + 1)}

    for i, epoch in enumerate(all_epochs):
        for layer in range(max_layer + 1):
            if layer in raw_epoch_data[epoch]:
                lnsa_vals = raw_epoch_data[epoch][layer]
                running_total = 0
                count = 1
                for val in lnsa_vals:
                    if val is not None and val <= 1:
                        running_total += val
                        count += 1
                lnsa_per_layer[layer][i] = running_total / count

    return all_epochs, lnsa_per_layer

def fit_func(t, mu, sigma):
    return mu * t + sigma

def plot_fit_per_category(filepaths, mode='lnsa', layer_idx=0):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    mus = []
    for i, filepath in enumerate(filepaths):
        ax = axes[i]
        label = f"Category {i + 1}"
        
        # Parse
        if mode == 'lnsa':
            epochs, values_per_layer = parse_lnsa(filepath)
        else:
            epochs, values_per_layer = parse_nsa(filepath)

        values = np.array(values_per_layer[layer_idx])
        epochs = np.array(epochs)
        valid_mask = [v is not None for v in values]
        values = np.array([v for v in values if v is not None])
        epochs = np.array([t for j, t in enumerate(epochs) if valid_mask[j]])
        if np.any(np.isnan(values)) or np.any(values is None):
            continue

        # Compute y = |x_t - x_0|
        x0 = values[0]
        y = np.abs(values - x0)
        y = y * 1000

        # Fit
        try:
            params, _ = curve_fit(fit_func, epochs, y)
            mu_fit, sigma_fit = params
            mus.append(mu_fit)
            y_fit = fit_func(epochs, mu_fit, sigma_fit)

            # Plot
            ax.plot(epochs, y, 'o', label='|x_t - x_0|')
            ax.plot(epochs, y_fit, '-', label=f'Fit: μt + σ\nμ={mu_fit:.4f}, σ={sigma_fit:.4f}')
            ax.set_title(label)
            ax.set_xlabel('Epoch')
            if i == 0:
                ax.set_ylabel('|x_t - x_0|')
            ax.grid(True)
            ax.legend()
        except Exception as e:
            ax.set_title(label + f" (fit failed: {e})")

    plt.suptitle(f"Fitting |x_t - x_0| = μt + σ for Layer {layer_idx+1}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{mode}_layer_{layer_idx+1}.png')

    return mus

def plot_mu(mu_lnsa_layers, mu_nsa_layers):
    # Labels and bar settings
    layer_labels = [f"Layer {i}" for i in range(4)]
    category_labels = ["Cat 1", "Cat 2", "Cat 3"]
    x = np.arange(len(layer_labels))  # Layer index
    bar_width = 0.2
    offsets = [-bar_width, 0, bar_width]
    colors = ['skyblue', 'salmon', 'lightgreen']

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # μ LNSA plot
    for i in range(3):  # 3 categories
        vals = [mu_lnsa_layers[layer][i] for layer in range(4)]
        axs[0].bar(x + offsets[i], vals, width=bar_width, label=category_labels[i], color=colors[i])
    axs[0].set_ylabel("μ (LNSA)")
    axs[0].set_title("LNSA: μ per Category Across Layers")
    axs[0].legend(title="Category")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(layer_labels)
    axs[0].grid(True, axis='y')

    # μ NSA plot
    for i in range(3):  # 3 categories
        vals = [mu_nsa_layers[layer][i] for layer in range(4)]
        axs[1].bar(x + offsets[i], vals, width=bar_width, label=category_labels[i], color=colors[i])
    axs[1].set_ylabel("μ (NSA)")
    axs[1].set_title("NSA: μ per Category Across Layers")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(layer_labels)
    axs[1].legend(title="Category")
    axs[1].grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('mu_plot.png')


filepaths = ["cat1_lnsa.txt", "cat2_lnsa.txt", "cat3_lnsa.txt"]  
mus_1_lnsa = plot_fit_per_category(filepaths, mode='lnsa', layer_idx=0)
mus_2_lnsa = plot_fit_per_category(filepaths, mode='lnsa', layer_idx=1)
mus_3_lnsa= plot_fit_per_category(filepaths, mode='lnsa', layer_idx=2)
mus_4_lnsa = plot_fit_per_category(filepaths, mode='lnsa', layer_idx=3)

filepaths = ["cat1_nsa_adapted.txt", "cat2_nsa_adapted.txt", "cat3_nsa_adapted.txt"]  
mus_1_nsa = plot_fit_per_category(filepaths, mode='nsa', layer_idx=0)
mus_2_nsa = plot_fit_per_category(filepaths, mode='nsa', layer_idx=1)
mus_3_nsa = plot_fit_per_category(filepaths, mode='nsa', layer_idx=2)
mus_4_nsa= plot_fit_per_category(filepaths, mode='nsa', layer_idx=3)

mu_lnsa = [mus_1_lnsa, mus_2_lnsa, mus_3_lnsa, mus_4_lnsa]
mu_nsa = [mus_1_nsa, mus_2_nsa, mus_3_nsa, mus_4_nsa]

plot_mu(mu_lnsa, mu_nsa)

