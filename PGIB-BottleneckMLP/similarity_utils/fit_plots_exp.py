
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
from math import log


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

def quadratic_func(t, a, b, c):
    return a * t**2 + b * t + c

def cubic_func(t, a, b, c, d):
    return a * t**3 + b * t**2 + c * t + d

def quartic_func(t, a, b, c, d, e):
    return a * t**4 + b * t**3 + c * t**2 + d * t + e

def quintic_func(t, a, b, c, d, e, f):
    return a * t**5 + b * t**4 + c * t**3 + d * t**2 + e * t + f

def fit_models(epochs, values):
    # Fit models: linear, quadratic, cubic
    params_lin, _ = curve_fit(fit_func, epochs, values)
    params_quad, _ = curve_fit(quadratic_func, epochs, values)
    params_cubic, _ = curve_fit(cubic_func, epochs, values)
    params_quartic, _ = curve_fit(quartic_func, epochs, values)
    params_quintic, _ = curve_fit(quintic_func, epochs, values)

    y_lin = fit_func(epochs, *params_lin)
    y_quad = quadratic_func(epochs, *params_quad)
    y_cubic = cubic_func(epochs, *params_cubic)
    y_quartic = quartic_func(epochs, *params_quartic)
    y_quintic = quintic_func(epochs, *params_quintic)

    # Calculate R^2 for each model
    def r_squared(y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true))**2)
        ss_residual = np.sum((y_true - y_pred)**2)
        return 1 - (ss_residual / ss_total)

    r2_lin = r_squared(values, y_lin)
    r2_quad = r_squared(values, y_quad)
    r2_cubic = r_squared(values, y_cubic)
    r2_quartic = r_squared(values, y_quartic)
    r2_quintic = r_squared(values, y_quintic)

    return (params_lin, r2_lin), (params_quad, r2_quad), (params_cubic, r2_cubic), (params_quartic, r2_quartic), (params_quintic, r2_quintic)

def plot_fit_per_category(filepaths, mode='lnsa', layer_idx=0):
    fig, axes = plt.subplots(5, 3, figsize=(30, 12), sharey=True)

    fit_labels = ["Linear", "Quadratic", "Cubic", "Quartic", "Quintic"]
    mus = []

    for col, filepath in enumerate(filepaths):
        label = f"Category {col + 1}"

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
        y = np.abs(values - x0) #* 1000

        # Fit models
        (params_lin, r2_lin), (params_quad, r2_quad), (params_cubic, r2_cubic), (params_quartic, r2_quartic), (params_quintic, r2_quintic) = fit_models(epochs, y)
        mus.append(params_lin[0])  # save mu from linear fit

        # Define fit functions
        lin_func = lambda t: params_lin[0] * t + params_lin[1]
        quad_func = lambda t: params_quad[0] * t**2 + params_quad[1] * t + params_quad[2]
        cubic_func = lambda t: params_cubic[0] * t**3 + params_cubic[1] * t**2 + params_cubic[2] * t + params_cubic[3]
        quartic_func = lambda t: params_quartic[0] * t**4 + params_quartic[1] * t**3 + params_quartic[2] * t**2 + params_quartic[3] * t + params_quartic[4]
        quintic_func = lambda t: params_quintic[0] * t**5 + params_quintic[1] * t**4 + params_quintic[2] * t**3 + params_quintic[3] * t**2 + params_quintic[4] * t + params_quintic[5]

        # Plot all 3 fits in 3 rows
        fits = [
            (lin_func, "Linear", r2_lin),
            (quad_func, "Quadratic", r2_quad),
            (cubic_func, "Cubic", r2_cubic),
            (quartic_func, "Quartic", r2_quartic),
            (quintic_func, "Quintic", r2_quintic)
        ]

        for row, (fit_fn, fit_label, r2) in enumerate(fits):
            ax = axes[row][col]
            ax.plot(epochs, y, 'o', label='|x_t - x_0|')
            ax.plot(epochs, fit_fn(epochs), '-', label=f'{fit_label} fit\n$R^2$={r2:.3f}')
            if row == 0:
                ax.set_title(label)
            if col == 0:
                ax.set_ylabel(f"{fit_label} Fit\n|x_t - x_0|")
            ax.set_xlabel('Epoch')
            ax.grid(True)
            ax.legend()

    plt.suptitle(f"Fit Types per Category for Layer {layer_idx+1}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{mode}_layer_{layer_idx+1}_fits_all.png')
    plt.show()

    return mus

def plot_mu(mu_lnsa_layers, mu_nsa_layers):
    # Labels and bar settings
    layer_labels = [f"Layer {i+1}" for i in range(4)]
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
    plt.savefig('mu_plot_exp.png')

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def poly8_func(t, a, b, c, d, e, f, g, h, i):
    return a*t**8 + b*t**7 + c*t**6 + d*t**5 + e*t**4 + f*t**3 + g*t**2 + h*t + i

def poly3_func(t, a, b, c, d, e):
    return e*t**4 + a*t**3 + b*t**2 + c*t + d

def bin_data(x, y, num_bins):
    bins = np.linspace(min(x), max(x), num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    binned_means = []

    for i in range(num_bins):
        mask = (x >= bins[i]) & (x < bins[i+1])
        if np.any(mask):
            binned_means.append(np.mean(y[mask]))
        else:
            binned_means.append(np.nan)

    return bin_centers, np.array(binned_means)

def plot_fit_degree8_with_binning(filepaths, mode='lnsa', layer_idx=0, num_bins=30):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    poly_func = poly8_func
    #poly_func = poly3_func

    mus = []
    for col, filepath in enumerate(filepaths):
        label = f"Category {col + 1}"
        ax = axes[col]

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
        if len(values) == 0:
            continue

        # Compute y = |x_t - x_0|
        x0 = values[0]
        y = np.abs(values - x0)

        # Bin the data
        bin_centers, binned_y = bin_data(np.array(epochs), y, num_bins)
        valid_bin_mask = ~np.isnan(binned_y)
        bin_centers = bin_centers[valid_bin_mask]
        binned_y = binned_y[valid_bin_mask]

        # Fit degree-8 polynomial
        try:
            params, _ = curve_fit(poly_func, bin_centers, binned_y)
            y_fit = poly_func(bin_centers, *params)
            mus.append(params[0])
        except Exception as e:
            print(f"Fit failed for {label}: {e}")
            continue

        # R² calculation
        ss_total = np.sum((binned_y - np.mean(binned_y)) ** 2)
        ss_residual = np.sum((binned_y - y_fit) ** 2)
        r2 = 1 - (ss_residual / ss_total)

        # Plot
        ax.plot(bin_centers, binned_y, 'o', label='Binned |x_t - x_0|')
        ax.plot(bin_centers, y_fit, '-', label=f'Degree-8 Fit\n$R^2$={r2:.3f}')
        ax.set_title(label)
        ax.set_xlabel('Epoch')
        if col == 0:
            ax.set_ylabel('|x_t - x_0|')
        ax.grid(True)
        ax.legend()

    plt.suptitle(f"Degree-8 Polynomial Fit with Binning ({num_bins} bins) - Layer {layer_idx+1}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{mode}_layer_{layer_idx+1}_deg8_binned{num_bins}.png')
    plt.show()

    return mus

filepaths_lnsa = ["cat1_lnsa.txt", "cat2_lnsa.txt", "cat3_lnsa.txt"] 

mus_1_lnsa = plot_fit_degree8_with_binning(filepaths_lnsa, mode='lnsa', layer_idx=0, num_bins=300)
mus_2_lnsa = plot_fit_degree8_with_binning(filepaths_lnsa, mode='lnsa', layer_idx=1, num_bins=300)
mus_3_lnsa = plot_fit_degree8_with_binning(filepaths_lnsa, mode='lnsa', layer_idx=2, num_bins=300)
mus_4_lnsa = plot_fit_degree8_with_binning(filepaths_lnsa, mode='lnsa', layer_idx=3, num_bins=300)
# mus_1_lnsa = plot_fit_per_category(filepaths_lnsa, mode='lnsa', layer_idx=0)
# mus_2_lnsa = plot_fit_per_category(filepaths_lnsa, mode='lnsa', layer_idx=1)
# mus_3_lnsa= plot_fit_per_category(filepaths_lnsa, mode='lnsa', layer_idx=2)
# mus_4_lnsa = plot_fit_per_category(filepaths_lnsa, mode='lnsa', layer_idx=3)

filepaths_nsa = ["cat1_nsa_adapted.txt", "cat2_nsa_adapted.txt", "cat3_nsa_adapted.txt"]  

mus_1_nsa = plot_fit_degree8_with_binning(filepaths_nsa, mode='nsa', layer_idx=0, num_bins=300)
mus_2_nsa = plot_fit_degree8_with_binning(filepaths_nsa, mode='nsa', layer_idx=1, num_bins=300)
mus_3_nsa = plot_fit_degree8_with_binning(filepaths_nsa, mode='nsa', layer_idx=2, num_bins=300)
mus_4_nsa = plot_fit_degree8_with_binning(filepaths_nsa, mode='nsa', layer_idx=3, num_bins=300)
# mus_1_nsa = plot_fit_per_category(filepaths_nsa, mode='nsa', layer_idx=0)
# mus_2_nsa = plot_fit_per_category(filepaths_nsa, mode='nsa', layer_idx=1)
# mus_3_nsa = plot_fit_per_category(filepaths_nsa, mode='nsa', layer_idx=2)
# mus_4_nsa = plot_fit_per_category(filepaths_nsa, mode='nsa', layer_idx=3)

mu_lnsa = [mus_1_lnsa, mus_2_lnsa, mus_3_lnsa, mus_4_lnsa]
mu_nsa = [mus_1_nsa, mus_2_nsa, mus_3_nsa, mus_4_nsa]

plot_mu(mu_lnsa, mu_nsa)
