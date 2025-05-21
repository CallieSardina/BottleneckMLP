import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description='Plot Info Plane from MI data')
parser.add_argument('--file_name', type=str)
args = parser.parse_args()
file_name = args.file_name

# Initialize the storage
epoch_mi_xz = defaultdict(list)
epoch_mi_zy = defaultdict(list)

# Read the data from file
file_path = f'{file_name}.txt'
with open(file_path, 'r') as file:
    for line in file:
        match = re.match(r'Epoch (\d+), MI_XZ: ([\d\.]+), MI_ZY: ([\d\.]+)', line)
        if match:
            epoch = int(match.group(1))
            epoch_mi_xz[epoch].append(float(match.group(2)))
            epoch_mi_zy[epoch].append(float(match.group(3)))

# Compute averages per epoch
epochs = sorted(epoch_mi_xz.keys())
avg_mi_xz = [np.mean(epoch_mi_xz[epoch]) for epoch in epochs]
avg_mi_zy = [np.mean(epoch_mi_zy[epoch]) for epoch in epochs]

# Set maximum epoch for colorbar range
COLORBAR_MAX_EPOCHS = max(epochs)
sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=COLORBAR_MAX_EPOCHS))
sm._A = []

# Create the main figure
fig, ax = plt.subplots(figsize=(8, 6))
for epoch, avg_xz, avg_zy in zip(epochs, avg_mi_xz, avg_mi_zy):
    c = sm.to_rgba(epoch)
    ax.scatter(avg_xz, avg_zy, s=30, facecolors=[c], edgecolor='none', alpha=1, zorder=2)

ax.set_xlabel('I(G; G_s)')
ax.set_ylabel('I(G_s; Y)')
ax.set_title('Info Plane Across Epochs -- GSAT')
cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8])
plt.colorbar(sm, label='Epoch', cax=cbaxes)
plt.tight_layout()
plt.savefig(f'plots/infoplane_{file_name}.png', bbox_inches='tight')
plt.show()
