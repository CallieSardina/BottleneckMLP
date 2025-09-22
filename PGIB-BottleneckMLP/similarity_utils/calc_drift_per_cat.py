import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def parse_log(file_path):
    # pattern to extract epoch and drift values
    pattern = re.compile(
        r"\[Epoch (\d+)\] Avg Drift - Cat1: ([\d\.nan]+), Cat2: ([\d\.nan]+), Cat3: ([\d\.nan]+)"
    )
    # store drifts per epoch per category
    epoch_drifts = defaultdict(lambda: {'Cat1': [], 'Cat2': [], 'Cat3': []})

    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                cat1 = match.group(2)
                cat2 = match.group(3)
                cat3 = match.group(4)
                
                # convert 'nan' to np.nan float type
                cat1 = float(cat1) if cat1.lower() != 'nan' else np.nan
                cat2 = float(cat2) if cat2.lower() != 'nan' else np.nan
                cat3 = float(cat3) if cat3.lower() != 'nan' else np.nan

                epoch_drifts[epoch]['Cat1'].append(cat1)
                epoch_drifts[epoch]['Cat2'].append(cat2)
                epoch_drifts[epoch]['Cat3'].append(cat3)
    
    return epoch_drifts

def compute_average_drifts(epoch_drifts):
    avg_drifts = {'Cat1': [], 'Cat2': [], 'Cat3': []}
    epochs = sorted(epoch_drifts.keys())

    for epoch in epochs:
        for cat in ['Cat1', 'Cat2', 'Cat3']:
            values = np.array(epoch_drifts[epoch][cat])
            # Ignore nan values in average
            avg_val = np.nanmean(values) if len(values) > 0 else np.nan
            avg_drifts[cat].append(avg_val)
    
    return epochs, avg_drifts

def plot_avg_drifts(epochs, avg_drifts):
    plt.figure(figsize=(10,6))
    plt.plot(epochs, avg_drifts['Cat1'], label='Category 1')
    plt.plot(epochs, avg_drifts['Cat2'], label='Category 2')
    plt.plot(epochs, avg_drifts['Cat3'], label='Category 3')
    #plt.ylim(4, 14)
    plt.xlabel('Epoch')
    plt.ylabel('Average Drift')
    plt.title('Average Node Drift per Category over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig("drift_per_cat_out_nci1_total_original_no_info_run1.png")

if __name__ == "__main__":
    log_file = "../out_nci1_total_original_no_info_run1.out" 
    epoch_drifts = parse_log(log_file)
    epochs, avg_drifts = compute_average_drifts(epoch_drifts)
    plot_avg_drifts(epochs, avg_drifts)
    # print("avg drift cat1: ", np.mean(avg_drifts['Cat1'][:149]))
    # print("avg drift cat2: ", np.mean(avg_drifts['Cat2'][:149]))
    # print("avg drift cat3: ", np.mean(avg_drifts['Cat3'][:149]))
    print("avg drift cat1: ", np.mean(avg_drifts['Cat1']))
    print("avg drift cat2: ", np.mean(avg_drifts['Cat2']))
    print("avg drift cat3: ", np.mean(avg_drifts['Cat3']))
