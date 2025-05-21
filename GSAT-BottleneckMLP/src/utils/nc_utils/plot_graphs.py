import matplotlib.pyplot as plt
import numpy as np
import os, torch
from compute_metrics import compute_nc1, compute_nc2
import pickle as pkl
import math

CLASSES = 2
METRIC_COMPUTATION_FREQUENCY = 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def process_results(FILE_PATH):

    with open(FILE_PATH, "rb") as f:
        feature_data = torch.load(f, map_location=device)

    num_classes = CLASSES
    within_class_variabilities = []
    activation_equinorms = []
    activation_cosines_means = []
    activation_cosines_stds = []
    test_accs = []
    train_accs = []

    for features, labels, test_clf_acc, train_clf_acc in feature_data: 
        labels = labels.squeeze() 
        features_max, _ = torch.max(features, dim=0)
        features_min, _ = torch.min(features, dim=0)
        
        features.sub_(features_min)
        features.div_(features_max - features_min)

        mask = ~torch.isnan(features).any(0)
        features = features[:, mask]

        within_class_variability = compute_nc1(features, labels, num_classes=num_classes)
        activation_equinorm, activation_cosines_mean, activation_cosines_std = compute_nc2(features, labels, num_classes=num_classes)

        within_class_variabilities.append(within_class_variability.item())
        activation_equinorms.append(activation_equinorm.item())
        activation_cosines_means.append(activation_cosines_mean.item())
        activation_cosines_stds.append(activation_cosines_std.item())

        test_accs.append(test_clf_acc * 100)
        train_accs.append(train_clf_acc * 100)

    print('features shape', features.shape)
    print('labels shape', labels.shape)

    metrics_tuple = (
        within_class_variabilities,
        activation_equinorms,
        activation_cosines_means,
        activation_cosines_stds,
        test_accs,
        train_accs
    )
    
    return metrics_tuple


def plot_with_interpolation(axis, x, y, label, color, marker, markersize, linewidth):
    axis.plot(x, y, label=label, color=color, marker=marker, markersize=markersize, linewidth=linewidth)

def create_graphs(file_path, file_name):
    if not os.path.exists("graphs"):
        os.makedirs("graphs")
    
    within_class_variability, activation_equinorm, activation_cosine_mean, activation_cosine_std, test_acc, train_acc= process_results(file_path)
    
    # print('BUNLAR NAN')
    # print(activation_equinorm)
    # print(activation_cosine_mean)
    # print(activation_cosine_std)

    #print(within_class_variability)
    epochs = 300
    x_axis = list(range(epochs))


    plt.style.use('default')  
    # plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 12

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 12))
    plt.xlabel("Epochs", fontsize=12)


    # Plot train and test accuracy
    plot_with_interpolation(axes[0], x_axis, train_acc, "Train Accuracy", "blue", "o", 3, 2)
    plot_with_interpolation(axes[0], x_axis, test_acc, "Test Accuracy", "red", "o", 3, 2)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_ylim((0,100))
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5, linewidth=0.5)

    # # Plot within-class variability and activation cosine mean
    # print(x_axis)
    # print(within_class_variability)
    plot_with_interpolation(axes[1], x_axis, within_class_variability, "Within-Class Variability", "green", "o", 3, 2)
    plot_with_interpolation(axes[1], x_axis, activation_equinorm, "Activation Equinorm", "orange", "o", 3, 2)

    axes[1].set_ylabel("Metric Value", fontsize=12)
    axes[1].set_ylim(0,1)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5, linewidth=0.5)
    
    # # Plot activation equinorm and activation cosine standard deviation
    # plot_with_interpolation(axes[2], x_axis, activation_cosine_mean, "Activation Cosine Mean", "purple", "o", 3, 2)
    # plot_with_interpolation(axes[2], x_axis, activation_cosine_std, "Activation Cosine Std", "brown", "o", 3, 2)

    # axes[2].set_ylabel("Metric Value", fontsize=12)
    # axes[2].set_ylim(-0.15,0.8)
    # axes[2].legend(fontsize=10)
    # axes[2].grid(True, linestyle='--', alpha=0.5, linewidth=0.5)

    fig.suptitle(f" trained on  with ")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0) 
    plt.savefig(os.path.join("graphs", f"{file_name}.png"), dpi=300)
    plt.close()

    return within_class_variability
    # TODO: Parametrize plots

if __name__ == "__main__":
    total = 0
    directory = '../../new_nc/'

    startFileName = 'nc-normal-mutag'
    res = []
    for file in os.listdir(directory):
        if file.startswith(startFileName):
            file_path = os.path.join(directory, file)
            within_class_variability = create_graphs(file_path, file_name = file)
            res.append(sum(within_class_variability)/len(within_class_variability))
    mean = np.mean(res)
    std_dev = np.std(res, ddof=1)
    print(f'{startFileName}, avg within class var: {mean:.3f} +- {std_dev:.3f}')


