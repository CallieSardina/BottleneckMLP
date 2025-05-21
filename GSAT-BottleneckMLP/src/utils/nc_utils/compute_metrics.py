import torch

def compute_nc1(features, labels, num_classes):
    """
    Compute NC1: within-class variability collapse.
    Args:
        features (torch.Tensor): Last-layer features of shape (N, D), where N is the number of samples and D is the feature dimension.
        labels (torch.Tensor): Ground truth labels of shape (N,).
        num_classes (int): Number of classes.
    Returns:
        float: Average within-class variability (NC1 metric).
    """
    within_class_variability = 0
    total_samples = features.size(0)

    for k in range(num_classes):
        class_features = features[labels == k]
        num_features = class_features.size(dim = 0)
        if class_features.size(0) == 0:
            continue
        class_mean = class_features.mean(dim=0)
        variability = (class_features - class_mean).pow(2).sum(dim=1).sum() / num_features
        within_class_variability += variability

    return within_class_variability / num_classes

def compute_nc2(features, labels, num_classes):
    """
    Compute NC2: Convergence to a simplex ETF.
    Args:
        features (torch.Tensor): Last-layer features of shape (N, D).
        labels (torch.Tensor): Ground truth labels of shape (N,).
        num_classes (int): Number of classes.
    Returns:
        tuple: a variety of metrics regarding the penultimate layer activation class means
    """
    global_mean = features.mean(dim=0)
    class_mean_norms = []
    class_centered_means = []

    for k in range(num_classes):
        class_features = features[labels == k]

        class_mean_centered = class_features.mean(dim=0) - global_mean
        class_centered_means.append(class_mean_centered)

        class_mean_centered_norm = torch.linalg.vector_norm(class_mean_centered)
        class_mean_norms.append(class_mean_centered_norm)

    class_mean_norms = torch.tensor(class_mean_norms)
    activation_equinorm = torch.std(class_mean_norms) / torch.mean(class_mean_norms)

    activation_cosines = []
    for i in range(num_classes):
        class1_mean_centered = class_centered_means[i]
        for j in range(i + 1, num_classes):
            class2_mean_centered = class_centered_means[j]

            cosine = torch.dot(class1_mean_centered, class2_mean_centered) / (torch.linalg.norm(class1_mean_centered) * torch.linalg.norm(class2_mean_centered))
            activation_cosines.append(cosine)
    
    activation_cosines = torch.tensor(activation_cosines)
    activation_cosines_mean = torch.mean(activation_cosines)
    activation_cosines_std = torch.std(activation_cosines)

    return activation_equinorm, activation_cosines_mean, activation_cosines_std

def compute_nc2_with_classifiers(features, classifiers, labels, num_classes):
    """
    Compute NC2: Convergence to a simplex ETF.
    Args:
        features (torch.Tensor): Last-layer features of shape (N, D).
        labels (torch.Tensor): Ground truth labels of shape (N,).
        num_classes (int): Number of classes.
    Returns:
        tuple: a variety of metrics regarding the penultimate layer activation class means
    """
    global_mean = features.mean(dim=0)
    class_mean_norms = []

    for k in range(num_classes):
        class_features = features[labels == k]
        if class_features.size(0) == 0:
            continue
        class_mean_centered = class_features.mean(dim=0) - global_mean
        class_mean_centered_norm = torch.linalg.vector_norm(class_mean_centered)
        class_mean_norms.append(class_mean_centered_norm)

    class_mean_norms = torch.tensor(class_mean_norms)
    activation_equinorm = torch.std(class_mean_norms) / torch.mean(class_mean_norms)

    activation_cosines = []
    activation_classifier_cosines = []
    for i in range(num_classes):
        class1_features = features[labels == i]
        class1_mean_centered = class1_features.mean(dim=0) - global_mean
        for j in range(i + 1, num_classes):
            if j > i:
                class2_features = features[labels == j]
                class2_mean_centered = class2_features.mean(dim=0) - global_mean

                cosine = torch.dot(class1_mean_centered, class2_mean_centered) / (torch.linalg.norm(class1_mean_centered) * torch.linalg.norm(class2_mean_centered))
                activation_cosines.append(cosine)
            if j != i:
                class2_classifier = classifiers[j]
                
                cosine = torch.dot(class1_mean_centered, class2_classifier) / (torch.linalg.norm(class1_mean_centered) * torch.linalg.norm(class2_classifier))
                activation_classifier_cosines.append(cosine)
    
    activation_cosines = torch.tensor(activation_cosines)
    activation_classifier_cosines = torch.tensor(activation_classifier_cosines)
    activation_cosines_mean = torch.mean(activation_cosines)
    activation_cosines_std = torch.std(activation_cosines)
    activation_classifier_cosines_mean = torch.mean(activation_classifier_cosines)
    activation_classifier_cosines_std = torch.std(activation_classifier_cosines)

    return activation_equinorm, activation_cosines_mean, activation_cosines_std, activation_classifier_cosines_mean, activation_classifier_cosines_std