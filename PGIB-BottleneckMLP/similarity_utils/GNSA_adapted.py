#!/usr/bin/python3
import numpy as np
import torch
import torch.nn as nn
# import ripserplusplus as rpp_py
import tqdm
from tqdm import tqdm
from scipy.spatial import distance_matrix

class NSALoss_adapted(nn.Module):
    def __init__(self, mode='raw',**kwargs):
        super().__init__()
        self.mode = mode
        
    def forward(self, x, z, origin_x, origin_z):
        mean_x = torch.mean(origin_x, dim=0)
        mean_z = torch.mean(origin_z, dim=0)
        if self.mode=='raw':

            normA1 = torch.quantile(torch.sqrt(torch.sum((x - mean_x) ** 2, dim=1)),0.98)
            normA2 = torch.quantile(torch.sqrt(torch.sum((z - mean_z) ** 2, dim=1)),0.98)
    
            A1_pairwise = torch.cdist(x,x)    # compute pairwise dist
            A2_pairwise = torch.cdist(z,z)    # compute pairwise dist
            
            mask = torch.triu(torch.ones_like(A1_pairwise), diagonal=1)
    
            A1_pairwise = A1_pairwise[mask.bool()]
            A2_pairwise = A2_pairwise[mask.bool()]
            A1_pairwise = A1_pairwise/(2*normA1)
            A2_pairwise = A2_pairwise/(2*normA2)
        elif self.mode =='dist':
            A1_pairwise = torch.flatten(x)
            A2_pairwise = torch.flatten(z)            

        loss = torch.mean(torch.square(A2_pairwise - A1_pairwise))
        return loss









import numpy as np

def gram_linear(x):
  """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
  """Compute Gram (kernel) matrix for an RBF kernel.

  Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  dot_products = x.dot(x.T)
  sq_norms = np.diag(dot_products)
  sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
  sq_median_distance = np.median(sq_distances)
  return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
  """Center a symmetric Gram matrix.

  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.

  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

  Returns:
    A symmetric matrix with centered columns and rows.
  """
  #gram = gram.detach().cpu().numpy()
  if not np.allclose(gram, gram.T):
    raise ValueError('Input must be a symmetric matrix.')
  gram = gram.copy()

  if unbiased:
    # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
    # L. (2014). Partial distance correlation with methods for dissimilarities.
    # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
    # stable than the alternative from Song et al. (2007).
    n = gram.shape[0]
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)
  else:
    means = np.mean(gram, 0, dtype=np.float64)
    means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

  return gram


class LNSA_loss(nn.Module):
    def __init__(self, k=5, eps=1e-7, full=False, **kwargs):
        super().__init__()
        self.k = k
        self.eps = eps
        self.full = full
    def compute_neighbor_mask(self, X, normA1):
        x_dist = torch.cdist(X,X)+self.eps
        x_dist = x_dist/normA1
        # HERE
        if X.shape[0] < self.k:
            k = X.shape[0]
        else:
            k = self.k
        values, indices = torch.topk(x_dist, k, largest=False)
        values, indices = values[:,1:], indices[:,1:]
        norm_values=values[:,-1].view(values.shape[0],1)
        lid_X = (1/self.k)*torch.sum(torch.log10(values) - torch.log10(norm_values),axis=1) + self.eps
        return indices, lid_X


    def forward(self, X, Z):
        mean_x = torch.mean(X, dim=0)
        mean_z = torch.mean(Z, dim=0)
        
        normA1 = torch.quantile(torch.sqrt(torch.sum((X - mean_x) ** 2, dim=1)),0.98)
        normA2 = torch.quantile(torch.sqrt(torch.sum((Z - mean_z) ** 2, dim=1)),0.98)

        nn_mask, lid_X = self.compute_neighbor_mask(X, normA1)
        z_dist = torch.cdist(Z,Z)+self.eps
        z_dist = z_dist/normA2
        rows = torch.arange(z_dist.shape[0]).view(-1, 1).expand_as(nn_mask)
        # # # Extract values
        extracted_values = z_dist[rows, nn_mask]
        norm_values=extracted_values[:,-1].view(extracted_values.shape[0],1)
        #print(norm_values)
        lid_Z = (1/self.k)*torch.sum(torch.log10(extracted_values) - torch.log10(norm_values),axis=1) + self.eps
        #lid_nsa = sum(torch.square(torch.exp(lid_X/self.k) - torch.exp(lid_Z/self.k)))/(len(X))
        lid_nsa = sum(torch.square(lid_X - lid_Z))/len(X)
        return lid_nsa#, lid_X, lid_Z