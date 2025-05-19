#!/usr/bin/python3
import numpy as np
import torch
import torch.nn as nn
# import ripserplusplus as rpp_py
import tqdm
from tqdm import tqdm
from scipy.spatial import distance_matrix

# from gph.python import ripser_parallel

def lp_loss(a, b, p=2):
    return (torch.sum(torch.abs(a-b)**p))

def get_indicies(DX, rc, dim, card):
    dgm = rc['dgms'][dim]
    pairs = rc['pairs'][dim]

    # Retrieve vertices v_a and v_b by picking the ones achieving the maximal
    # distance among all pairwise distances between the simplex vertices
    indices, pers = [], []
    for i in range(len(pairs)):
        s1, s2 = pairs[i]
        if len(s1) == dim+1 and len(s2) > 0:
            l1, l2 = np.array(s1), np.array(s2)
            i1 = [s1[v] for v in np.unravel_index(np.argmax(DX[l1,:][:,l1]),[len(s1), len(s1)])]
            i2 = [s2[v] for v in np.unravel_index(np.argmax(DX[l2,:][:,l2]),[len(s2), len(s2)])]
            indices += i1
            indices += i2
            pers.append(dgm[i][1] - dgm[i][0])
    
    # Sort points with distance-to-diagonal
    perm = np.argsort(pers)
    indices = list(np.reshape(indices, [-1,4])[perm][::-1,:].flatten())
    
    # Output indices
    indices = indices[:4*card] + [0 for _ in range(0,max(0,4*card-len(indices)))]
    return list(np.array(indices, dtype=np.compat.long))

def Rips(DX, dim, card, n_threads, engine):
    # Parameters: DX (distance matrix), 
    #             dim (homological dimension), 
    #             card (number of persistence diagram points, sorted by distance-to-diagonal)
    if dim < 1:
        dim = 1
        
    if engine == 'ripser':
        DX_ = DX.numpy()
        DX_ = (DX_ + DX_.T) / 2.0 # make it symmetrical
        DX_ -= np.diag(np.diag(DX_))
    #     rc = rpp_py.run("--format distance --dim " + str(dim), DX_)
    # elif engine == 'giotto':
    #     rc = ripser_parallel(DX, maxdim=dim, metric="precomputed", collapse_edges=False, n_threads=n_threads)
    
    all_indicies = [] # for every dimension
    for d in range(1, dim+1):
        all_indicies.append(get_indicies(DX, rc, d, card))
    return all_indicies

class RTD_differentiable(nn.Module):
    def __init__(self, dim=1, card=50, mode='minimum', n_threads=25, engine='giotto'):
        super().__init__()
            
        if dim < 1:
            raise ValueError(f"Dimension should be greater than 1. Provided dimension: {dim}")
        self.dim = dim
        self.mode = mode
        self.card = card
        self.n_threads = n_threads
        self.engine = engine
        
    def forward(self, Dr1, Dr2, immovable=None):
        # inputs are distance matricies
        d, c = self.dim, self.card
        
        if Dr1.shape[0] != Dr2.shape[0]:
            raise ValueError(f"Point clouds must have same size. Size Dr1: {Dr1.shape} and size Dr2: {Dr2.shape}")
            
        if Dr1.device != Dr2.device:
            raise ValueError(f"Point clouds must be on the same devices. Device Dr1: {Dr1.device} and device Dr2: {Dr2.device}")
            
        device = Dr1.device
        # Compute distance matrices
#         Dr1 = torch.cdist(r1, r1)
#         Dr2 = torch.cdist(r2, r2)

        Dzz = torch.zeros((len(Dr1), len(Dr1)), device=device)
        if self.mode == 'minimum':
            Dr12 = torch.minimum(Dr1, Dr2)
            DX = torch.cat((torch.cat((Dzz, Dr1.T), 1), torch.cat((Dr1, Dr12), 1)), 0)
            if immovable == 2:
                DX_2 = torch.cat((torch.cat((Dzz, Dr1.T), 1), torch.cat((Dr1, Dr1), 1)), 0)   # Transfer gradient for edge minimization to edges in cloud #1
            elif immovable == 1:
                DX_2 = torch.cat((torch.cat((Dzz, Dr1.T), 1), torch.cat((Dr1, Dr2), 1)), 0)   # Transfer gradient from edge minimization to edges in cloud #2
            else:
                DX_2 = DX
        else:
            Dr12 = torch.maximum(Dr1, Dr2)
            DX = torch.cat((torch.cat((Dzz, Dr12.T), 1), torch.cat((Dr12, Dr2), 1)), 0)
            if immovable == 2:
                DX_2 = torch.cat((torch.cat((Dzz, Dr1.T), 1), torch.cat((Dr1, Dr2), 1)), 0)   # Transfer gradient for edge minimization to edges in cloud #1
            elif immovable == 1:
                DX_2 = torch.cat((torch.cat((Dzz, Dr2.T), 1), torch.cat((Dr2, Dr2), 1)), 0)   # Transfer gradient from edge minimization to edges in cloud #2
            else:
                DX_2 = DX
        
        # Compute vertices associated to positive and negative simplices 
        # Don't compute gradient for this operation
        all_ids = Rips(DX.detach().cpu(), self.dim, self.card, self.n_threads, self.engine)
        all_dgms = []
        for ids in all_ids:
            # Get persistence diagram by simply picking the corresponding entries in the distance matrix
            tmp_idx = np.reshape(ids, [2*c,2])
            if self.mode == 'minimum':
                dgm = torch.hstack([torch.reshape(DX[tmp_idx[::2, 0], tmp_idx[::2, 1]], [c,1]), torch.reshape(DX_2[tmp_idx[1::2, 0], tmp_idx[1::2, 1]], [c,1])])
            else:
                dgm = torch.hstack([torch.reshape(DX_2[tmp_idx[::2, 0], tmp_idx[::2, 1]], [c,1]), torch.reshape(DX[tmp_idx[1::2, 0], tmp_idx[1::2, 1]], [c,1])])
            all_dgms.append(dgm)
        return all_dgms
    
class RTDLoss(nn.Module):
    def __init__(self, dim=1, card=50, n_threads=25, engine='giotto', mode='minimum', is_sym=True, lp=1.0, **kwargs):
        super().__init__()

        self.is_sym = is_sym
        self.mode = mode
        self.p = lp
        self.rtd = RTD_differentiable(dim, card, mode, n_threads, engine)
    
    def forward(self, x_dist, z_dist):
        # x_dist is the precomputed distance matrix
        # z is the batch of latent representations
        loss = 0.0
        loss_xz = 0.0
        loss_zx = 0.0
        rtd_xz = self.rtd(x_dist, z_dist, immovable=1)
        if self.is_sym:
            rtd_zx = self.rtd(z_dist, x_dist, immovable=2)
        for d, rtd in enumerate(rtd_xz): # different dimensions
            loss_xz += lp_loss(rtd_xz[d][:, 1], rtd_xz[d][:, 0], p=self.p)
            if self.is_sym:
                loss_zx += lp_loss(rtd_zx[d][:, 1], rtd_zx[d][:, 0], p=self.p)
        loss = (loss_xz + loss_zx) / 2.0
        return loss_xz, loss_zx, loss

class NSALoss(nn.Module):
    def __init__(self, mode='raw',**kwargs):
        super().__init__()
        self.mode = mode
        
    def forward(self, x, z):
        mean_x = torch.mean(x, dim=0)
        mean_z = torch.mean(z, dim=0)
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

def rtd(X,Y, batch_size=400,n_runs = 10):
    loss = RTDLoss(dim=1, engine='ripser')
    X = X.reshape(len(X), -1)
    Y = Y.reshape(len(Y), -1)
    print(X.shape)
    print(Y.shape)
    if batch_size > len(X):
        n_runs=1
    max_dim = 1
    results = []
    
    for i in tqdm(range(n_runs)):
        ids = np.random.choice(np.arange(0, len(X)), size=min(batch_size, len(X)), replace=False)
        
        x = X[ids]
        x_distances = distance_matrix(x, x)
        norm1 = np.percentile(x_distances.flatten(), 90)
        x_distances = x_distances/np.percentile(x_distances.flatten(), 90)
        
        z = Y[ids]
        z_distances = distance_matrix(z, z)
        norm1 = np.percentile(x_distances.flatten(), 90)
        z_distances = z_distances/np.percentile(z_distances.flatten(), 90)
        with torch.no_grad():
            _, _, value = loss(torch.tensor(x_distances), torch.tensor(z_distances))
        results.append(value.item())
    return np.mean(results)

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


def cka(gram_x, gram_y, debiased=False):
  """Compute CKA.

  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

  Returns:
    The value of CKA between X and Y.
  """
  gram_x = center_gram(gram_x, unbiased=debiased)
  gram_y = center_gram(gram_y, unbiased=debiased)

  # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
  # n*(n-3) (unbiased variant), but this cancels for CKA.
  scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

  normalization_x = np.linalg.norm(gram_x)
  normalization_y = np.linalg.norm(gram_y)
  return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
    xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
    n):
  """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
  # This formula can be derived by manipulating the unbiased estimator from
  # Song et al. (2007).
  return (
      xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
      + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


def feature_space_linear_cka(features_x, features_y, debiased=False):
  """Compute CKA with a linear kernel, in feature space.

  This is typically faster than computing the Gram matrix when there are fewer
  features than examples.

  Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.
    debiased: Use unbiased estimator of dot product similarity. CKA may still be
      biased. Note that this estimator may be negative.

  Returns:
    The value of CKA between X and Y.
  """
  features_x = features_x - np.mean(features_x, 0, keepdims=True)
  features_y = features_y - np.mean(features_y, 0, keepdims=True)

  dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
  normalization_x = np.linalg.norm(features_x.T.dot(features_x))
  normalization_y = np.linalg.norm(features_y.T.dot(features_y))

  if debiased:
    n = features_x.shape[0]
    # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
    sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
    sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
    squared_norm_x = np.sum(sum_squared_rows_x)
    squared_norm_y = np.sum(sum_squared_rows_y)

    dot_product_similarity = _debiased_dot_product_similarity_helper(
        dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
        squared_norm_x, squared_norm_y, n)
    normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
        squared_norm_x, squared_norm_x, n))
    normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
        squared_norm_y, squared_norm_y, n))

  return dot_product_similarity / (normalization_x * normalization_y)

class LNSA_loss(nn.Module):
    def __init__(self, k=5, eps=1e-7, full=False, **kwargs):
        super().__init__()
        self.k = k
        self.eps = eps
        self.full = full
    def compute_neighbor_mask(self, X, normA1):
        x_dist = torch.cdist(X,X)+self.eps
        x_dist = x_dist/normA1
        values, indices = torch.topk(x_dist, self.k+1, largest=False)
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