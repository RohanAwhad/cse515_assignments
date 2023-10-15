from config import DEBUG
from similarity_metrics import get_similarity

import numpy as np
import random
import torch

from hottbox.core import Tensor, TensorTKD
from hottbox.algorithms.decomposition import CPD
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import non_negative_factorization as NNMF
from tqdm import tqdm
from typing import Tuple

def svd_custom(A: torch.Tensor, K: int) -> torch.Tensor:
  # TODO (rohan): sometimes the eigen vectors are flipped, need to fix that
  # Only in DEBUG mode calculate U * Sigma
  if DEBUG:
    print('computing eigen values and vectors...')
    eigen_values1, U = map(torch.tensor, np.linalg.eig(A @ A.T))

  eigen_values2, V = map(torch.tensor, np.linalg.eig(A.T @ A))

  # sort eigen values in descending order along with U and V
  if DEBUG:
    print('sorting eigen values and vectors...')
    tmp_1, u_tmp = [], []
    for ev, u in sorted(zip(eigen_values1, U.T), key=lambda x: x[0], reverse=True):
      tmp_1.append(ev)
      u_tmp.append(u)
    eigen_values1, U = torch.tensor(tmp_1), torch.stack(u_tmp).T

  tmp_2, v_tmp = [], []
  for ev, v in sorted(zip(eigen_values2, V.T), key=lambda x: x[0], reverse=True):
    tmp_2.append(ev)
    v_tmp.append(v)
  eigen_values2, V = torch.tensor(tmp_2), torch.stack(v_tmp).T
  eigen_values2, V = eigen_values2[:K], V[:, :K]

  if DEBUG:
    eigen_values1, U = eigen_values1[:K], U[:, :K]
    Sigma = eigen_values1.pow(0.5)
    print("Sigma: ", Sigma)
    print("U    : ", U)
    print("V    : ", V)

    print("Feat DB @ V: ", A @ V)
    print("U * Sigma  : ", U * Sigma)

  return V.T


def nnmf_custom(X, K, max_iter=1000, tol=1e-4):
  # create torch random tensors for W and H
  W = torch.rand((X.shape[0], K))
  H = torch.rand((K, X.shape[1]))

  for i in tqdm(range(max_iter), desc='NNMF', leave=False):
    W = W * (X @ H.T) / (W @ H @ H.T + 1e-10)
    H = H * (W.T @ X) / (W.T @ W @ H + 1e-10)
    if np.linalg.norm(X - W @ H) < tol: break

  return W, H


def kmeans_custom(X, K, max_iter=500):
  # Initialize centroids randomly
  centroids = X[random.sample(range(len(X)), K)]

  for _ in tqdm(range(max_iter), desc='KMeans', leave=False):
    distances = torch.cdist(X, centroids)
    labels = torch.argmin(distances, dim=1)

    # Update centroids by computing the mean of all data points in each cluster
    new_centroids = torch.stack([X[labels == k].mean(dim=0) for k in range(K)])

    # Check for convergence
    if torch.equal(centroids, new_centroids): break
    centroids = new_centroids

  if DEBUG:
    print("KMeans components: ", centroids)
    print("KMeans weight_mat: ", torch.cdist(X, centroids))

  return centroids



def reduce_(feat_db: torch.Tensor, K: int, dim_red: str):

  if dim_red == 'svd':
    components = svd_custom(feat_db, K)
    weight_mat = feat_db @ components.T
    
  elif dim_red == 'lda':
    model = LDA(n_components=K)
    weight_mat = model.fit_transform(feat_db.abs())
    components = model.components_

  elif dim_red == 'nnmf':
    weight_mat, components = nnmf_custom(abs(feat_db), K, max_iter=200)

  elif dim_red == 'kmeans':
    components = kmeans_custom(feat_db, K)
    weight_mat = torch.cdist(feat_db, components)
  
  elif dim_red == 'cp':
    # print("Shape", feat_db.numpy().shape)
    tensor = Tensor(feat_db.numpy())
    cpd = CPD()
    tensor_tkd = cpd.decompose(tensor, rank=(K,))
    factor_matrices = tensor_tkd.fmat
    weight_mat=factor_matrices
    components = None
  else:
    raise NotImplementedError(f'Haven\'t implemented {dim_red} algorithm')

  return weight_mat, components
