from config import DEBUG
from similarity_metrics import get_similarity

import torch
import numpy as np

from hottbox.core import Tensor, TensorTKD
from hottbox.algorithms.decomposition import CPD
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import non_negative_factorization as NNMF
from tqdm import tqdm
from typing import Tuple

def svd_custom(A: torch.Tensor, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
  # TODO (rohan): sometimes the eigen vectors are flipped, need to fix that
  if DEBUG: print('computing eigen values and vectors...')
  eigen_values1, U = map(torch.tensor, np.linalg.eig(A @ A.T))
  eigen_values2, V = map(torch.tensor, np.linalg.eig(A.T @ A))

  # select only top k eigen values
  eigen_values1, U = eigen_values1[:K], U[:, :K]
  eigen_values2, V = eigen_values2[:K], V[:, :K]

  # sort eigen values in descending order along with U and V
  if DEBUG: print('sorting eigen values and vectors...')
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

  Sigma = eigen_values1.pow(0.5)
  if DEBUG:
    print("Sigma: ", Sigma)
    print("U    : ", U)
    print("V    : ", V)

  return (U * Sigma, V.T)

  

  

class nnmf_custom: 
    def __init__(self, n_components: int) -> None:
        self.components_ = None
        self.n_components = n_components
        self.reconstruction_error_ = None
   
    def fit_transform(self, X, n_components):
        W, H, reconstruction_error = nnmf_custom(X, n_components)
        self.components_ = H.T
        self.reconstruction_error_ = reconstruction_error
        return W

    def transform(self, X):
        return np.dot(X, self.components_)

def nnmf_custom(X, K, max_iter=1000):
    W = np.random.rand(X.shape[0], K)
    H = np.random.rand(K, X.shape[1])

    for i in range(max_iter):
        # Update H
        
        H *= (np.asarray(W.T) @ np.asarray(X)) / (np.asarray(W.T) @ np.asarray(W) @ np.asarray(H) + 1e-10)

        # Update W
        W *= (np.asarray(X) @ np.asarray(H.T)) / (np.asarray(W) @ np.asarray(H) @ np.asarray(H.T) + 1e-10)

    return W, H, None

def recaliberate_centroids(cluster_idx, K, X):
    _, n = np.shape(X)
    centroids = np.empty((K, n))
    for i in range(K):
        points = X[cluster_idx == i] 
        centroids[i] = np.mean(points, axis=0) 
    return centroids

def Kmeans(K, X, max_iterations=500):
    m, n = np.shape(X)
    centroids = np.empty((K, n))
    for i in range(K):
        centroids[i] =  X[np.random.choice(range(m))] 
    for _ in tqdm(range(max_iterations)):
        clusters = np.empty(m)
        for i in range(m):
            distances = np.empty(K)
            for j in range(K):
                distances[j] = np.sqrt(np.sum((centroids[j] - X[i])**2))
            clusters[i] =  np.argmin(distances)
        tmp_centroids = centroids                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        centroids = recaliberate_centroids(clusters, K, X)
        if not (tmp_centroids - centroids).any():
            return centroids, clusters 
    return centroids, clusters

def reduce_(feat_db, K, dim_red):

  if dim_red == 'svd':
    '''
    model = SVD(n_components=K)
    weight_mat = model.fit_transform(feat_db)
    components = model.components_
    '''
    #using custom SVD
    weight_mat, components = svd_custom(feat_db, K)
    
  elif dim_red == 'nnmf_custom':
    W, H, _ = nnmf_custom(abs(feat_db), K)
    weight_mat, components = W, H
    
  elif dim_red == 'lda':
    model = LDA(n_components=K)
    weight_mat = model.fit_transform(feat_db)
    components = model.components_

  elif dim_red == 'nnmf':
    weight_mat, components, _ = NNMF(abs(feat_db), n_components=K)

  elif dim_red == 'kmeans':
    components, *_ = Kmeans(K, feat_db.numpy())
    components = torch.tensor(components)
    #weight_mat = get_similarity(components, feat_db, "euclidean_distance")
    raise NotImplementedError('for kmeans haven\'t yet calculated weight matrix')
  
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
