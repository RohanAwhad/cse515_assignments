from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import non_negative_factorization as NNMF
from sklearn.cluster import k_means as KMeans
from hottbox.core import Tensor, TensorTKD
from hottbox.algorithms.decomposition import CPD
from tqdm import tqdm
import torch
import numpy as np

from similarity_metrics import get_similarity


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
    model = SVD(n_components=K)
    weight_mat = model.fit_transform(feat_db)
    components = model.components_

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