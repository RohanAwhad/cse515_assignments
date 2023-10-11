from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import non_negative_factorization as NNMF
from sklearn.cluster import k_means as KMeans
from hottbox.core import Tensor, TensorTKD
from hottbox.algorithms.decomposition import CPD
import numpy

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
    components, *_ = KMeans(feat_db, n_clusters=K)  # TODO (rohan): calculate similarity between feat_db and W and return them as imageid-weight pairs
    weight_mat = None
    raise NotImplementedError('for kmeans haven\'t yet calculated weight matrix')
  
  elif dim_red == 'cp':
    print("Shape", feat_db.numpy().shape)
    tensor = Tensor(feat_db.numpy())
    cpd = CPD()
    tensor_tkd = cpd.decompose(tensor, rank=(K,))
    factor_matrices = tensor_tkd.fmat
    weight_mat=factor_matrices[2]
    components = None
  else:
    raise NotImplementedError(f'Haven\'t implemented {dim_red} algorithm')

  return weight_mat, components
