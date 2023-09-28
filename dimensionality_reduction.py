from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import non_negative_factorization as NNMF
from sklearn.cluster import k_means as KMeans

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
    weight_mat, components, _ = NNMF(feat_db, n_components=K)

  elif dim_red == 'kmeans':
    components, *_ = KMeans(feat_db, n_clusters=K)  # TODO (rohan): calculate similarity between feat_db and W and return them as imageid-weight pairs
    weight_mat = None
    pass

  else:
    raise NotImplementedError(f'Haven\'t implemented {dim_red} algorithm')

  return weight_mat, components
