#!python3

import config
import helper

# 3rd-party libs
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import non_negative_factorization as NNMF
from sklearn.clusters import k_means as KMeans


def get_top_k_latent_semantics(feat_space, K, dim_red):
  print(feat_space, K, dim_red)
  feat_db = config.FEAT_DESC_FUNCS[feat_space][config.FEAT_DB]

  if dim_red == 'svd':
    model = SVD(n_components=K)
    out = model.fit_transform(feat_db)
    print(model.components_.shape)
    print(out)
    print(out.shape)

  elif dim_red == 'lda':
    model = LDA(n_components=K)
    out = model.fit_transform(feat_db)
    print(model.components_.shape)
    print(out)
    print(out.shape)

  elif dim_red == 'nnmf':
    W, H, _ = NNMF(feat_db, n_components=K)
    print(W.shape)
    print(H.shape)
    
  elif dim_red == 'kmeans':
    W, _* = KMeans(feat_db, n_clusters=K)  # TODO (rohan): calculate similarity between feat_db and W and return them as imageid-weight pairs
    print(W.shape)
    pass

  else:
    print(f'Haven\'t implemented {dim_red} algorithm')


if __name__ == '__main__':
  #inp = helper.get_user_input('feat_space,K,dim_red', None, None)
  inp = {
    'feat_space': 'resnet_layer3',
    'K': 5,
    'dim_red': 'nnmf'
  }
  get_top_k_latent_semantics(**inp)
