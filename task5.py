#!python3

import config
import helper
import dimensionality_reduction

from similarity_metrics import get_similarity, get_similarity_mat_x_mat

#3rd-party libs
import functools
import torch
import numpy as np

@functools.lru_cache
def get_label_vecs(feat_space: str):
  ret = []
  for label in range(max(config.DATASET.y) + 1):
    # get images feats for the given label
    *_, feat_db_idx, feat_db = config.FEAT_DESC_FUNCS[feat_space]
    label_feats = [x[0] for x in filter(lambda x: x[1][1] == label, feat_db_idx.items())]

    # pool feat_matrix into vector
    # TODO (rohan): Decide a better pooling method
    label_feat = torch.tensor(feat_db[label_feats, :].mean(0))
    ret.append(label_feat)

  return torch.stack(ret, dim=0)

if __name__ == '__main__':
  #inp = helper.get_user_input('feat_space,K,dim_red', None, None)
  inp = {
    #'feat_space': 'color_moment',
    #'feat_space': 'hog',
    #'feat_space': 'resnet_layer3',
    'feat_space': 'resnet_avgpool',
    #'feat_space': 'resnet_fc',
    'K': 5,
    'dim_red': 'svd'
  }
  #feat_db = config.FEAT_DESC_FUNCS[inp['feat_space']][config.FEAT_DB]
  #TODO (rohan): feat_db of label-label similarity matrix
  feat_db = get_label_vecs(inp['feat_space'])
  similarity_metric = config.FEAT_DESC_FUNCS[inp['feat_space']][config.SIMILARITY_METRIC]
  similarity_scores = get_similarity_mat_x_mat(feat_db, feat_db, similarity_metric)
  print(similarity_scores.shape)
  W, H = dimensionality_reduction.reduce_(similarity_scores, inp['K'], inp['dim_red'])
  print(W.shape, H.shape)
  helper.save_pickle(H, config.LATENT_SEMANTICS_FN.format(task='5_label_label_simi_mat', **inp))

  
