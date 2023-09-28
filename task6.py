#!python3

import config
import helper
import dimensionality_reduction

from similarity_metrics import get_similarity, get_similarity_mat_x_mat

#3rd-party libs
import functools
import torch
import numpy as np

if __name__ == '__main__':
  inp = helper.get_user_input('feat_space,K,dim_red')
  '''
  inp = {
    'feat_space': 'color_moment',
    #'feat_space': 'hog',
    #'feat_space': 'resnet_layer3',
    #'feat_space': 'resnet_avgpool',
    #'feat_space': 'resnet_fc',
    'K': 5,
    'dim_red': 'svd'
  }
  '''
  #feat_db = config.FEAT_DESC_FUNCS[inp['feat_space']][config.FEAT_DB]
  #TODO (rohan): feat_db of label-label similarity matrix
  _tmp = config.FEAT_DESC_FUNCS[inp['feat_space']]
  feat_db = _tmp[config.FEAT_DB]
  similarity_metric = _tmp[config.SIMILARITY_METRIC]

  if not isinstance(feat_db, torch.Tensor): feat_db = torch.tensor(feat_db)
  similarity_scores = get_similarity_mat_x_mat(feat_db, feat_db, similarity_metric)
  print(similarity_scores.shape)
  W, H = dimensionality_reduction.reduce_(similarity_scores, inp['K'], inp['dim_red'])
  print(W.shape, H.shape)
  helper.save_pickle(H, config.LATENT_SEMANTICS_FN.format(task='6_img_img_simi_mat', **inp))

  
