#!python3

import config
import helper
import dimensionality_reduction

from similarity_metrics import get_similarity, get_similarity_mat_x_mat
from task3 import print_img_id_weight_pairs

import functools
import os


@functools.lru_cache
def get_img_img_similarity_matrix(feat_space):
  # check if already computed
  save_fn = config.SIMI_MAT_FN.format(
    feat_space=feat_space,
    mat1='img',
    mat2='img'
  )
  if os.path.exists(save_fn): return helper.load_pickle(save_fn)
  
  _tmp = config.FEAT_DESC_FUNCS[feat_space]
  feat_db = _tmp[config.FEAT_DB]
  similarity_metric = _tmp[config.SIMILARITY_METRIC]
  similarity_scores = get_similarity_mat_x_mat(feat_db, feat_db, similarity_metric)
  helper.save_pickle(similarity_scores, save_fn)
  return similarity_scores


def main():
  inp = helper.get_user_input('feat_space,K,dim_red')

  # get img-img similarity matrix
  similarity_scores = get_img_img_similarity_matrix(inp['feat_space'])

  # reduce dimension
  W, H = dimensionality_reduction.reduce_(similarity_scores, inp['K'], inp['dim_red'])
  helper.save_pickle(H, config.LATENT_SEMANTICS_FN.format(task='6_img_img_simi_mat', **inp))
  print_img_id_weight_pairs(W, config.FEAT_DESC_FUNCS[inp['feat_space']][config.IDX])

if __name__ == '__main__':
  while True: main()
