#!python3

import config
import helper
import dimensionality_reduction

from similarity_metrics import get_similarity, get_similarity_mat_x_mat
from task4 import print_label_weight_pairs

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


def main():
    inp = helper.get_user_input('feat_space,K,dim_red', None, None)
    feat_db = get_label_vecs(inp['feat_space'])
    similarity_metric = config.FEAT_DESC_FUNCS[inp['feat_space']][config.SIMILARITY_METRIC]
    similarity_scores = get_similarity_mat_x_mat(feat_db, feat_db, similarity_metric)
    W, H = dimensionality_reduction.reduce_(similarity_scores, inp['K'], inp['dim_red'])
    helper.save_pickle(H, config.LATENT_SEMANTICS_FN.format(task='5_label_label_simi_mat', **inp))
    save_fn = config.SIMI_MAT_FN.format(
      feat_space=inp['feat_space'],
      mat1='label',
      mat2='label'
    )
    helper.save_pickle(similarity_scores, save_fn)
    print_label_weight_pairs(W)

if __name__ == '__main__':
  while True: main()
