#!python3

import config
import helper
import dimensionality_reduction
from similarity_metrics import get_similarity, get_similarity_mat_x_mat


def print_img_id_weight_pairs(weight_mat, idx_dict):
  for i, row in enumerate(weight_mat):
    print()
    print(f'Image ID: {idx_dict[i][0]:4d} | Label: {idx_dict[i][1]:4d}')
    row = [(x, j) for j, x in enumerate(row)]
    row = sorted(row, key=lambda x: x[0], reverse=True)
    for x, j in row:
      print(f' - Latent Feat {j:3d}\t\tWeight := {x:10.3f}')

  print('='*53)


def main():
  inp = helper.get_user_input('feat_space,K,dim_red')
  _tmp = config.FEAT_DESC_FUNCS[inp['feat_space']]
  feat_db = _tmp[config.FEAT_DB]
  similarity_metric = _tmp[config.SIMILARITY_METRIC]
  idx_dict = _tmp[config.IDX]

  similarity_scores = get_similarity_mat_x_mat(feat_db, feat_db, similarity_metric)
  W, H = dimensionality_reduction.reduce_(similarity_scores, inp['K'], inp['dim_red'])
  helper.save_pickle(H, config.LATENT_SEMANTICS_FN.format(task='6_img_img_simi_mat', **inp))
  print_img_id_weight_pairs(W, idx_dict)

if __name__ == '__main__':
  while True: main()
