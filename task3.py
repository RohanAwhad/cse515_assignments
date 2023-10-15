#!python3
import numpy as np
import config
import helper
import dimensionality_reduction
from config import DEBUG

def print_img_id_weight_pairs(weight_mat, idx_dict):
  '''
  for i, row in enumerate(weight_mat):
    print()
    print(f'Image ID: {idx_dict[i][0]:4d} | Label: {idx_dict[i][1]:4d}')
    row = [(x, j) for j, x in enumerate(row)]
    row = sorted(row, key=lambda x: x[0], reverse=True)
    for x, j in row:
      print(f' - Latent Feat {j:3d}\t\tWeight := {x:10.3f}')
  '''
  for i, row in enumerate(weight_mat.T):
    print()
    print(f'Latent Feature ID: {i:4d}')
    row = [(x, j) for j, x in enumerate(row)]
    row = sorted(row, key=lambda x: x[0], reverse=True)
    _n = len(row) if DEBUG else 30
    print(f' Top {_n} Image ID, Label and Weights')
    for x, j in row[:_n]: print(f' - Image ID {idx_dict[j][0]:4d}\tLabel: {idx_dict[j][1]:4d}\tWeight := {x:10.3f}')

  print('='*53)


def main():
  inp = helper.get_user_input('feat_space,K,dim_red')
  _tmp = config.FEAT_DESC_FUNCS[inp['feat_space']]
  feat_db, idx_dict = _tmp[config.FEAT_DB], _tmp[config.IDX]
  W, H = dimensionality_reduction.reduce_(feat_db, inp['K'], inp['dim_red'])  # W: img latent feature vectors, H: latent to original feature vectors
  print(f'W: {W.shape}, H: {H.shape}')

  helper.save_pickle(H, config.LATENT_SEMANTICS_FN.format(task=3, **inp))
  helper.save_pickle(W, config.LATENT_FEATURES_FN.format(task=3, **inp))
  
  print_img_id_weight_pairs(W, idx_dict)


if __name__ == '__main__':
  while True: main()

