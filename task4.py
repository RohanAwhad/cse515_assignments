#!python3

import config
import helper
import dimensionality_reduction
from config import DEBUG

import torch
import numpy as np

def print_label_weight_pairs(weight_mat):
  for i, row in enumerate(weight_mat.T):
    print()
    print(f'Latent Feat: {i:3d}')
    row = [(x, j) for j, x in enumerate(row)]
    row = sorted(row, key=lambda x: x[0], reverse=True)
    for x, j in row:
      print(f' - Label {j:3d}\t\tWeight := {x:10.6f}')

  print('='*53)

def create_tensor(weight_mat, idx_dict):
  m, n = weight_mat.shape
  tensor = torch.zeros(m, n, max(config.DATASET.y) + 1)
  for i, row in enumerate(weight_mat): tensor[i, :, idx_dict[i][1]] = row
  return tensor


def main():
  inp = helper.get_user_input('feat_space,K')
  inp['dim_red'] = 'cp'
  
  _tmp = config.FEAT_DESC_FUNCS[inp['feat_space']]
  feat_db, idx_dict = _tmp[config.FEAT_DB], _tmp[config.IDX]
  tensor = create_tensor(feat_db, idx_dict)

  if DEBUG: print(f'Created tensor of shape {tensor.shape}')
  W, _ = dimensionality_reduction.reduce_(tensor, inp['K'], 'cp')

  for j, i in enumerate(['image','feature','label']):
    helper.save_pickle(np.transpose(W[j]), config.LATENT_SEMANTICS_MODES_FN.format(task=4, mode=i, **inp))

  print_label_weight_pairs(W[2])


if __name__ == '__main__':
  while True: main()

