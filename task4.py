#!python3

import config
import helper
import dimensionality_reduction
import torch

def print_img_id_weight_pairs(weight_mat, idx_dict):
  for i, row in enumerate(weight_mat):
    print()
    print(f'Image ID: {idx_dict[i][0]:4d} | Label: {idx_dict[i][1]:4d}')
    row = [(x, j) for j, x in enumerate(row)]
    row = sorted(row, key=lambda x: x[0], reverse=True)
    for x, j in row:
      print(f' - Latent Feat {j:3d}\t\tWeight := {x:10.3f}')

  print('='*53)

def create_tensor(weight_mat, idx_dict):
  tensor = torch.zeros(weight_mat.shape[0],weight_mat.shape[1],max(config.DATASET.y) + 1)
  for i, row in enumerate(weight_mat):
    # print()
    label = idx_dict[i][1]
    tensor[:,:,label] = row
    

  return tensor


def main():
  inp = helper.get_user_input('feat_space,K')
  _tmp = config.FEAT_DESC_FUNCS[inp['feat_space']]
  feat_db, idx_dict = _tmp[config.FEAT_DB], _tmp[config.IDX]
  tensor = create_tensor(feat_db,idx_dict)
  W,_ = dimensionality_reduction.reduce_(tensor, inp['K'], 'cp')
#   helper.save_pickle(H, config.LATENT_SEMANTICS_FN.format(task=3, **inp))
  print_img_id_weight_pairs(W, idx_dict)


if __name__ == '__main__':
  while True: main()

