#!python3

import config
import helper
import dimensionality_reduction
import torch
import numpy as np

def print_img_id_weight_pairs(weight_mat, idx_dict):
  for j in range(weight_mat.shape[1]):
    print()
    print(f'Latent feature : {j}')
    enum_arr = list(enumerate(weight_mat[:,j]))

    # Sort the array in descending order based on values
    sorted_arr = sorted(enum_arr, key=lambda x: x[1], reverse=True)

    # Print the sorted values along with their original indices
    for index, value in sorted_arr:
        print(f"Label: {index}, Weight: {value}")

  print('='*53)

def create_tensor(weight_mat, idx_dict):
  #Discuss other ways
  tensor = torch.zeros(weight_mat.shape[0],weight_mat.shape[1],max(config.DATASET.y) + 1)
  for i, row in enumerate(weight_mat):
    # print()
    label = idx_dict[i][1]
    tensor[i,:,label] = row
  return tensor


def main():
  inp = helper.get_user_input('feat_space,K')
  inp['dim_red'] = 'cp'
  _tmp = config.FEAT_DESC_FUNCS[inp['feat_space']]
  feat_db, idx_dict = _tmp[config.FEAT_DB], _tmp[config.IDX]
  tensor = create_tensor(feat_db,idx_dict)
  W,_ = dimensionality_reduction.reduce_(tensor, inp['K'], 'cp')
  helper.save_pickle(np.transpose(W), config.LATENT_SEMANTICS_FN.format(task=4, **inp))
  print_img_id_weight_pairs(W, idx_dict)


if __name__ == '__main__':
  while True: main()

