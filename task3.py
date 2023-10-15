#!python3
import numpy as np
import config
import helper
import dimensionality_reduction

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
  feat_db, idx_dict = _tmp[config.FEAT_DB], _tmp[config.IDX]
  W, H = dimensionality_reduction.reduce_(feat_db, inp['K'], inp['dim_red'])

    # Using scikit-learn SVD
    print("Using scikit-learn SVD:")
    print_img_id_weight_pairs(W, idx_dict)
    
    
    # Using custom SVD
    W_custom, H_custom = dimensionality_reduction.reduce_(feat_db, inp['K'], 'svd_custom')
    print("\nUsing custom SVD:")
    print_img_id_weight_pairs(W_custom, idx_dict)
    
    # Using scikit-learn NMF
    print("Using scikit-learn NMF:")
    print_img_id_weight_pairs(W, idx_dict)

    # Using custom NMF 
    W_custom, H_custom= dimensionality_reduction.reduce_(feat_db, inp['K'], 'nnmf_custom')
    
    print("Using custom NMF:")
    print_img_id_weight_pairs(W_custom,idx_dict)
    
    
    # Save the results from custom nnmf
    helper.save_pickle(H_custom, config.LATENT_SEMANTICS_FN.format(task=3, feat_space=inp['feat_space'], K=inp['K'], dim_red='nnmf_custom'))


if __name__ == '__main__':
  while True: main()

