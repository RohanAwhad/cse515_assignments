#!python3

import config
import helper
import dimensionality_reduction


if __name__ == '__main__':
  while True:
    inp = helper.get_user_input('feat_space,K,dim_red')
    feat_db = config.FEAT_DESC_FUNCS[inp['feat_space']][config.FEAT_DB]
    W, H = dimensionality_reduction.reduce_(feat_db, inp['K'], inp['dim_red'])
    helper.save_pickle(H, config.LATENT_SEMANTICS_FN.format(task=3, **inp))

  
