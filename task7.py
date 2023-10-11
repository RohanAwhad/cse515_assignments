#!python3

import config
import helper
import dimensionality_reduction
from feature_descriptor import FeatureDescriptor
import numpy as np
import similarity_metrics
import torch
from similarity_metrics import get_similarity, get_top_k_ids_n_scores

feature_descriptor = FeatureDescriptor(net=config.RESNET_MODEL)


def main():
  inp = helper.get_user_input('feat_space,K,dim_red,img_id',len(config.DATASET))
  _tmp = config.FEAT_DESC_FUNCS[inp['feat_space']]
  feat_db, idx_dict, similarity_metric = _tmp[config.FEAT_DB], _tmp[config.IDX], _tmp[config.SIMILARITY_METRIC]
  latent_space = helper.load_semantics(inp['feat_space']+"_"+inp['dim_red'])
  if isinstance(inp['img_id'], str): img = helper.load_img_file(inp['img_id'])
  else: img = config.DATASET[inp['img_id']][0]
  if img.mode != 'RGB': img = img.convert('RGB')
  query_feat = feature_descriptor.extract_features(img, inp['feat_space'])
  train_latent_space = np.dot(feat_db, np.transpose(latent_space))
  train_latent_space = torch.tensor(train_latent_space)
  query_latent_space = np.dot(query_feat, np.transpose(latent_space))
  query_latent_space = torch.tensor(query_latent_space)

  similarity_scores = get_similarity(query_latent_space, train_latent_space, similarity_metric)
  top_k_ids, top_k_scores = get_top_k_ids_n_scores(similarity_scores, idx_dict, inp['K'])
  top_k_imgs = [config.DATASET[x][0] for x in top_k_ids]
  helper.plot(img, inp['img_id'], top_k_imgs, top_k_ids, top_k_scores, inp['K'], inp['feat_space'], similarity_metric)



if __name__ == '__main__':
  while True: main()

