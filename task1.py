#!/opt/homebrew/bin/python3

import config
import helper

from feature_descriptor import FeatureDescriptor

# 3rd-party libs
import torch

feature_descriptor = FeatureDescriptor(net=config.RESNET_MODEL)

def retrieve(label, feature_desc, K):
  # calc label vec

  # get images feats for the given label
  *_, feat_db_idx, feat_db = config.FEAT_DESC_FUNCS[feature_desc]
  print(feat_db_idx)
  label_feats = [x[0] for x in filter(lambda x: x[1][1] == label, feat_db_idx.items())]
  #label_feats = list(map(lambda x: x[0], filter(lambda x:x[1][1] == label, feat_db_idx.items())))  # less readable and long
  print(label_feats)
  print(type(label), feature_desc, K, len(label_feats))
  print(feat_db[label_feats, :].shape)

  # pool feat_matrix into vector
  label_feat = torch.tensor(feat_db[label_feats, :].mean(0))
  print(label_feat.shape)

  similarity_metric, feat_db_idx, feat_db = config.FEAT_DESC_FUNCS[feature_desc]
  simiarity_scores = get_similarity(label_feat, feat_db, similarity_metric)
  top_k_ids, top_k_scores = get_top_k_ids_n_scores(similarity_scores, feat_db_idx, K)
  top_k_imgs = [config.DATASET[x][0] for x in top_k_ids]
  helper.plot(None, None, top_k_imgs, top_k_ids, top_k_scores, K, feature_desc, similarity_metric)

if __name__ == '__main__':
  while True:
    inp = helper.get_user_input('K,feat_space,label', len(config.DATASET), len(set(config.DATASET.y)))
    retrieve(inp['label'], inp['feat_space'], inp['K'])
