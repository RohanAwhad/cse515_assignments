#!/opt/homebrew/bin/python3

import config
import helper
import similarity_metrics

from feature_descriptor import FeatureDescriptor
from similarity_metrics import get_similarity, get_top_k_ids_n_scores

# 3rd-party libs
import bz2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from numpy.linalg import norm


feature_descriptor = FeatureDescriptor(net=config.RESNET_MODEL)

def retrieve(img_id, feature_desc, K):
  if isinstance(img_id, str): img = helper.load_img_file(img_id)
  else: img = config.DATASET[img_id][0]
  if img.mode != 'RGB': img = img.convert('RGB')

  query_feat = feature_descriptor.extract_features(img, feature_desc)
  similarity_metric, feat_db_idx, feat_db = config.FEAT_DESC_FUNCS[feature_desc]
  similarity_scores = get_similarity(query_feat, feat_db, similarity_metric)
  top_k_ids, top_k_scores = get_top_k_ids_n_scores(similarity_scores, feat_db_idx, K)
  top_k_imgs = [config.DATASET[x][0] for x in top_k_ids]
  helper.plot(img, img_id, top_k_imgs, top_k_ids, top_k_scores, K, feature_desc, similarity_metric)


  

if __name__ == '__main__':
  while True:
    inp = helper.get_user_input('K,img_id,feat_space', len(config.DATASET))
    retrieve(inp['img_id'], inp['feat_space'], inp['K'])
