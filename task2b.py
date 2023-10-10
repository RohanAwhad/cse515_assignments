#!python3

import config
import helper

from feature_descriptor import FeatureDescriptor
from similarity_metrics import get_top_k_ids_n_scores, get_similarity

# 3rd-party libs
import functools
import torch

feature_descriptor = FeatureDescriptor(config.RESNET_MODEL)

@functools.lru_cache
def get_label_vecs(feat_space: str):
  ret = []
  for label in range(max(config.DATASET.y) + 1):
    # get images feats for the given label
    *_, feat_db_idx, feat_db = config.FEAT_DESC_FUNCS[feat_space]
    feat_db = torch.nn.functional.softmax(feat_db, dim=1)
    label_feats = [x[0] for x in filter(lambda x: x[1][1] == label, feat_db_idx.items())]

    # pool feat_matrix into vector
    # TODO (rohan): Decide a better pooling method
    label_feat = torch.tensor(feat_db[label_feats, :].mean(0))
    ret.append(label_feat)

  return torch.stack(ret, dim=0)
  

def retrieve(img_id,K, feat_space = 'resnet_fc'):
  feat_db = get_label_vecs(feat_space)
  feat_db_idx = dict((x, (x,)) for x in range(len(feat_db)))

  if isinstance(img_id, str): img = helper.load_img_file(img_id)
  else: img = config.DATASET[img_id][0]
  if img.mode != 'RGB': img = img.convert('RGB')

  query_feat = feature_descriptor.extract_features(img, feat_space)
  query_feat = torch.nn.functional.softmax(query_feat, dim=0)
  similarity_metric = config.FEAT_DESC_FUNCS[feat_space][config.SIMILARITY_METRIC]

  if not isinstance(query_feat, torch.Tensor): query_feat = torch.tensor(query_feat)  # TODO (rohan): make torch universal and remove dependency on numpy
  similarity_scores = get_similarity(query_feat, feat_db, similarity_metric)
  top_k_ids, top_k_scores = get_top_k_ids_n_scores(similarity_scores, feat_db_idx, K)

  # print output
  print('-'*50)
  print(f'Labels similar to img_id: {img_id}')
  for idx, score in zip(top_k_ids, top_k_scores): print(f'  - Label: {idx} | Score: {score}')
  print(f'\n  - Original Label: {config.DATASET[img_id][1]}')
  print('-'*50)


if __name__ == '__main__':
  inp = {
    'img_id': 3,
    'feat_space': 'hog',
    'K': 5,
  }
  while True:
    inp = helper.get_user_input('img_id,K', len(config.DATASET), max(config.DATASET.y))

    retrieve(**inp)
  
