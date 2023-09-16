#!/opt/homebrew/bin/python3

import config
import helper
import similarity_metrics

from feature_descriptor import FeatureDescriptor

# 3rd-party libs
import bz2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from numpy.linalg import norm

K = int(input('Input K for top-k: '))

feature_descriptor = FeatureDescriptor(net=config.RESNET_MODEL)

print('Loading embeddings ... ')
COLOR_MMT_IDX, COLOR_MMT_FEATS = helper.load_data('color_moment')
HOG_IDX, HOG_FEATS = helper.load_data('hog')
RESNET_AVGPOOL_IDX, RESNET_AVGPOOL_FEATS = helper.load_data('resnet_avgpool')
RESNET_LAYER3_IDX, RESNET_LAYER3_FEATS = helper.load_data('resnet_layer3')
RESNET_FC_IDX, RESNET_FC_FEATS = helper.load_data('resnet_fc')

HOG_FEATS = torch.tensor(HOG_FEATS)
class Retriever:
# TODO (rohan): Instead of having feature wise retrieval, wouldn't it better if we could have distance calculation functions, which would give back similarity scores and indices, based on argmin, or argmax
  def __init__(self,
    color_mmt_idx,
    color_mmt_feats,
    hog_idx,
    hog_feats,
    resnet_avgpool_idx,
    resnet_avgpool_feats,
    resnet_layer3_idx,
    resnet_layer3_feats,
    resnet_fc_idx,
    resnet_fc_feats,
  ):
    self.color_mmt_feats = color_mmt_feats
    self.hog_feats = hog_feats
    self.resnet_avgpool_feats = resnet_avgpool_feats
    self.resnet_layer3_feats = resnet_layer3_feats
    self.resnet_fc_feats = resnet_fc_feats

    self.color_mmt_idx = color_mmt_idx
    self.hog_idx = hog_idx
    self.resnet_avgpool_idx = resnet_avgpool_idx
    self.resnet_layer3_idx = resnet_layer3_idx
    self.resnet_fc_idx = resnet_fc_idx

  def retrieve_using_color_moments(self, query_feats):
    '''Retrieve using Pearson Coefficient'''
    return similarity_metrics.pearson_coefficient(query_feats, self.color_mmt_feats, self.color_mmt_idx, K)

  def retrieve_using_hog(self, query_feats):
    '''Retrieve using Intersection Similarity'''
    return similarity_metrics.intersection_similarity(query_feats, self.hog_feats, self.hog_idx, K)

  def retrieve_using_resnet_avgpool(self, query_feats):
    '''Retrieve using Cosine Similarity '''
    return similarity_metrics.cosine_similarity(query_feats, self.resnet_avgpool_feats, self.resnet_avgpool_idx, K)

  def retrieve_using_resnet_layer3(self, query_feats):
    '''Retrieve using Cosine Similarity '''
    return similarity_metrics.cosine_similarity(query_feats, self.resnet_layer3_feats, self.resnet_layer3_idx, K)

  def retrieve_using_resnet_fc(self, query_feats):
    '''Retrieve using Manhattan Distance'''
    return similarity_metrics.manhattan_distance(query_feats, self.resnet_fc_feats, self.resnet_fc_idx, K)
    query_feats = query_feats.numpy()
    similarity_scores = np.abs(query_feats[np.newaxis, :] - self.resnet_fc_feats).sum(-1)  # manhattan distance
    ss_arg_idx = similarity_scores.argsort()[:K]
    top_k_scores = [similarity_scores[x] for x in ss_arg_idx]
    top_k_idx = [self.resnet_layer3_idx[x][0] for x in ss_arg_idx]
    return top_k_idx, top_k_scores


retriever = Retriever(
  COLOR_MMT_IDX,
  COLOR_MMT_FEATS,
  HOG_IDX,
  HOG_FEATS,
  RESNET_AVGPOOL_IDX,
  RESNET_AVGPOOL_FEATS,
  RESNET_LAYER3_IDX,
  RESNET_LAYER3_FEATS,
  RESNET_FC_IDX,
  RESNET_FC_FEATS,
)

FEAT_DESC_TO_SIMILARITY_METRIC = {
  'color_moment': 'pearson_coefficient',
  'hog': 'intersection_similarity',
  'resnet_avgpool': 'cosine_similarity',
  'resnet_layer3': 'cosine_similarity',
  'resnet_fc': 'manhattan_distance'
}
FEAT_DESC_TO_RETRIEVAL_FUNC = {
  'color_moment': retriever.retrieve_using_color_moments,
  'hog': retriever.retrieve_using_hog,
  'resnet_avgpool': retriever.retrieve_using_resnet_avgpool,
  'resnet_layer3': retriever.retrieve_using_resnet_layer3,
  'resnet_fc': retriever.retrieve_using_resnet_fc,
}

def retrieve(img_id, feature_desc):
  img = config.DATASET[img_id][0]
  if img.mode != 'RGB': img = img.convert('RGB')

  '''
  color_mmt = feature_descriptor.extract_color_moments(img)#.numpy()
  hog = feature_descriptor.extract_hog_features(img)
  resnet_layer3, resnet_avgpool, resnet_fc = tuple(map(lambda x: x.numpy(), feature_descriptor.extract_resnet_features(img)))
  '''

  #top_k_ids, top_k_scores = retriever.retrieve_using_resnet_avgpool(resnet_avgpool)
  top_k_ids, top_k_scores = FEAT_DESC_TO_RETRIEVAL_FUNC[feature_desc](feature_descriptor.extract_features(img, feature_desc))
  top_k_imgs = [config.DATASET[x][0] for x in top_k_ids]
  helper.plot(img, img_id, top_k_imgs, top_k_ids, top_k_scores, K, feature_desc, FEAT_DESC_TO_SIMILARITY_METRIC[feature_desc])

  '''

  # intersection similarity

  resnet_avgpool_similarity = (resnet_avgpool @ RESNET_AVGPOOL_FEATS.T) / (norm(resnet_avgpool) * norm(RESNET_AVGPOOL_FEATS.T, axis=0))  # cosine similarity
  resnet_avgpool_top_k_img_ids = resnet_avgpool_similarity.argsort()[-(K+1):-1][::-1]
  resnet_avgpool_top_k_imgs = [config.DATASET[RESNET_AVGPOOL_IDX[x]][0] for x in resnet_avgpool_top_k_img_ids]
  top_k_imgs.append(resnet_avgpool_top_k_imgs)
  top_k_ids.append([RESNET_AVGPOOL_IDX[x] for x in resnet_avgpool_top_k_img_ids])

  resnet_layer3_similarity = (resnet_layer3 @ RESNET_LAYER3_FEATS.T) / (norm(resnet_layer3) * norm(RESNET_LAYER3_FEATS.T, axis=0))  # cosine similarity
  resnet_layer3_top_k_img_ids = resnet_layer3_similarity.argsort()[-(K+1):-1][::-1]
  resnet_layer3_top_k_imgs = [config.DATASET[RESNET_LAYER3_IDX[x]][0] for x in resnet_layer3_top_k_img_ids]
  top_k_imgs.append(resnet_layer3_top_k_imgs)
  top_k_ids.append([RESNET_LAYER3_IDX[x] for x in resnet_layer3_top_k_img_ids])

  resnet_fc_similarity = np.abs(resnet_fc[np.newaxis, :] - RESNET_FC_FEATS).sum(-1)  # manhattan distance
  resnet_fc_top_k_img_ids = resnet_fc_similarity.argsort()[1:K+1]
  resnet_fc_top_k_imgs = [config.DATASET[RESNET_FC_IDX[x]][0] for x in resnet_fc_top_k_img_ids]
  top_k_imgs.append(resnet_fc_top_k_imgs)
  top_k_ids.append([RESNET_FC_IDX[x] for x in resnet_fc_top_k_img_ids])

  helper.save_top_k(img, img_id, top_k_imgs, top_k_ids, K, f'outputs/{img_id}.png')
  '''

if __name__ == '__main__':
  while True:
    img_id = int(input(f'Enter an image id [0, {len(config.DATASET)-1}]: '))
    if img_id < 0 or img_id >= len(config.DATASET): print(f'img id invalid. should be between [0, {len(config.DATASET)-1}], try again. you got it!')
    else:
      retrieve(img_id, 'color_moment')
      retrieve(img_id, 'hog')
      retrieve(img_id, 'resnet_avgpool')
      retrieve(img_id, 'resnet_layer3')
      retrieve(img_id, 'resnet_fc')
