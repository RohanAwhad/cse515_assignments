#!/opt/homebrew/bin/python3

import helper

from feature_descriptor import FeatureDescriptor

# 3rd-party libs
import bz2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torchvision

# env setup
torch.set_grad_enabled(False)
torch.hub.set_dir('/Users/rohan/3_Resources/ai_models/torch_hub')

K = 10

print('Loading data ... ')
ds = torchvision.datasets.Caltech101('/Users/rohan/3_Resources/ai_datasets/caltech_101', download=False)
print('Loading model ... ')
feature_descriptor = FeatureDescriptor(
  net=torchvision.models.resnet50(weights="ResNet50_Weights.DEFAULT").eval()
)

print('Loading embeddings ... ')
COLOR_MMT_IDX, COLOR_MMT_FEATS = helper.load_data('color_moment')
HOG_IDX, HOG_FEATS = helper.load_data('hog')
RESNET_AVGPOOL_IDX, RESNET_AVGPOOL_FEATS = helper.load_data('resnet_avgpool')
RESNET_LAYER3_IDX, RESNET_LAYER3_FEATS = helper.load_data('resnet_layer3')
RESNET_FC_IDX, RESNET_FC_FEATS = helper.load_data('resnet_fc')

def retrieve(img_id):
  img = ds[img_id][0]
  if img.mode != 'RGB': img = img.convert('RGB')

  color_mmt = feature_descriptor.extract_color_moments(img).numpy()
  hog = feature_descriptor.extract_hog_features(img).numpy()
  resnet_layer3, resnet_avgpool, resnet_fc = tuple(map(lambda x: x.numpy(), feature_descriptor.extract_resnet_features(img)))

  top_k_imgs = []
  top_k_ids = []

  color_mmt_similarity = np.square(color_mmt[np.newaxis, :] - COLOR_MMT_FEATS).sum(-1) ** 0.5
  color_mmt_top_k_img_ids = color_mmt_similarity.argsort()[:K]
  color_mmt_top_k_imgs = [ds[COLOR_MMT_IDX[x]][0] for x in color_mmt_top_k_img_ids]
  top_k_imgs.append(color_mmt_top_k_imgs)
  top_k_ids.append([COLOR_MMT_IDX[x] for x in color_mmt_top_k_img_ids])

  hog_similarity = np.square(hog[np.newaxis, :] - HOG_FEATS).sum(-1) ** 0.5  # manhattan distance
  hog_top_k_img_ids = hog_similarity.argsort()[:K]
  hog_top_k_imgs = [ds[HOG_IDX[x]][0] for x in hog_top_k_img_ids]
  top_k_imgs.append(hog_top_k_imgs)
  top_k_ids.append([HOG_IDX[x] for x in color_mmt_top_k_img_ids])

  resnet_avgpool_similarity = np.square(resnet_avgpool[np.newaxis, :] - RESNET_AVGPOOL_FEATS).sum(-1) ** 0.5  # euclidean distance
  resnet_avgpool_top_k_img_ids = resnet_avgpool_similarity.argsort()[:K]
  resnet_avgpool_top_k_imgs = [ds[RESNET_AVGPOOL_IDX[x]][0] for x in resnet_avgpool_top_k_img_ids]
  top_k_imgs.append(resnet_avgpool_top_k_imgs)
  top_k_ids.append([RESNET_AVGPOOL_IDX[x] for x in color_mmt_top_k_img_ids])

  #resnet_layer3_similarity = resnet_layer3 @ RESNET_LAYER3_FEATS.T  # dot_product
  resnet_layer3_similarity = np.square(resnet_layer3[np.newaxis, :] - RESNET_LAYER3_FEATS).sum(-1) ** 0.5  # euclidean distance
  resnet_layer3_top_k_img_ids = resnet_layer3_similarity.argsort()[:K]
  resnet_layer3_top_k_imgs = [ds[RESNET_LAYER3_IDX[x]][0] for x in resnet_layer3_top_k_img_ids]
  top_k_imgs.append(resnet_layer3_top_k_imgs)
  top_k_ids.append([RESNET_LAYER3_IDX[x] for x in color_mmt_top_k_img_ids])

  resnet_fc_similarity = np.abs(resnet_fc[np.newaxis, :] - RESNET_FC_FEATS).sum(-1)  # manhattan distance
  resnet_fc_top_k_img_ids = resnet_fc_similarity.argsort()[:K]
  resnet_fc_top_k_imgs = [ds[RESNET_FC_IDX[x]][0] for x in resnet_fc_top_k_img_ids]
  top_k_imgs.append(resnet_fc_top_k_imgs)
  top_k_ids.append([RESNET_FC_IDX[x] for x in color_mmt_top_k_img_ids])

  helper.save_top_k(img, img_id, top_k_imgs, top_k_ids, K, f'outputs/{img_id}.png')

if __name__ == '__main__':
  while True:
    img_id = int(input(f'Enter an image id [0, {len(ds)-1}]: '))
    if img_id < 0 or img_id >= len(ds): print(f'img id invalid. should be between [0, {len(ds)-1}], try again. you got it!')
    else: retrieve(img_id)
