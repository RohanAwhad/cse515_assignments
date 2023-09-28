'''
Changes since Phase 1 submission:
- created this new file and moved global config vars here
'''

import helper

# 3rd-party libs

import os
import torch
import torchvision

TORCH_HUB = './models/'
torch.set_grad_enabled(False)
torch.hub.set_dir(TORCH_HUB)

print('loading caltech101 dataset ...')
DATA_DIR = './data/caltech101'
DATASET = torchvision.datasets.Caltech101(DATA_DIR, download=False)

print('loading resnet models ...')
MODEL_NAME = 'ResNet50_Weights.DEFAULT'
RESNET_MODEL = torchvision.models.resnet50(weights=MODEL_NAME).eval()

print('loading embeddings ... ')
COLOR_MMT_IDX, COLOR_MMT_FEATS = helper.load_data('color_moment')
HOG_IDX, HOG_FEATS = helper.load_data('hog')
RESNET_AVGPOOL_IDX, RESNET_AVGPOOL_FEATS = helper.load_data('resnet_avgpool')
RESNET_LAYER3_IDX, RESNET_LAYER3_FEATS = helper.load_data('resnet_layer3')
RESNET_FC_IDX, RESNET_FC_FEATS = helper.load_data('resnet_fc')

# preprocessing feats db
HOG_FEATS = torch.tensor(HOG_FEATS)

# ---
SIMILARITY_METRIC = 0
IDX = 1
FEAT_DB = 2

FEAT_DESC_FUNCS = {
  'color_moment': ('pearson_coefficient',COLOR_MMT_IDX, COLOR_MMT_FEATS),
  'hog': ('intersection_similarity',HOG_IDX, HOG_FEATS),
  'resnet_avgpool': ('cosine_similarity',RESNET_AVGPOOL_IDX, RESNET_AVGPOOL_FEATS),
  'resnet_layer3': ('cosine_similarity',RESNET_LAYER3_IDX, RESNET_LAYER3_FEATS),
  'resnet_fc': ('manhattan_distance',RESNET_FC_IDX, RESNET_FC_FEATS),
}

LATENT_SEMANTICS_DIR = 'latent_semantics'
os.makedirs(LATENT_SEMANTICS_DIR, exist_ok=True)
LATENT_SEMANTICS_FN = 'latent_semantics/task{task}_{feat_space}_{dim_red}_{K}.pkl'
