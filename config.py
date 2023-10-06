'''
Changes since Phase 1 submission:
- created this new file and moved global config vars here
'''

import helper

# python libs
import os
from typing import Dict, Tuple, Union

# 3rd-party libs

import numpy as np
import torch
import torchvision

DEBUG = int(os.environ.get('DEBUG', 0))
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
if not isinstance(COLOR_MMT_FEATS, torch.Tensor): COLOR_MMT_FEATS = torch.tensor(COLOR_MMT_FEATS)
if not isinstance(HOG_FEATS, torch.Tensor): HOG_FEATS = torch.tensor(HOG_FEATS)
if not isinstance(RESNET_AVGPOOL_FEATS, torch.Tensor): RESNET_AVGPOOL_FEATS = torch.tensor(RESNET_AVGPOOL_FEATS)
if not isinstance(RESNET_LAYER3_FEATS, torch.Tensor): RESNET_LAYER3_FEATS = torch.tensor(RESNET_LAYER3_FEATS)
if not isinstance(RESNET_FC_FEATS, torch.Tensor): RESNET_FC_FEATS = torch.tensor(RESNET_FC_FEATS)

# ---
SIMILARITY_METRIC = 0
IDX = 1
FEAT_DB = 2

FEAT_DESC_FUNCS: Dict[str, Tuple[str, Dict[int, Tuple[int, int]], torch.Tensor]] = {
  'color_moment': ('pearson_coefficient',COLOR_MMT_IDX, COLOR_MMT_FEATS),
  'hog': ('intersection_similarity',HOG_IDX, HOG_FEATS),
  'resnet_avgpool': ('cosine_similarity',RESNET_AVGPOOL_IDX, RESNET_AVGPOOL_FEATS),
  'resnet_layer3': ('cosine_similarity',RESNET_LAYER3_IDX, RESNET_LAYER3_FEATS),
  'resnet_fc': ('manhattan_distance',RESNET_FC_IDX, RESNET_FC_FEATS),
}

LATENT_SEMANTICS_DIR = 'latent_semantics'
os.makedirs(LATENT_SEMANTICS_DIR, exist_ok=True)
LATENT_SEMANTICS_FN = 'latent_semantics/task{task}_{feat_space}_{dim_red}_{K}.pkl'
