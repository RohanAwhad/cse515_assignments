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

torch.set_grad_enabled(False)
DEBUG = int(os.environ.get('DEBUG', 0))

#TORCH_HUB = './models/'
TORCH_HOME = os.environ.get('TORCH_HOME', None)
if TORCH_HOME is not None: torch.hub.set_dir(TORCH_HOME)

print('loading caltech101 dataset ...')
#DATA_DIR = './data/caltech101'
DATA_DIR = os.environ.get('DATA_DIR', None)
_download = False if DATA_DIR is not None else True
DATASET = torchvision.datasets.Caltech101(DATA_DIR, download=_download)

print('loading resnet models ...')
MODEL_NAME = 'ResNet50_Weights.DEFAULT'
RESNET_MODEL = torchvision.models.resnet50(weights=MODEL_NAME).eval()


# === Feature Store ===
from feature_descriptor import FeatureStore
SIMILARITY_METRIC = 0
IDX = 1
FEAT_DB = 2

# TODO (rohan): update var name
FEAT_DESC_FUNCS = FeatureStore(helper.load_data)
DISTANCE_MEASURES = ['manhattan_distance', 'kl_divergence']
# ===

FEATURES_DIR = 'features'
os.makedirs(FEATURES_DIR, exist_ok=True)
FEATURES_FN = FEATURES_DIR + '/{feat_space}.pkl'
LATENT_FEATURES_FN = FEATURES_DIR + '/task{task}_{feat_space}_{dim_red}_{K}.pkl'

LATENT_SEMANTICS_DIR = 'latent_semantics'
os.makedirs(LATENT_SEMANTICS_DIR, exist_ok=True)
LATENT_SEMANTICS_FN = 'latent_semantics/task{task}_{feat_space}_{dim_red}_{K}.pkl'

SIMI_MAT_DIR = 'similarity_matrices'
os.makedirs(SIMI_MAT_DIR, exist_ok=True)
SIMI_MAT_FN = SIMI_MAT_DIR + '/{feat_space}_{mat1}_{mat2}_mat.pkl'
