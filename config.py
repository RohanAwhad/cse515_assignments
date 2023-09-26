'''
Changes since Phase 1 submission:
- created this new file and moved global config vars here
'''

import torch
import torchvision
import helper
import similarity_metrics

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
SIMILARITY_FUNC = 0
METRIC = 1
IDX = 2
FEAT_DB = 3

FEAT_DESC_FUNCS = {
  'color_moment': (similarity_metrics.pearson_coefficient, 'pearson_coefficient',COLOR_MMT_IDX, COLOR_MMT_FEATS),
  'hog': (similarity_metrics.intersection_similarity, 'intersection_similarity',HOG_IDX, HOG_FEATS),
  'resnet_avgpool': (similarity_metrics.cosine_similarity, 'cosine_similarity',RESNET_AVGPOOL_IDX, RESNET_AVGPOOL_FEATS),
  'resnet_layer3': (similarity_metrics.cosine_similarity, 'cosine_similarity',RESNET_LAYER3_IDX, RESNET_LAYER3_FEATS),
  'resnet_fc': (similarity_metrics.manhattan_distance, 'manhattan_distance',RESNET_FC_IDX, RESNET_FC_FEATS),
}

SIMILARITY_METRIC_FUNC = {
  'pearson_coefficient': similarity_metrics.pearson_coefficient,
  'intersection_similarity': similarity_metrics.intersection_similarity,
  'cosine_similarity': similarity_metrics.cosine_similarity,
  'manhattan_distance': similarity_metrics.manhattan_distance,
}
