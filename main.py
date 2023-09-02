#!/opt/homebrew/bin/python3

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF

from scipy import stats

from feature_descriptor import SaveOutput, FeatureDescriptor

# Custom variables
torch.set_grad_enabled(False)
torch.hub.set_dir('/Users/rohan/3_Resources/ai_models/torch_hub')


ds = torchvision.datasets.Caltech101('/Users/rohan/3_Resources/ai_datasets/caltech_101', download=False)
img = ds[0][0]
net = torchvision.models.resnet50(weights="ResNet50_Weights.DEFAULT").eval()
fd = FeatureDescriptor(net)

color_moments = fd.extract_features(img, descriptor='color_moment')
'''
hog_features = fd.extract_features(img, descriptor='hog')
layer3_features = fd.extract_features(img, descriptor='resnet_layer3')
avgpool_features = fd.extract_features(img, descriptor='resnet_avgpool')
fc_features = fd.extract_features(img, descriptor='resnet_fc')
'''

