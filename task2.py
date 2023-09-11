#!/opt/homebrew/bin/python3

import helper

from feature_descriptor import FeatureDescriptor

# 3rd-party libs
import torch
import torchvision

from tqdm import tqdm

# TODO (rohan): move this to config
# env setup 
torch.set_grad_enabled(False)
torch.hub.set_dir('/Users/rohan/3_Resources/ai_models/torch_hub')

# TODO (rohan): move this to helper


# TODO (rohan): move this to config
# TODO (rohan): soft link "./data/caltech101" to the SSD
ds = torchvision.datasets.Caltech101('/Users/rohan/3_Resources/ai_datasets/caltech_101', download=False)
feature_descriptor = FeatureDescriptor(
  net=torchvision.models.resnet50(weights="ResNet50_Weights.DEFAULT").eval()
)

# TODO (rohan): instead of saving each FD as a single file, can save just one. But that would require changing task3.py
for fd in ('color_moment', 'hog'):
  embd_idx_to_img_id = dict()
  features_list = []
  for img_id, (img, _) in tqdm(enumerate(ds), total=len(ds), desc=f'{fd} feats'):
    if img.mode == 'L': img = img.convert('RGB')
    feats = feature_descriptor.extract_features(img, fd)
    if not isinstance(feats, torch.Tensor): continue
    embd_idx_to_img_id[len(features_list)] = img_id
    features_list.append(feats)

  helper.save_data((embd_idx_to_img_id, torch.stack(features_list).numpy()), fd)


embd_idx_to_img_id = dict()
l3_list, ap_list, fc_list = [], [], []
for img_id, (img, _) in tqdm(enumerate(ds), total=len(ds), desc='ResNet feats'):
  if img.mode == 'L': img = img.convert('RGB')
  tmp = feature_descriptor.extract_resnet_features(img)
  if tmp == 2: continue
  embd_idx_to_img_id[len(l3_list)] = img_id
  l3, ap, fc = tmp
  l3_list.append(l3)
  ap_list.append(ap)
  fc_list.append(fc)

helper.save_data((embd_idx_to_img_id, torch.stack(l3_list).numpy()), 'resnet_layer3')
helper.save_data((embd_idx_to_img_id, torch.stack(ap_list).numpy()), 'resnet_avgpool')
helper.save_data((embd_idx_to_img_id, torch.stack(fc_list).numpy()), 'resnet_fc')

