#!/opt/homebrew/bin/python3
'''
Changes made since Phase 1 submission
- moved config vars to config file
- renamed task2 to task0a for phase 2
- added storage of labels for phase 2
'''

import config
import helper

from feature_descriptor import FeatureDescriptor

# 3rd-party libs
import torch

from tqdm import tqdm


feature_descriptor = FeatureDescriptor(config.RESNET_MODEL)

for fd in ('color_moment', 'hog'):
  embd_idx_to_img_id = dict()
  features_list = []
  for img_id, (img, label) in tqdm(enumerate(config.DATASET), total=len(config.DATASET), desc=f'{fd} feats'):
    if img_id % 2: continue
    if img.mode == 'L': img = img.convert('RGB')
    feats = feature_descriptor.extract_features(img, fd)
    if not isinstance(feats, torch.Tensor): continue
    embd_idx_to_img_id[len(features_list)] = (img_id, label)
    features_list.append(feats)

  helper.save_data((embd_idx_to_img_id, torch.stack(features_list).numpy()), fd)


embd_idx_to_img_id = dict()
l3_list, ap_list, fc_list = [], [], []
for img_id, (img, _) in tqdm(enumerate(config.DATASET), total=len(config.DATASET), desc='ResNet feats'):
  if img_id % 2: continue
  if img.mode == 'L': img = img.convert('RGB')
  tmp = feature_descriptor.extract_resnet_features(img)
  if tmp == 2: continue
  embd_idx_to_img_id[len(l3_list)] = (img_id, label)
  l3, ap, fc = tmp
  l3_list.append(l3)
  ap_list.append(ap)
  fc_list.append(fc)

helper.save_data((embd_idx_to_img_id, torch.stack(l3_list).numpy()), 'resnet_layer3')
helper.save_data((embd_idx_to_img_id, torch.stack(ap_list).numpy()), 'resnet_avgpool')
helper.save_data((embd_idx_to_img_id, torch.stack(fc_list).numpy()), 'resnet_fc')

