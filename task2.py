#!/opt/homebrew/bin/python3

import bz2
import pickle
import torch
import torchvision

from tqdm import tqdm

from feature_descriptor import FeatureDescriptor

# Custom variables
torch.set_grad_enabled(False)
torch.hub.set_dir('/Users/rohan/3_Resources/ai_models/torch_hub')

def save_data(data_tuple, fd):
  binary_file = pickle.dumps(data_tuple)
  compressed_bin_file = bz2.compress(binary_file)
  print(fd)
  print(f'- orginal size: {len(binary_file) / 1e6} MB')
  print(f'- compressed size: {len(compressed_bin_file) / 1e6} MB')
  with open(f'features/{fd}.bin', 'wb') as f: f.write(compressed_bin_file)


ds = torchvision.datasets.Caltech101('/Users/rohan/3_Resources/ai_datasets/caltech_101', download=False)
feature_descriptor = FeatureDescriptor(
  net=torchvision.models.resnet50(weights="ResNet50_Weights.DEFAULT").eval()
)

for fd in ('color_moment', 'hog'):
  embd_idx_to_img_id = dict()
  features_list = []
  for img_id, (img, _) in tqdm(enumerate(ds), total=len(ds), desc=f'{fd} feats'):
    feats = feature_descriptor.extract_features(img, fd)
    if not isinstance(feats, torch.Tensor): continue
    features_list.append(feats)
    embd_idx_to_img_id[len(features_list)] = img_id

  save_data((embd_idx_to_img_id, torch.stack(features_list).numpy()), fd)


embd_idx_to_img_id = dict()
l3_list, ap_list, fc_list = [], [], []
for img_id, (img, _) in tqdm(enumerate(ds), total=len(ds), desc='ResNet feats'):
  tmp = feature_descriptor.extract_resnet_features(img)
  if tmp == 2: continue
  l3, ap, fc = tmp
  l3_list.append(l3)
  ap_list.append(ap)
  fc_list.append(fc)

save_data((embd_idx_to_img_id, torch.stack(l3_list).numpy()), 'resnet_layer3')
save_data((embd_idx_to_img_id, torch.stack(ap_list).numpy()), 'resnet_avgpool')
save_data((embd_idx_to_img_id, torch.stack(fc_list).numpy()), 'resnet_fc')

