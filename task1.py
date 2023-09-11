#!/opt/homebrew/bin/python3

import json
import torch
import torchvision

from feature_descriptor import FeatureDescriptor

# Custom variables
torch.set_grad_enabled(False)
torch.hub.set_dir('/Users/rohan/3_Resources/ai_models/torch_hub')
FEATURE_DESCRIPTOR_DICT = {
  1: 'color_moment',
  2: 'hog',
  3: 'resnet_layer3',
  4: 'resnet_avgpool',
  5: 'resnet_fc',
}

ds = torchvision.datasets.Caltech101('/Users/rohan/3_Resources/ai_datasets/caltech_101', download=False)
fd = FeatureDescriptor(
  net=torchvision.models.resnet50(weights="ResNet50_Weights.DEFAULT").eval()
)

# === Task 1 ===
img_id = int(input(f'Enter an image id [0, {len(ds)-1}]: '))
if img_id < 0 or img_id >= len(ds): raise ValueError(f'img id invalid. should be between [0, {len(ds)-1}]')
fd_id = int(input('''Enter id of feature descriptor

1: color_moment
2: hog
3: resnet_layer3
4: resnet_avgpool
5: resnet_fc

>'''))
if not (0 < fd_id < 6): raise ValueError('value should be between [1, 5]')

img = ds[img_id][0]
if img.mode == 'L': img = img.convert('RGB')
features = fd.extract_features(img, FEATURE_DESCRIPTOR_DICT[fd_id])

print(json.dumps(features.tolist(), indent=2))
img.show()

# TODO (rohan): visualize/print features
