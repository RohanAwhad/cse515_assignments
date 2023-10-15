#!/opt/homebrew/bin/python3
import config

from feature_descriptor import FeatureDescriptor

import json

# Custom variables
FEATURE_DESCRIPTOR_DICT = {
  1: 'color_moment',
  2: 'hog',
  3: 'resnet_layer3',
  4: 'resnet_avgpool',
  5: 'resnet_fc',
}
FEAT_DESC = FeatureDescriptor(net=config.RESNET_MODEL)


# === Task 1 ===
img_id = int(input(f'Enter an image id [0, {len(config.DATASET)-1}]: '))
if img_id < 0 or img_id >= len(config.DATASET): raise ValueError(f'img id invalid. should be between [0, {len(config.DATASET)-1}]')
fd_id = int(input('''Enter id of feature descriptor

1: color_moment
2: hog
3: resnet_layer3
4: resnet_avgpool
5: resnet_fc

>'''))
if not (0 < fd_id < 6): raise ValueError('value should be between [1, 5]')

img = config.DATASET[img_id][0]
if img.mode == 'L': img = img.convert('RGB')
features = FEAT_DESC.extract_features(img, FEATURE_DESCRIPTOR_DICT[fd_id])

print(json.dumps(features.tolist(), indent=2))
img.show()

# TODO (rohan): visualize/print features
