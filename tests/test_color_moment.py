import torch
import torchvision
import torchvision.transforms.functional as TF
from scipy import stats

from feature_descriptor import FeatureDescriptor

torch.set_grad_enabled(False)
torch.hub.set_dir('/Users/rohan/3_Resources/ai_models/torch_hub')

def test_mean_mmt():
  ds = torchvision.datasets.Caltech101('/Users/rohan/3_Resources/ai_datasets/caltech_101', download=False)
  img = ds[0][0]

  # expected
  img = TF.resize(img, size=(100, 300))
  img = TF.to_tensor(img)
  C, H, W = img.size()
  row_stride, col_stride = H//10, W//10
  expected = []
  for channel in range(C):
    for i in range(0, H, row_stride):
      for j in range(0, W, col_stride):
        expected.append(img[channel, i:i+row_stride, j:j+col_stride].mean())
  expected = torch.tensor(expected)

  # actual
  net = torchvision.models.resnet50(weights="ResNet50_Weights.DEFAULT").eval()
  fd = FeatureDescriptor(net)
  img = ds[0][0]
  actual = fd.extract_features(img, 'color_moment')[:300]

  assert torch.allclose(expected, actual)

