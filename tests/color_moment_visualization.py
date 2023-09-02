import matplotlib.pyplot as plt
import numpy as np

# ==== Required Libs ====
import torch
import torchvision

from feature_descriptor import FeatureDescriptor

# Custom variables
torch.set_grad_enabled(False)
torch.hub.set_dir('/Users/rohan/3_Resources/ai_models/torch_hub')

ds = torchvision.datasets.Caltech101('/Users/rohan/3_Resources/ai_datasets/caltech_101', download=False)
img = ds[0][0]

fd = FeatureDescriptor(
  net=torchvision.models.resnet50(weights="ResNet50_Weights.DEFAULT").eval()
)


def plot_color_moments(color_moments):
  y = color_moments.numpy()
  mean, std, skew = y[:300], y[300:600], y[600:]
  x = np.arange(len(mean))

  fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
  axes[0].plot(x, mean, color='r')
  axes[0].set_ylabel('Mean')
  axes[0].set_title('Mean Moment')

  axes[1].plot(x, std, color='g')
  axes[1].set_ylabel('Std. Dev')
  axes[1].set_title('Standard Deviation Moment')

  axes[2].plot(x, skew, color='b')
  axes[2].set_ylabel('Skewness')
  axes[2].set_title('Skewness Moment')

# test that 2 different plots are shown
plot_color_moments(
  fd.extract_features(img, descriptor='color_moment')
)
plot_color_moments(
  fd.extract_features(ds[len(ds)-983][0], descriptor='color_moment')
)
plt.tight_layout()
plt.show()
