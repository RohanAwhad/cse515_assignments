import cv2
import numpy as np
import PIL
import scipy.stats as stats
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from typing import Union, Callable, Tuple, Dict, Any

class FeatureStore:
  def __init__(self, load_data: Callable[[str], Tuple[Any, ...]]):
    self.features = {}
    self.load_data = load_data

    self.similarity_measures = {
      'color_moment': 'pearson_coefficient',
      'hog': 'intersection_similarity',
      'resnet_layer3': 'cosine_similarity',
      'resnet_avgpool': 'cosine_similarity',
      'resnet_fc': 'manhattan_distance',
      'resnet_softmax': 'kl_divergence',
    }

  def __getitem__(self, key: str) -> Dict[str, Tuple[str, Dict[int, Tuple[int, int]], torch.Tensor]]:
    if key not in self.features:
      # load features from disk
      idx, feats = self.load_data(key)
      if isinstance(feats, np.ndarray): feats = torch.from_numpy(feats)
      self.features[key] = (self.similarity_measures[key], idx, feats)

    return self.features[key]



class SaveOutput:
  def __init__(self): self.output = []
  def __call__(self, module, inp, out): self.output.append(out)
  def __getitem__(self, index): return self.output[index]
  def clear(self): self.output = []

class FeatureDescriptor:
  def __init__(self, net):
    self.net = net
    self.edge_filter = torch.tensor([[-1, 0, 1]])

    self.save_out = SaveOutput()
    self.hook_handles = [
      self.net.get_submodule('layer3').register_forward_hook(self.save_out),
      self.net.get_submodule('avgpool').register_forward_hook(self.save_out),
      self.net.get_submodule('fc').register_forward_hook(self.save_out),
    ]

  def extract_features(self, img: Union[PIL.Image.Image, np.array, torch.Tensor], descriptor: str):
    if descriptor == 'hog': return self.extract_hog_features(img)
    elif descriptor == 'color_moment': return self.extract_color_moments(img)
    elif descriptor == 'resnet_layer3': return self.extract_resnet_features(img)[0]
    elif descriptor == 'resnet_avgpool': return self.extract_resnet_features(img)[1]
    elif descriptor == 'resnet_fc': return self.extract_resnet_features(img)[2]
    elif descriptor == 'resnet_softmax': return (self.extract_resnet_features(img)[2]).softmax(-1)
    else: raise ValueError(f'{descriptor} is either not a valid descriptor or not implemented')


  def extract_color_moments(self, img):
    if img.mode != 'RGB': raise ValueError(f'expected "RGB" image. got {img.mode}')
    img = TF.resize(img, size=(100, 300))
    img = TF.to_tensor(img).unsqueeze(0)

    *_, H, W = img.size()
    kernel_size = (H//10, W//10)
    img = F.unfold(img, kernel_size, stride=kernel_size).squeeze().view(3, 300, 100)

    mean_mmt = img.mean(-2)
    std_mmt = img.std(-2)

    # calculating skew
    n_pixels = img.size()[1]
    skew_mmt = img - mean_mmt.unsqueeze(1)
    skew_mmt = (skew_mmt ** 3).sum(1) / n_pixels
    skew_mmt = skew_mmt.abs().pow(1/3) * (skew_mmt/skew_mmt.abs())
    skew_mmt[skew_mmt.isnan()] = 0

    return torch.cat((mean_mmt.flatten(), std_mmt.flatten(), skew_mmt.flatten()), axis=0)


  def extract_hog_features(self, img):
    '''
    - Resize image to 100x300
    - convert to grayscale
    - compute X and Y gradients with [-1, 0, 1] filters
    - compute magnitude and direction of gradients
    - bin direction into 9 bins each having a range of 40 degrees
    - compute signed magnitude weighted histogram of gradients with image split into 10x10 grid
    '''
    img = TF.resize(img, size=(100, 300))
    img = TF.to_grayscale(img)
    img = TF.to_tensor(img).transpose(-1, -2).squeeze()#.transpose(0, -1)  # to_tensor scales values between [0, 1]

    H, W = img.size()
    #img_tensor = torch.tensor(img, dtype=torch.float).expand(1, 1, H, W)
    img_tensor = img.expand(1, 1, H, W)

    # calc grad_x and grad_y
    # using torch functional package's unfold method to get the grids
    # and while calculating gradients, we pad the image on all four
    # four sides. We remove the extra padding by slicing
    # rows from grad_x and cols from grad_y
    img_unfld_x = F.unfold(img_tensor, (1, 3), stride=1, padding=1).squeeze().T
    grad_x = (img_unfld_x * self.edge_filter).sum(-1).view(H+2, W)[1:-1]
    img_unfld_y = F.unfold(img_tensor, (3, 1), stride=1, padding=1).squeeze().T
    grad_y = (img_unfld_y * self.edge_filter).sum(-1).view(H, W+2)[:, 1:-1]

    # normalization messess up the directions. Gives NaN values because of divide by zero
    #grad_x = (grad_x + 1) / 2
    #grad_y = (grad_y + 1) / 2

    # magnitude & direction
    grad_mag = (grad_x**2 + grad_y**2)**0.5
    grad_direct = (grad_y / grad_x).atan().rad2deg()
    grad_direct_bin = ((grad_direct + 360) % 360) // 40  # to get rid of -ve values we add and take remainder

    # signed-mag weighted histograms
    kernel_stride_size = (H//10, W//10)
    grad_mag_unfld = F.unfold(grad_mag.expand(1, 1, H, W),
      kernel_stride_size, stride=kernel_stride_size).squeeze().T
    grad_direct_bin_unfld = F.unfold(grad_direct_bin.expand(1, 1, H, W),
      kernel_stride_size, stride=kernel_stride_size).squeeze().T

    return torch.cat([
      (grad_mag_unfld * (grad_direct_bin_unfld == i)).sum(-1) for i in range(9)
    ])

  def extract_resnet_features(self, img):
    if img.mode != 'RGB': raise ValueError(f'expected "RGB" image. got {img.mode}')
    x = TF.resize(img, size=(224, 224))
    x = TF.to_tensor(x).unsqueeze(0)
    #x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # mean and std taken from https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html
    self.net(x)

    layer3_features = self.save_out[0].squeeze()
    layer3_features = layer3_features.view(len(layer3_features), -1).mean(-1).flatten()
    avgpool_features = self.save_out[1].view(-1, 2).mean(-1).flatten()
    fc_features = self.save_out[2].flatten()
    self.save_out.clear()
    return layer3_features, avgpool_features, fc_features
