import torch
import torchvision

TORCH_HUB = './models/'
torch.set_grad_enabled(False)
torch.hub.set_dir(TORCH_HUB)

DATA_DIR = './data/caltech101'
DATASET = torchvision.datasets.Caltech101(DATA_DIR, download=False)

MODEL_NAME = 'ResNet50_Weights.DEFAULT'
RESNET_MODEL = torchvision.models.resnet50(weights=MODEL_NAME).eval()
