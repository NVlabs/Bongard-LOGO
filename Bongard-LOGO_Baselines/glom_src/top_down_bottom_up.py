import time
import numpy as np


import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class TopDownNet(nn.Module):
  def __init__(self, number_of_patches, vector_dim):
    super(TopDownNet, self).__init__()
    self.number_of_patches = number_of_patches
    self.vector_dim = vector_dim
    self.layer1 = nn.Sequential(nn.Linear(number_of_patches*vector_dim+number_of_patches, 256),
                                nn.GELU(),)
    self.layer2 = nn.Sequential(nn.Linear(256, 256),
                                nn.GELU(),)
    self.layer3 = nn.Sequential(nn.Linear(256, vector_dim*number_of_patches))

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = torch.reshape(x, (x.shape[0], self.number_of_patches, 1, self.vector_dim))
    return x

class BottomUpNet(nn.Module):
  def __init__(self, number_of_patches, vector_dim):
    super(BottomUpNet, self).__init__()
    self.number_of_patches = number_of_patches
    self.vector_dim = vector_dim
    self.layer1 = nn.Sequential(nn.Linear(number_of_patches*vector_dim, 256),
                                nn.GELU(),)
    self.layer2 = nn.Sequential(nn.Linear(256, 256),
                                nn.GELU(),)
    self.layer3 = nn.Sequential(nn.Linear(256, vector_dim*number_of_patches))

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = torch.reshape(x, (x.shape[0], self.number_of_patches, 1, self.vector_dim))
    return x

class CNN(nn.Module):
  def __init__(self, in_channels=1, image_dim=3, vector_dim=256):
    super(CNN, self).__init__()
    self.vector_dim = vector_dim
    self.layer1 = nn.Sequential(nn.Conv2d(in_channels, 32, 1), 
                                nn.GELU(),)
    self.layer2 = nn.Sequential(nn.Conv2d(32, 64, 1),
                                # nn.Conv2d(64, 64, 1),
                                nn.GELU(),)
    self.layer3 = nn.Sequential(nn.Linear(image_dim*image_dim*64, 256),
                                nn.GELU(),
                                nn.Linear(256, vector_dim))

    # self.backbone = torchvision.models.resnet18(pretrained=True)

    # if in_channels == 1:
    #   self.backbone.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)

    # self.backbone.fc = nn.Linear(512, vector_dim)

    
  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = nn.Flatten()(out)
    out = self.layer3(out)
    # out = self.backbone(x)
    return out