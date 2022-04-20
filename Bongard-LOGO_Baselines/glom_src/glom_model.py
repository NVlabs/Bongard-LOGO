import time
import numpy as np


import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .top_down_bottom_up import *
from .attention import *


class GLOM_encoder(nn.Module):
  def __init__(self, in_channels=1, num_timesteps=10, image_dim=(28,28), patch_dim=3, number_of_embeddings=5, vector_dim=256, device="cpu"):
    super(GLOM_encoder, self).__init__()
    self.num_timesteps = num_timesteps
    self.image_dim = image_dim
    self.patch_dim = patch_dim
    self.number_of_embeddings = number_of_embeddings
    self.vector_dim = vector_dim
    self.device = device

    pad_needed = (patch_dim - image_dim[0] % patch_dim)

    if pad_needed % 2 == 0:
      self.pad = (int(pad_needed/2), int(pad_needed/2), int(pad_needed/2), int(pad_needed/2))

    else:
      self.pad = (pad_needed, pad_needed-1, pad_needed, pad_needed-1)

    # self.pad = (pad_needed, pad_needed, pad_needed, pad_needed)
    # self.pad = (1,0,1,0)
    # print(self.pad)

    self.num_patches = int((image_dim[0] + pad_needed)/patch_dim)**2
    # num_patches = 121

    self.cnn = CNN(in_channels, patch_dim, vector_dim)
    self.top_down_net = TopDownNet(self.num_patches, vector_dim)
    self.bottom_up_net = BottomUpNet(self.num_patches, vector_dim)
    self.attention_net = SelfAttention(self.num_patches, self.num_patches, 1, groups=int(np.sqrt(self.num_patches)))

  def get_num_patches(self):
    return self.num_patches

  def forward(self, x):
    out = F.pad(x, self.pad)
    patches = []
    for i in range(0, self.image_dim[0], self.patch_dim):
      for j in range(0, self.image_dim[0], self.patch_dim):
        patch = out[:, :, i:i+3, j:j+3].unsqueeze(1)
        patches.append(patch)

    patches = torch.cat(patches, dim = 1)
    batch_size = patches.shape[0]
    num_patches = patches.shape[1]
    out = patches.reshape((batch_size*num_patches, patches.shape[2], patches.shape[3], patches.shape[4]))
    out = self.cnn(out)
    out = out.reshape((batch_size, num_patches, out.shape[1]))

    # initialize columns of each level at timestep = level with this CNN output 
    # columns = out # (B, P, L, V)
    columns = torch.stack([out]*self.number_of_embeddings, dim=2) # (B, P, L, V)
    # print(columns.shape)
    for t in range(1, self.num_timesteps):
      temp = []
      for i in range(self.number_of_embeddings):
        if i == 0:
          t1 = columns[:, :, i, :].unsqueeze(2)

          t2_i_1 = columns[:, :, i+1, :].reshape((batch_size, num_patches*1*self.vector_dim)) 
          t2_i_2 = torch.stack([torch.FloatTensor(list(range(0,num_patches)))]*batch_size, dim=0).to(self.device)
          
          t2_i = torch.cat([t2_i_1, t2_i_2], 1)

          t2 = self.top_down_net(t2_i)
          t3 = self.attention_net(t1)

          c = t1 + 0.33 * t2 + 0.33 * t3

        elif i == self.number_of_embeddings - 1:
          t1 = columns[:, :, i, :].unsqueeze(2)

          t2_i = columns[:, :, i-1, :].reshape((batch_size, num_patches*1*self.vector_dim))

          t2 = self.bottom_up_net(t2_i)
          t3 = self.attention_net(t1)

          c = t1 + 0.33 * t2 + 0.33 * t3

        else:
          t1 = columns[:, :, i, :].unsqueeze(2)

          t2_i_1 = columns[:, :, i+1, :].reshape((batch_size, num_patches*1*self.vector_dim))
          t2_i_2 = torch.stack([torch.FloatTensor(list(range(0, num_patches)))]*batch_size).to(self.device)

          t2_i = torch.cat([t2_i_1, t2_i_2], 1).to(self.device)

          t3_i = columns[:, :, i-1, :].reshape((batch_size, num_patches*1*self.vector_dim))

          t2 = self.top_down_net(t2_i)
          t3 = self.bottom_up_net(t3_i)
          t4 = self.attention_net(t1)

          c = t1 + 0.33 * t2 + 0.33 * t3 + 0.33 * t4

        temp.append(c)
      columns = torch.stack(temp, dim=2).squeeze(3)     # (B, P, L, V)
      result = columns[:, :, 4, :]      # (B, P, V)
      # result.unsqueeze(2)           # (B, P, 1, V)
      result = result.mean(1)     # (B, V)
    return result

class GLOM_decoder(nn.Module):
  def __init__(self, num_patches, num_classes=10, vector_dim=64):
    super(GLOM_decoder, self).__init__()
    self.layer1 = nn.Sequential(nn.Linear(num_patches*vector_dim, 256),
                                nn.ReLU(),
                                nn.Linear(256, 256),
                                nn.ReLU(),
                                nn.Linear(256, num_classes),
                                nn.Softmax(dim=1))
  def forward(self, x):
    out = nn.Flatten()(x)
    out = self.layer1(out)
    return out

class GLOM_reconst_decoder(nn.Module):
    def __init__(self, num_patches, vector_dim, img_dim=(1, 28, 28)):
        super(GLOM_reconst_decoder, self).__init__()
        self.img_dim = img_dim
        h = img_dim[0] * img_dim[1] * img_dim[2]
        self.layer1 = nn.Sequential(nn.Linear(num_patches*vector_dim, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, h),)
    def forward(self, x):
        out = nn.Flatten()(x)
        out = self.layer1(out)
        out = out.reshape((x.shape[0], self.img_dim[0], self.img_dim[1], self.img_dim[2]))
        return out


class FeedForwardNet_CNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNet_CNN, self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
                                    nn.GELU(),
                                    nn.Dropout(0.2),
									)

    def forward(self, x):
        return self.layer(x)

class FeedBackNet_CNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedBackNet_CNN, self).__init__()
        self.layer = nn.Sequential(
								#    nn.Upsample(2, mode='bilinear'),
                  nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2),
                  nn.GELU(),
                  nn.Dropout(0.2),)

    def forward(self, x):
        return self.layer(x)

class ClassNet_CNN(nn.Module):
    def __init__(self, in_chans=96, classes=10):
        super(ClassNet_CNN, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_chans, classes),
                                   nn.Softmax(dim=1))

    def forward(self, x):
        x = nn.AdaptiveMaxPool2d(1)(x)
        x = nn.Flatten()(x)
        return self.layer(x)



class GLOM_CNN(nn.Module):
	def __init__(self, layers, alpha=0.33, beta=0.33, gamma=0.33, img_dim=(3, 256, 256), num_classes=10, num_timesteps=10):
		super(GLOM_CNN, self).__init__()
		self.layers = layers
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma
		self.img_dim = img_dim
		self.num_classes = num_classes
		self.num_timesteps = num_timesteps

		self.forward_layers = []
		self.backward_layers = []
		self.attention_layers = []
		self.forward_layers.append(FeedForwardNet_CNN(1, layers[0]))
		self.backward_layers.append(FeedBackNet_CNN(layers[0], 1))
		self.attention_layers.append(SelfAttention(layers[0], layers[0]))	
		for i in range(1, len(layers)):
			self.forward_layers.append(FeedForwardNet_CNN(layers[i-1], layers[i]))
			self.backward_layers.append(FeedBackNet_CNN(layers[i], layers[i-1]))
			self.attention_layers.append(SelfAttention(layers[i], layers[i]))

		self.fc = ClassNet_CNN(layers[-1], num_classes)
		self.forward_layers = nn.ModuleList(self.forward_layers)
		self.backward_layers = nn.ModuleList(self.backward_layers)
		self.attention_layers = nn.ModuleList(self.attention_layers)
		
		self.error_term = nn.MSELoss()

	def forward(self, x, criterion=nn.CrossEntropyLoss(), mode='forward'):
		if mode == 'forward':
			for i in range(len(self.forward_layers)):
				x = self.forward_layers[i](x)
			# x = self.fc(x)
			# loss = criterion(x, y)
			loss = 0.0
			return x, loss
		elif mode == 'full':
			fs, bs, errors, errors_all = [], [], [], []
			out = self.forward_layers[0](x)
			fs.append(out)
			for i in range(1, len(self.forward_layers)): 
				out = self.forward_layers[i](out)
				b = self.backward_layers[i](out)
				e = self.error_term(fs[i-1], b)
				
				fs.append(out)
				bs.append(b)
				errors.append(e)
				errors_all.append(e)

			for t in range(1, self.num_timesteps):
				out = x
				ff = self.forward_layers[0](out)
				fs[0] = ff 
				for i in range(1,len(self.forward_layers)):
					ff = self.forward_layers[i](fs[i-1])

					fb = torch.zeros_like(ff)

					if i < 4:
						fb = bs[i]

					fs[i] = (self.alpha*ff + self.beta*fb + (1 - self.alpha - self.beta)*self.attention_layers[i](fs[i])- self.gamma * errors[i-1])
					bs[i-1] = self.backward_layers[i](fs[i])
					errors[i-1] = self.error_term(bs[i-1], fs[i-1])
					errors_all.append(errors[i-1])
			
			out = x
			for i in range(len(self.forward_layers)):
				out = self.forward_layers[i](out)
				# print(out.shape)
			# out = self.fc(out)
			# loss = criterion(out, y)

			out = nn.Flatten()(out)
			loss = 0.0	
			for l in errors_all:
				loss+=l

			return out, loss

		else:
			raise ValueError('mode must be either forward or backward')