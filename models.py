# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from WarpST import WarpST
from ops import *
import matplotlib.pyplot as plt
import numpy as np

def plot(im):
  im = np.array(im.tolist())
  plt.imshow(im, cmap='gray', vmin=0, vmax=1)
  plt.show()
  return None

class CNN(nn.Module):

  def __init__(self):
    super().__init__()

    self.enc_x = nn.Sequential(
      conv2d(2, 64, 3, 1, 1, bias=False), # 64 x 28 x 28
      nn.BatchNorm2d(64, momentum=0.9, eps=1e-5),
      nn.ELU(),
      nn.AvgPool2d(2, 2, 0), # 64 x 14 x 14

      conv2d(64, 128, 3, 1, 1, bias=False),
      nn.BatchNorm2d(128, momentum=0.9, eps=1e-5),
      nn.ELU(),
      conv2d(128, 128, 3, 1, 1, bias=False),
      nn.BatchNorm2d(128, momentum=0.9, eps=1e-5),
      nn.ELU(),
      nn.AvgPool2d(2, 2, 0),  # 64 x 7 x 7

      conv2d(128, 2, 3, 1, 1), # 2 x 7 x 7
      nn.Tanh()
    )

  def forward(self, x):
    x = self.enc_x(x)
    return x


class DIRNet(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.vCNN = CNN()
    self.config = config

  def forward(self, x, y):
    xy = torch.cat((x, y), dim = 1)
    v = self.vCNN(xy)
    # print(str(v.max().item()) + ' '+ str(v.min().item()))
    z = WarpST(x, v, self.config.im_size)
    # loss = ncc(y, z)
    z = z.permute(0, 3, 1, 2)
    loss = mse(y, z)
    return z, loss

  def deploy(self, dir_path, x, y):
    with torch.no_grad():
      z, _ = self.forward(x, y)
      for i in range(z.shape[0]):
        save_image_with_scale(dir_path+"/{:02d}_x.tif".format(i+1), x.permute(0, 2, 3, 1)[i,:,:,0].numpy())
        save_image_with_scale(dir_path+"/{:02d}_y.tif".format(i+1), y.permute(0, 2, 3, 1)[i,:,:,0].numpy())
        save_image_with_scale(dir_path+"/{:02d}_z.tif".format(i+1), z.permute(0, 2, 3, 1)[i,:,:,0].numpy())
