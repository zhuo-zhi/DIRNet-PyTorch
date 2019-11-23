import torch
import torch.nn as nn
import os
import skimage.io
import numpy as np
import torch.nn.functional as F

# def ncc(x, y):
#   mean_x = tf.reduce_mean(x, [1,2,3], keep_dims=True)
#   mean_y = tf.reduce_mean(y, [1,2,3], keep_dims=True)
#   mean_x2 = tf.reduce_mean(tf.square(x), [1,2,3], keep_dims=True)
#   mean_y2 = tf.reduce_mean(tf.square(y), [1,2,3], keep_dims=True)
#   stddev_x = tf.reduce_sum(tf.sqrt(
#     mean_x2 - tf.square(mean_x)), [1,2,3], keep_dims=True)
#   stddev_y = tf.reduce_sum(tf.sqrt(
#     mean_y2 - tf.square(mean_y)), [1,2,3], keep_dims=True)
#   return tf.reduce_mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))

def mse(x, y):
  batch_size = x.size(0)
  return ((x - y) ** 2).sum() / batch_size

def mkdir(dir_path):
  try :
    os.makedirs(dir_path)
  except: pass 

def save_image_with_scale(path, arr):
  arr = np.clip(arr, 0., 1.)
  arr = arr * 255.
  arr = arr.astype(np.uint8)
  skimage.io.imsave(path, arr)



def conv2d(in_channels, out_channels, kernel_size, stride=1,
           padding=0, dilation=1, groups=1,
           bias=True, padding_mode='zeros',
           gain=1., bias_init=0.):
  m = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                padding, dilation, groups, bias, padding_mode)

  nn.init.orthogonal_(m.weight, gain)
  if bias:
    nn.init.constant_(m.bias, bias_init)

  return m



class Conv2dBlock(nn.Module):

  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
    super().__init__()

    self.m = conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

  def forward(self, x):
    x = F.elu(self.m(x))
    return F.layer_norm(x, x.size()[1:])


