import torch
from models import DIRNet
from config import get_config
from data import MNISTDataHandler
from ops import mkdir
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as tf
from torch.optim.lr_scheduler import StepLR
import numpy as np


torch.manual_seed(0)
train_batch = 60000
test_batch = 10000


def main():
  config = get_config(is_train=True)
  mkdir(config.tmp_dir)
  mkdir(config.ckpt_dir)

  model = DIRNet(config)

  transform = tf.Compose([tf.Resize([16, 16]), tf.ToTensor()])
  train_loader = DataLoader(MNIST('mnist', train=True, download=True, transform=transform), batch_size=train_batch)
  test_loader = DataLoader(MNIST('mnist', train=False, download=True, transform=transform), batch_size=test_batch)
  for batch, (data, label) in enumerate(train_loader):
    if batch == 0:
      num_images = 16000
      # num_images = 300
      digit_0 = data.index_select(0, label.eq(0).nonzero().squeeze())[:3000]
      digit_1 = data.index_select(0, label.eq(1).nonzero().squeeze())[:3000]
      digit_2 = data.index_select(0, label.eq(2).nonzero().squeeze())[:3000]
      digit_3 = data.index_select(0, label.eq(3).nonzero().squeeze())[:3000]
      digit_4 = data.index_select(0, label.eq(4).nonzero().squeeze())[:3000]
      digit_5 = data.index_select(0, label.eq(5).nonzero().squeeze())[:3000]
      digit_6 = data.index_select(0, label.eq(6).nonzero().squeeze())[:3000]
      digit_7 = data.index_select(0, label.eq(7).nonzero().squeeze())[:3000]
      digit_8 = data.index_select(0, label.eq(8).nonzero().squeeze())[:3000]
      digit_9 = data.index_select(0, label.eq(9).nonzero().squeeze())[:3000]

  digit = torch.stack([digit_0, digit_1, digit_2, digit_3, digit_4, digit_5, digit_6, digit_7,
                       digit_8, digit_9], dim=0)

  for batch, (data, label) in enumerate(test_loader):
    if batch == 0:
      num_images = 16000
      # num_images = 300
      digit_0_t = data.index_select(0, label.eq(0).nonzero().squeeze())[:500]
      digit_1_t = data.index_select(0, label.eq(1).nonzero().squeeze())[:500]
      digit_2_t = data.index_select(0, label.eq(2).nonzero().squeeze())[:500]
      digit_3_t = data.index_select(0, label.eq(3).nonzero().squeeze())[:500]
      digit_4_t = data.index_select(0, label.eq(4).nonzero().squeeze())[:500]
      digit_5_t = data.index_select(0, label.eq(5).nonzero().squeeze())[:500]
      digit_6_t = data.index_select(0, label.eq(6).nonzero().squeeze())[:500]
      digit_7_t = data.index_select(0, label.eq(7).nonzero().squeeze())[:500]
      digit_8_t = data.index_select(0, label.eq(8).nonzero().squeeze())[:500]
      digit_9_t = data.index_select(0, label.eq(9).nonzero().squeeze())[:500]

  digit_t = torch.stack([digit_0_t, digit_1_t, digit_2_t, digit_3_t, digit_4_t, digit_5_t, digit_6_t, digit_7_t,
                       digit_8_t, digit_9_t], dim=0)

  optim = torch.optim.Adam(model.parameters(), lr = config.lr)
  scheduler = StepLR(optim, step_size=200, gamma=0.5)

  train_pr = MNISTDataHandler(digit)
  test_pr = MNISTDataHandler(digit_t)

  total_loss = 0
  for i in range(config.iteration):

    batch_x, batch_y = train_pr.sample_pair(config.batch_size)
    optim.zero_grad()
    _, loss = model(batch_x, batch_y)
    loss.backward()
    optim.step()
    scheduler.step()
    total_loss += loss

    if (i+1) % 100 == 0:
      print("iter {:>6d} : {}".format(i + 1, total_loss))
      total_loss = 0
      batch_x, batch_y = test_pr.sample_pair(config.batch_size)
      model.deploy(config.tmp_dir, batch_x, batch_y)
      # reg.save(config.ckpt_dir)

if __name__ == "__main__":
  main()
