import numpy as np

class MNISTDataHandler(object):
  """
    Members :
      is_train - Options for sampling
      path - MNIST data path
      data - a list of np.array w/ shape [batch_size, 28, 28, 1]
  """
  def __init__(self, digit):
    self.data = digit

  def sample_pair(self, batch_size, label=None):
    label = np.random.randint(10) if label is None else label
    images = self.data[label]
    
    choice1 = np.random.choice(images.shape[0], batch_size)
    choice2 = np.random.choice(images.shape[0], batch_size)
    x = images[choice1]
    y = images[choice2]

    return x, y
