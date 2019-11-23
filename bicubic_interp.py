import torch
import numpy as np

def bicubic_interp_2d(input_, new_size):
  """
  Args :
    input_ : Input tensor. Its shape should be
        [batch_size, height, width, channel].
        In this implementation, the shape should be fixed for speed.
    new_size : The output size [new_height, new_width]
  ref : http://blog.demofox.org/2015/08/15/resizing-images-with-bicubic-interpolation/
  """

  shape = input_.size()
  batch_size = shape[0]
  height  = shape[1]
  width   = shape[2]
  channel = shape[3]
 
  def _hermite(A, B, C, D, t):
    a = A * -0.5 + B * 1.5 + C * -1.5 + D * 0.5
    b = A + B * -2.5 + C * 2.0 + D * -0.5
    c = A * -0.5 + C * 0.5
    d = B
    t = t.type_as(A)

    return a*t*t*t + b*t*t + c*t + d

  def _get_grid_array(n_i, y_i, x_i, c_i):
    n, y, x, c = np.meshgrid(n_i, y_i, x_i, c_i, indexing='ij')
    n = np.expand_dims(n, axis=4)
    y = np.expand_dims(y, axis=4)
    x = np.expand_dims(x, axis=4)
    c = np.expand_dims(c, axis=4)
    return np.concatenate([n,y,x,c], axis=4)

  def _get_frac_array(x_d, y_d, n, c):
    x = x_d.shape[0]
    y = y_d.shape[0]
    x_t = x_d.reshape([1, 1, -1, 1])
    y_t = y_d.reshape([1, -1, 1, 1])
    y_t = np.tile(y_t, (n,1,x,c))
    x_t = np.tile(x_t, (n,y,1,c))
    return x_t, y_t

  def _get_index_tensor(grid, x, y):
    new_grid = np.array(grid)
    
    grid_y = grid[:,:,:,:,1] + y
    grid_x = grid[:,:,:,:,2] + x
    grid_y = np.clip(grid_y, 0, height-1)
    grid_x = np.clip(grid_x, 0, width-1)

    new_grid[:,:,:,:,1] = grid_y
    new_grid[:,:,:,:,2] = grid_x

    return torch.LongTensor(new_grid)

  new_height = new_size[0]
  new_width  = new_size[1]

  n_i = np.arange(batch_size)
  c_i = np.arange(channel)

  y_f = np.linspace(0., height-1, new_height)
  y_i = y_f.astype(np.int32)
  y_d = y_f - np.floor(y_f)

  x_f = np.linspace(0., width-1, new_width)
  x_i = x_f.astype(np.int32)
  x_d = x_f - np.floor(x_f)

  grid = _get_grid_array(n_i, y_i, x_i, c_i)
  x_t, y_t = _get_frac_array(x_d, y_d, batch_size, channel)

  i_00 = _get_index_tensor(grid, -1, -1)
  i_10 = _get_index_tensor(grid, +0, -1)
  i_20 = _get_index_tensor(grid, +1, -1)
  i_30 = _get_index_tensor(grid, +2, -1)
      
  i_01 = _get_index_tensor(grid, -1, +0)
  i_11 = _get_index_tensor(grid, +0, +0)
  i_21 = _get_index_tensor(grid, +1, +0)
  i_31 = _get_index_tensor(grid, +2, +0)
      
  i_02 = _get_index_tensor(grid, -1, +1)
  i_12 = _get_index_tensor(grid, +0, +1)
  i_22 = _get_index_tensor(grid, +1, +1)
  i_32 = _get_index_tensor(grid, +2, +1)
      
  i_03 = _get_index_tensor(grid, -1, +2)
  i_13 = _get_index_tensor(grid, +0, +2)
  i_23 = _get_index_tensor(grid, +1, +2)
  i_33 = _get_index_tensor(grid, +2, +2)

  p_00 = input_[i_00[:, :, :, :, 0], i_00[:, :, :, :, 1], i_00[:, :, :, :, 2], i_00[:, :, :, :, 3]]
  p_10 = input_[i_10[:, :, :, :, 0], i_10[:, :, :, :, 1], i_10[:, :, :, :, 2], i_10[:, :, :, :, 3]]
  p_20 = input_[i_20[:, :, :, :, 0], i_20[:, :, :, :, 1], i_20[:, :, :, :, 2], i_20[:, :, :, :, 3]]
  p_30 = input_[i_30[:, :, :, :, 0], i_30[:, :, :, :, 1], i_30[:, :, :, :, 2], i_30[:, :, :, :, 3]]

  p_01 = input_[i_01[:, :, :, :, 0], i_01[:, :, :, :, 1], i_01[:, :, :, :, 2], i_01[:, :, :, :, 3]]
  p_11 = input_[i_11[:, :, :, :, 0], i_11[:, :, :, :, 1], i_11[:, :, :, :, 2], i_11[:, :, :, :, 3]]
  p_21 = input_[i_21[:, :, :, :, 0], i_21[:, :, :, :, 1], i_21[:, :, :, :, 2], i_21[:, :, :, :, 3]]
  p_31 = input_[i_31[:, :, :, :, 0], i_31[:, :, :, :, 1], i_31[:, :, :, :, 2], i_31[:, :, :, :, 3]]

  p_02 = input_[i_02[:, :, :, :, 0], i_02[:, :, :, :, 1], i_02[:, :, :, :, 2], i_02[:, :, :, :, 3]]
  p_12 = input_[i_12[:, :, :, :, 0], i_12[:, :, :, :, 1], i_12[:, :, :, :, 2], i_12[:, :, :, :, 3]]
  p_22 = input_[i_22[:, :, :, :, 0], i_22[:, :, :, :, 1], i_22[:, :, :, :, 2], i_22[:, :, :, :, 3]]
  p_32 = input_[i_32[:, :, :, :, 0], i_32[:, :, :, :, 1], i_32[:, :, :, :, 2], i_32[:, :, :, :, 3]]

  p_03 = input_[i_03[:, :, :, :, 0], i_03[:, :, :, :, 1], i_03[:, :, :, :, 2], i_03[:, :, :, :, 3]]
  p_13 = input_[i_13[:, :, :, :, 0], i_13[:, :, :, :, 1], i_13[:, :, :, :, 2], i_13[:, :, :, :, 3]]
  p_23 = input_[i_23[:, :, :, :, 0], i_23[:, :, :, :, 1], i_23[:, :, :, :, 2], i_23[:, :, :, :, 3]]
  p_33 = input_[i_33[:, :, :, :, 0], i_33[:, :, :, :, 1], i_33[:, :, :, :, 2], i_33[:, :, :, :, 3]]

  col0 = _hermite(p_00, p_10, p_20, p_30, torch.from_numpy(x_t))
  col1 = _hermite(p_01, p_11, p_21, p_31, torch.from_numpy(x_t))
  col2 = _hermite(p_02, p_12, p_22, p_32, torch.from_numpy(x_t))
  col3 = _hermite(p_03, p_13, p_23, p_33, torch.from_numpy(x_t))
  value = _hermite(col0, col1, col2, col3, torch.from_numpy(y_t))
  
  return value


# Future : bicubic_interp_3d
