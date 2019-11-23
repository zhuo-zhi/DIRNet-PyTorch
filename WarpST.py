import torch
from bicubic_interp import bicubic_interp_2d

def diff_clamp(tensor, min, max):
    tensor = tensor - tensor.detach() + torch.max(tensor, torch.full_like(tensor, min)).detach()
    tensor = tensor - tensor.detach() + torch.min(tensor, torch.full_like(tensor, max)).detach()
    return tensor

def WarpST(U, V, out_size, **kwargs):
    """Deformable Transformer Layer with bicubic interpolation
    U : tf.float, [num_batch, height, width, num_channels].
        Input tensor to warp
    V : tf.float, [num_batch, height, width, 2]
        Warp map. It is interpolated to out_size.
    out_size: a tuple of two ints
        The size of the output of the network (height, width)
    ----------
    References :
      https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py
    """

    def _repeat(x, n_repeats):
        rep = torch.ones((n_repeats, )).unsqueeze(1).transpose(1, 0)
        rep = rep.long()
        x = torch.matmul(x.reshape(-1, 1), rep.type_as(x))
        return x.reshape(-1)

    def _interpolate(im, x, y, out_size):
        # constants
        num_batch = im.size(0)
        height = im.size(1)
        width = im.size(2)
        channels = im.size(3)

        x = x.float()
        y = y.float()
        height_f = float(height)
        width_f = float(width)
        out_height = out_size[0]
        out_width = out_size[1]
        zero = torch.zeros([]).long()
        max_y = im.size(1) - 1
        max_x = im.size(2) - 1

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0)*(width_f) / 2.0
        y = (y + 1.0)*(height_f) / 2.0

        # do sampling
        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        # x0 = torch.clamp(x0, zero.item(), max_x)
        # x1 = torch.clamp(x1, zero.item(), max_x)
        # y0 = torch.clamp(y0, zero.item(), max_y)
        # y1 = torch.clamp(y1, zero.item(), max_y)

        x0 = diff_clamp(x0, zero.item(), max_x)
        x1 = diff_clamp(x1, zero.item(), max_x)
        y0 = diff_clamp(y0, zero.item(), max_y)
        y1 = diff_clamp(y1, zero.item(), max_y)

        dim2 = width
        dim1 = width*height
        base = _repeat(torch.range(0, num_batch - 1, 1)*dim1, out_height*out_width)
        base_y0 = base.type_as(y0) + y0*dim2
        base_y1 = base.type_as(y1) + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = im.reshape(-1, channels)
        im_flat = im_flat.float()
        Ia = im_flat[idx_a]
        Ib = im_flat[idx_b]
        Ic = im_flat[idx_c]
        Id = im_flat[idx_d]

        # and finally calculate interpolated values
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()
        wa = ((x1_f-x) * (y1_f-y)).unsqueeze(1)
        wb = ((x1_f-x) * (y-y0_f)).unsqueeze(1)
        wc = ((x-x0_f) * (y1_f-y)).unsqueeze(1)
        wd = ((x-x0_f) * (y-y0_f)).unsqueeze(1)
        output = wa*Ia + wb*Ib + wc*Ic + wd*Id
        return output

    def _meshgrid(height, width):
        # This should be equivalent to:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        x_t = torch.matmul(torch.ones((height, 1)),
                        torch.linspace(-1.0, 1.0, width).unsqueeze(1).transpose(1, 0))
        y_t = torch.matmul(torch.linspace(-1.0, 1.0, height).unsqueeze(1), torch.ones((1, width)))

        x_t_flat = x_t.reshape(1, -1)
        y_t_flat = y_t.reshape(1, -1)

        grid = torch.cat((x_t_flat, y_t_flat), dim = 0)
        return grid

    def _transform(V, U, out_size):
        num_batch = U.size(0)
        height = U.size(2)
        width = U.size(3)
        num_channels = U.size(1)

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        # height_f = tf.cast(height, 'float32')
        height_f = float(height)
        # width_f = tf.cast(width, 'float32')
        width_f = float(width)

        out_height = out_size[0]
        out_width = out_size[1]
        grid = _meshgrid(out_height, out_width)     # [2, h*w]

        grid = grid.reshape(-1)               # [2*h*w]
        grid = grid.unsqueeze(0).repeat(num_batch, 1)      # [n, 2*h*w]
        grid = grid.reshape(num_batch, 2, -1)       # [n, 2, h*w]


        # Set source position (x+vx, y+vy)^T
        V = bicubic_interp_2d(V.permute(0, 2, 3, 1), out_size)
        V = V.permute(0, 3, 1, 2)           # [n, 2, h, w]
        V = V.reshape(num_batch, 2, -1)       # [n, 2, h*w]
        T_g = V + grid                       # [n, 2, h*w]

        x_s = T_g[:, 0, :]
        y_s = T_g[:, 1, :]
        x_s_flat = x_s.reshape(-1)
        y_s_flat = y_s.reshape(-1)

        input_transformed = _interpolate(
            U.permute(0, 2, 3, 1), x_s_flat, y_s_flat, out_size)

        output = input_transformed.reshape(num_batch, out_height, out_width, num_channels)
        return output

    output = _transform(V, U, out_size)
    return output
