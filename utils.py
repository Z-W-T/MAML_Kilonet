import re

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import deque, Mapping
import math
import os
from thop import profile
from scipy.interpolate import RegularGridInterpolator as rgi
from renderer import VolumeRender


# Only this function had to be changed to account for multi networks
# (weight tensors have aditionally a network dimension)
def _calculate_fan_in_and_fan_out(tensor):
    fan_in = tensor.size(-1)
    fan_out = tensor.size(-2)
    return fan_in, fan_out


def calculate_gain(nonlinearity, param=None):
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


# All of the above functions are copy pasted from PyTorch's codebase.
# This is nessecary because of the adapted fan in computation
def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_uniform(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def kaiming_normal(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)


def xavier_uniform(tensor, gain=1.):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-a, a)


def xavier_normal(tensor, gain=1.):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    with torch.no_grad():
        return tensor.normal_(0., std)


def get_random_points_inside_domain(num_points, domain_min, domain_max):
    x = np.random.uniform(domain_min[0], domain_max[0], size=(num_points,))
    y = np.random.uniform(domain_min[1], domain_max[1], size=(num_points,))
    z = np.random.uniform(domain_min[2], domain_max[2], size=(num_points,))
    return np.column_stack((x, y, z))


def get_random_directions(num_samples):
    random_directions = np.random.randn(num_samples, 3)
    random_directions /= np.linalg.norm(random_directions, axis=1).reshape(-1, 1)
    return random_directions


def extract_domain_boxes_from_tree(root_node):
    nodes_to_process = deque([root_node])
    boxes = []
    while nodes_to_process:
        node = nodes_to_process.popleft()
        if hasattr(node, 'leq_child'):
            nodes_to_process.append(node.leq_child)
            nodes_to_process.append(node.gt_child)
        else:
            boxes.append([node.domain_min, node.domain_max])

    return boxes


def index_to_domain_xyz(index, domain_min, domain_max, res):
    z_index = index // (res[0]*res[1])
    y_index = index % (res[0]*res[1]) // res[1]
    x_index = index % (res[0]*res[1]) % res[1]
    z = z_index * (domain_max[2]-domain_min[2]) / res[2]
    y = y_index * (domain_max[1]-domain_min[1]) / res[1]
    x = x_index * (domain_max[0]-domain_min[0]) / res[0]
    return np.array([x,y,z])


def index_to_domain_xyz_index(index, res):
    z_index = index // (res[0]*res[1])
    y_index = index % (res[0]*res[1]) // res[0]
    x_index = index % (res[0]*res[1]) % res[0]
    return np.array([x_index, y_index, z_index])


# res表示网格 w h d
def discretize_domain_to_grid(domain_min, domain_max, res, point):
    domain = domain_max - domain_min
    grid_h = domain / res
    coord = torch.floor(torch.div(point-domain_min, grid_h, rounding_mode=None))
    return coord[:, 2]*res[1]*res[0]+coord[:, 1]*res[0]+coord[:, 0]


# 返回每个格子的domain
def discretize_domain_to_aabb(domain_min, domain_max, res):
    x = torch.linspace(domain_min[0], domain_max[0], res[0]+1)
    y = torch.linspace(domain_min[1], domain_max[1], res[1]+1)
    z = torch.linspace(domain_min[2], domain_max[2], res[2]+1)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
    domain_min_grid = torch.stack((grid_x[:-1, :-1, :-1], grid_y[:-1, :-1, :-1], grid_z[:-1, :-1, :-1]), -1)
    domain_max_grid = torch.stack((grid_x[1:, 1:, 1:], grid_y[1:, 1:, 1:], grid_z[1:, 1:, 1:]), -1)
    return torch.stack((domain_min_grid, domain_max_grid), -2)


def normalize_point_in_grid(x, domain_min, domain_max, res):
    domain = domain_max - domain_min
    domain_size = domain / res
    index = torch.floor(torch.div(x-domain_min, domain_size, rounding_mode=None))
    coord = x - domain_min - index * domain_size
    return (coord/domain_size)*2.0-1.0


def vec3_len(res):
    return res[2]*res[1]*res[0]


def vec3f(x, y=None, z=None):
    if y is None:
        return torch.tensor([x, x, x])
    else:
        return torch.tensor([x, y, z])


def get_device(device_gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = device_gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def div_round_up(val, divisor):
    return (val + divisor - 1) / divisor


def next_multiple(val, divisor):
    return div_round_up(val, divisor) * divisor


def compute_flops_params(model, input_tensor):
    return profile(model, (input_tensor, ))


def get_index_from_xyz(x, y, z, res):
    return z*res[0]*res[1]+y*res[0]+x


def create_samples(N=256, max_batch=32768, offset=None, scale=None):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    return samples


def normalize(x, full_normalize=False):
    """
        Normalize input to lie between 0, 1.
        Inputs:
            x: Input signal
            fullnormalize: If True, normalize such that minimum is 0 and
                maximum is 1. Else, normalize such that maximum is 1 alone.
        Outputs:
            xnormalized: Normalized x.
    """

    if x.sum() == 0:
        return x

    xmax = x.max()

    if full_normalize:
        xmin = x.min()
    else:
        xmin = 0

    xnormalized = (x - xmin) / (xmax - xmin)

    return xnormalized


def boxify(im, topleft, boxsize, color=[1, 1, 1], width=2):
    """
        Generate a box around a region.
    """
    h, w = topleft
    dh, dw = boxsize

    im[h:h + dh + 1, w:w + width, :] = color
    im[h:h + width, w:w + dh + width, :] = color
    im[h:h + dh + 1, w + dw:w + dw + width, :] = color
    im[h + dh:h + dh + width, w:w + dh + width, :] = color

    return im


def get_coords(H, W, T=None):
    """
        Get 2D/3D coordinates
    """
    if T is None:
        X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
        coords = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
    else:
        X, Y, Z = np.meshgrid(np.linspace(-1, 1, W),
                              np.linspace(-1, 1, H),
                              np.linspace(-1, 1, T))
        coords = np.hstack((X.reshape(-1, 1),
                            Y.reshape(-1, 1),
                            Z.reshape(-1, 1)))

    return torch.tensor(coords.astype(np.float32)).cuda()


def get_scheduler(scheduler_type, optimizer, args):
    """
        Get a scheduler

        Inputs:
            scheduler_type: 'none', 'step', 'exponential', 'cosine'
            optimizer: One of torch.optim optimizers
            args: Namspace containing arguments relevant to each optimizer

        Outputs:
            scheduler: A torch learning rate scheduler
    """
    scheduler = None
    if scheduler_type == 'none':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=args.epochs)
    elif scheduler_type == 'step':
        # Compute gamma
        gamma = pow(10, -1 / (args.epochs / args.step_size))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=args.step_size,
                                                    gamma=gamma)
    elif scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           gamma=args.gamma)

    return scheduler


def generate_shuffle_number(num):
    return np.random.permutation(num)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_device(model):
    return next(model.parameters()).device


# pts 是 三维数组的array
def interp3(x, y, z, V, pts):
    my_interpolating_function = rgi((x, y, z), V)
    Vi = my_interpolating_function(np.array(pts))
    return Vi


def add_noise(volume, max_shift):
    """
        Uniformly jitter the values at each coordinate

        Inputs:
            volume: HxWxT binary volume
            max_shift: Maximum allowable jitter at each pixel

        Outputs:
            volume_noisy: Noisy volume
    """
    batch_size = int(50e7)
    H, W, T = volume.shape

    x = np.linspace(-1, 1, H)
    y = np.linspace(-1, 1, W)
    z = np.linspace(-1, 1, T)

    X, Y, Z = np.meshgrid(x, y, z)

    Xn = np.clip(X + (2 * np.random.rand(H, W, T) - 1) * max_shift / H, -1, 1)
    Yn = np.clip(Y + (2 * np.random.rand(H, W, T) - 1) * max_shift / W, -1, 1)
    Zn = np.clip(Z + (2 * np.random.rand(H, W, T) - 1) * max_shift / T, -1, 1)

    func = rgi((x, y, z), volume, method='nearest')

    coords = np.hstack((Xn.reshape(-1, 1), Yn.reshape(-1, 1), Zn.reshape(-1, 1)))
    volume_noisy = np.zeros(H * W * T, dtype=np.float32)
    for idx in range(0, coords.shape[0], batch_size):
        idx2 = min(idx + batch_size, H * W * T - 1)
        volume_noisy[idx:idx2] = func(coords[idx:idx2, :])

    volume_noisy = np.transpose(volume_noisy.reshape(H, W, T), [1, 0, 2])
    volume_noisy[volume_noisy <= 0.5] = 0
    volume_noisy[volume_noisy > 0.5] = 1

    return volume_noisy


def get_grid_xyz(minlim, maxlim, cube_res):
    x = np.linspace(minlim[0], maxlim[0], cube_res[0])
    y = np.linspace(minlim[1], maxlim[1], cube_res[1])
    z = np.linspace(minlim[2], maxlim[2], cube_res[2])
    return x, y, z


def get_query_coords(minlim, maxlim, cube_res):
    """
        Get regular coordinates for querying the block implicit representation
    """
    x, y, z = get_grid_xyz(minlim, maxlim, cube_res)

    X, Y, Z = np.meshgrid(x, y, z)
    coords_gen = np.hstack((X.reshape(-1, 1),
                            Y.reshape(-1, 1),
                            Z.reshape(-1, 1)))

    return coords_gen


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


def dict_to_gpu(ob):
    if isinstance(ob, Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    else:
        return ob.cuda()


def get_coords(imsize, ksize, coordstype, unfold):
    """
        Generate coordinates for MINER training

        Inputs:
            imsize: (H, W) image size
            ksize: Kernel size
            coordstype: 'global' or 'local'
            unfold: Unfold operator
    """
    ndim = len(imsize)
    if ndim == 2:
        H, W = imsize
        nchunks = int(H * W / (ksize ** ndim))
        # Create inputs
        if coordstype == 'global':
            X, Y = torch.meshgrid(torch.linspace(-1, 1, W),
                                  torch.linspace(-1, 1, H))
            coords = torch.cat((X[None, None, ...], Y[None, None, ...]), 0)
            coords_chunked = unfold(coords).permute(2, 1, 0)
        elif coordstype == 'local':
            Xsub, Ysub = torch.meshgrid(torch.linspace(-1, 1, ksize),
                                        torch.linspace(-1, 1, ksize))
            coords_sub = torch.cat((Xsub[None, None, ...],
                                    Ysub[None, None, ...]), 0)
            coords_chunked_sub = unfold(coords_sub).permute(2, 1, 0)
            coords_chunked = torch.repeat_interleave(coords_chunked_sub,
                                                     nchunks, 0)
        else:
            raise AttributeError('Coordinate type not understood')
    else:
        H, W, T = imsize
        nchunks = int(H * W * T / (ksize ** ndim))
        # Create inputs
        if coordstype == 'global':
            X, Y, Z = torch.meshgrid(torch.linspace(-1, 1, W),
                                     torch.linspace(-1, 1, H),
                                     torch.linspace(-1, 1, T))
            coords = torch.cat((X[None, None, ...],
                                Y[None, None, ...],
                                Z[None, None, ...]), 0)
            coords_chunked = unfold(coords).permute(2, 1, 0)
        elif coordstype == 'local':
            Xsub, Ysub, Zsub = torch.meshgrid(torch.linspace(-1, 1, ksize),
                                              torch.linspace(-1, 1, ksize),
                                              torch.linspace(-1, 1, ksize))
            coords_sub = torch.cat((Xsub[None, None, ...],
                                    Ysub[None, None, ...],
                                    Zsub[None, None, ...]), 0)
            coords_chunked_sub = unfold(coords_sub).permute(2, 1, 0)
            coords_chunked = torch.repeat_interleave(coords_chunked_sub,
                                                     nchunks, 0)
        else:
            raise AttributeError('Coordinate type not understood')

    return coords_chunked


def blocks_to_volume(temp, res, block_size):
    [w, h, d] = block_size
    [w_, h_, d_] = temp.size()
    res = res.reshape((-1, w, h, d))
    for i in range(res.shape[0]):
        x,y,z = index_to_domain_xyz_index(i, vec3f(w_//w, h_//h, d_//d))
        temp[x*w:(x+1)*w, y*h:(y+1)*h, z*d:(z+1)*d] = res[i, ...]


# 输出template对应的volume
def template_to_volume(template, data_block_size):
    with torch.no_grad():
        [model, network_query_fn] = template
        pos = get_query_coords(vec3f(-1), vec3f(1), data_block_size)
        pos = torch.Tensor(pos).to(torch.float32).to(get_model_device(model))
        res = network_query_fn(pos, model)
        res = res.cpu().detach().numpy()
    return res


def multi_modules_to_volume(multi_modules, data_block_size, volume_res):
    with torch.no_grad():
        pos = get_query_coords(vec3f(-1), vec3f(1), data_block_size)
        pos = np.expand_dims(pos, axis=0)
        pos = np.repeat(pos, multi_modules.num_networks, axis=0)
        pos = torch.Tensor(pos).to(torch.float32).to(get_model_device(multi_modules.multi_MLP[-1][-1]))
        res = multi_modules(pos)
        temp = torch.ones(volume_res[0], volume_res[1], volume_res[2]). \
            to(torch.float32).to(get_model_device(multi_modules.multi_MLP[-1][-1]))
        blocks_to_volume(temp, res, data_block_size)
        res = temp.cpu().numpy()
    return res


def print_error(value):
    print("error: ", value)


def repeat_coordinates(coordinates, batch_size):
    """Repeats the coordinate tensor to create a batch of coordinates.
    Args:
        coordinates (torch.Tensor): Shape (*spatial_shape, len(spatial_shape)).
        batch_size (int): If not zero, repeats the coordinate tensor to create
            a batch of coordinates.
    """
    if batch_size:
        ones_like_shape = (1,) * coordinates.ndim
        return coordinates.unsqueeze(0).repeat(batch_size, *ones_like_shape)
    else:
        return coordinates


def shape2coordinates(spatial_shape: torch.Size, batch_size: int = 0):
    """Converts a shape tuple to a tensor of coordinates.
    Args:
        spatial_shape (tuple of ints): Tuple describing shape of data. For
            example (height, width) or (depth, height, width).
        batch_size (int): If not zero, repeats the coordinate tensor to create
            a batch of coordinates.
    Notes:
        The coordinate tensor will have coordinates lying in [0, 1] regardless
        of the input shape. Be careful if you have inputs that have very non
        square shapes, e.g. (4, 128) as each coordinate grid might then need to
        be scaled differently.
    """
    coords = []
    for i in range(len(spatial_shape)):
        coords.append(torch.linspace(0.0, 1.0, spatial_shape[i]))
    # Tensor will have shape (*spatial_shape, len(spatial_shape))
    return torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)


def deg_to_rad(degrees):
    return torch.pi * degrees / 180.0


def rad_to_deg(radians):
    return 180.0 * radians / torch.pi


def shape2spherical_coordinates(spatial_shape):
    """Returns spherical coordinates on a uniform latitude and longitude grid.
    Args:
        spatial_shape (tuple of int): Tuple (num_lats, num_lons) containing
            number of latitudes and longitudes in grid.
    """
    num_lats, num_lons = spatial_shape
    # Uniformly spaced latitudes and longitudes corresponding to ERA5 grids
    latitude = torch.linspace(90.0, -90.0, num_lats)
    longitude = torch.linspace(0.0, 360.0 - (360.0 / num_lons), num_lons)
    # Create a grid of latitude and longitude values (num_lats, num_lons)
    longitude_grid, latitude_grid = torch.meshgrid(longitude, latitude, indexing="xy")
    # Create coordinate tensor
    # Spherical coordinates have 3 dimensions
    coordinates = torch.zeros(latitude_grid.shape + (3,))
    long_rad = deg_to_rad(longitude_grid)
    lat_rad = deg_to_rad(latitude_grid)
    coordinates[..., 0] = torch.cos(lat_rad) * torch.cos(long_rad)
    coordinates[..., 1] = torch.cos(lat_rad) * torch.sin(long_rad)
    coordinates[..., 2] = torch.sin(lat_rad)
    return


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def vtk_draw_blocks(block_arrays, off_screen=False, file_name=None):
    paths = []
    arrs = []
    for array in block_arrays:
        arrs.append(array.v)
        paths.append(None)
    res = block_arrays[-1].res
    ren = VolumeRender(paths, arrs, res, len(block_arrays))
    ren.render(off_screen, file_name)
    del ren


def vtk_draw_templates(templates, block_size, off_screen=False, file_name=None):
    paths = []
    arrs = []
    for template in templates:
        paths.append(None)
        volume = template_to_volume(template, block_size)
        arrs.append(volume)
    # render
    ren = VolumeRender(paths, arrs, block_size, len(templates))
    ren.render(off_screen, file_name)
    del ren


def vtk_draw_multi_modules(multi_modules, block_size, volume_res, off_screen=False, file_name=None):
    arrs = []
    paths = []
    res = multi_modules_to_volume(multi_modules, volume_res=volume_res, data_block_size=block_size)
    arrs.append(res)
    paths.append(None)
    ren = VolumeRender(paths, arrs, volume_res, 1)
    ren.render(off_screen, file_name, view_blocks=1)


# volume data 绘制
def vtk_draw_single_volume(volume, off_screen=False, file_name=None):
    size = volume.res
    data = volume.v
    arrs = []
    paths = [None]
    arrs.append(data)
    ren = VolumeRender(paths, arrs, size, 1)
    ren.render(off_screen, file_name, view_blocks=1)


def save_txt(filename, losses):
    losses = np.array(losses)
    np.savetxt(filename, losses, delimiter=' ')


def read_dat(filename):
    data = np.fromfile(filename, dtype=np.float32)
    return data


def create_path(path):
    os.makedirs(path, exist_ok=True)


# 单例测试
def main():
    pass


if __name__ == '__main__':
    main()

