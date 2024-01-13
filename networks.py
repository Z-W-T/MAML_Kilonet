from siren import *
from utils import *


class Embedder:
    def __init__(self, **kwargs):
        self.embed_fns = None
        self.out_dim = None
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self, **kwargs):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        # 59 60
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


# FCN
class MLP(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, skips=[4], output_ch=1):
        super(MLP, self).__init__()
        self.skips = skips
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.pts_linear = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])
        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, vars=None):
        if vars is None:
            input_pts = x
            h = input_pts
            for i, l in enumerate(self.pts_linear):
                h = self.pts_linear[i](h)

                # relu激活函数
                h = F.relu(h)
                if i in self.skips:
                    h = torch.cat([input_pts, h], -1)

            outputs = self.output_linear(h)

            return outputs
        # 渐变更新 用在元学习
        else:
            # pts_linear = nn.ModuleList([nn.Linear(self.input_ch, self.W)] + [nn.Linear(self.W, self.W)
            #             if i not in self.skips else nn.Linear(self.W + self.input_ch, self.W)
            #                                                                  for i in range(self.D - 1)])\
            #     .to(vars[-1].device)
            # output_linear = nn.Linear(self.W, self.output_ch).to(vars[-1].device)
            # 手动改参数
            input_pts = x
            h = input_pts
            idx = 0
            for i, l in enumerate(self.pts_linear):
                w, b = vars[idx], vars[idx + 1]
                h = F.linear(h, w, b)
                # relu激活函数
                h = F.relu(h)
                if i in self.skips:
                    h = torch.cat([input_pts, h], -1)
                idx += 2
            w, b = vars[idx], vars[idx + 1]
            outputs = F.linear(h, w, b)

            return outputs

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                tanh_gain = nn.init.calculate_gain('tanh')
                nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)

    def clone_weights(self):
        return [w.clone().detach() for w in self.parameters()]


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


def batchify(fn, chunk, vars=None):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk], vars) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


# 输入的向量是(batch, w*h, input_ch) 或者 (batch_size, input_ch)
def run_network(inputs, fn, embed_fn, netchunk=1024*64, vars=None):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)
    outputs_flat = batchify(fn, netchunk, vars)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def run_modulated_network(inputs, fn, latent_vector, embed_fn, vars=None):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)
    outputs_flat = fn.modulated_forward(embedded, latent_vector)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def create_net(args):
    # 创建网络的部分
    if args.model == 'mlp_relu':
        model, network_query_fn = create_mlp_maml(args)
    elif args.model == 'siren':
        model, network_query_fn = create_siren_maml(args)
    elif args.model == 'film_siren':
        model, network_query_fn = create_film_siren_maml(args)
    else:
        raise NotImplementedError
    return [model, network_query_fn]


def create_mlp(args):
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    output_ch = args.output_ch
    skips = []
    device = get_device(args.GPU)
    model = MLP(D=args.netdepth, W=args.netwidth, input_ch=input_ch, output_ch=output_ch, skips=skips).to(device)
    grad_vars = list(model.parameters())
    network_query_fn = lambda inputs, network_fn, vars=None: run_network(inputs, network_fn, embed_fn, args.netchunk,
                                                                         vars)
    # create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ###################

    # Load checkpoints

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    # print("Found ckpts", ckpts)

    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])

    return start, network_query_fn, grad_vars, optimizer, model


def create_mlp_maml(args):
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    output_ch = args.output_ch
    skips = []
    device = get_device(args.GPU)
    model = MLP(D=args.netdepth, W=args.netwidth, input_ch=input_ch, output_ch=output_ch, skips=skips).to(device)

    # 不需要优化器
    network_query_fn = lambda inputs, network_fn, vars=None: run_network(inputs, network_fn, embed_fn, args.netchunk,
                                                                         vars)
    return model, network_query_fn


def create_siren(args):
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    output_ch = args.output_ch
    skips = []
    device = get_device(args.GPU)
    model = Siren(D=args.netdepth, W=args.netwidth, input_ch=input_ch, output_ch=output_ch, skips=skips,
                  nonlinearity='sine', use_bias=True).to(device)
    grad_vars = list(model.parameters())
    network_query_fn = lambda inputs, network_fn, vars=None: run_network(inputs, network_fn, embed_fn, args.netchunk,
                                                                         vars)
    # create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ###################

    # Load checkpoints

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    # print("Found ckpts", ckpts)

    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])

    return start, network_query_fn, grad_vars, optimizer, model


def create_siren_maml(args):
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    output_ch = args.output_ch
    # without skips
    skips = []
    device = get_device(args.GPU)
    model = Siren(D=args.netdepth, W=args.netwidth, input_ch=input_ch, output_ch=output_ch, skips=skips,
                  nonlinearity='sine', use_bias=True).to(device)
    network_query_fn = lambda inputs, network_fn, vars=None: run_network(inputs, network_fn, embed_fn, args.netchunk,
                                                                         vars)

    return model, network_query_fn


def create_film_siren(args):
    embed_fn, input_ch = get_embedder(args.multires, 1)
    output_ch = args.output_ch
    skips = []
    device = get_device(args.GPU)
    model = FilmSiren(D=args.netdepth, W=args.netwidth, input_ch=input_ch, output_ch=output_ch,
                      skips=skips, w0=args.w0).to(device)
    grad_vars = list(model.parameters())
    network_query_fn = lambda inputs, network_fn, vars=None: run_network(inputs, network_fn, embed_fn, args.netchunk,
                                                                         vars)
    # create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ###################

    # Load checkpoints

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    # print("Found ckpts", ckpts)

    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])

    return start, network_query_fn, grad_vars, optimizer, model


def create_film_siren_maml(args):
    embed_fn, input_ch = get_embedder(args.multires, 1)
    output_ch = args.output_ch
    # without skips
    skips = []
    device = get_device(args.GPU)
    model = FilmSiren(D=args.netdepth, W=args.netwidth, input_ch=input_ch, output_ch=output_ch,
                      skips=skips, w0=args.w0).to(device)
    network_query_fn = lambda inputs, network_fn, vars=None: run_network(inputs, network_fn, embed_fn, args.netchunk,
                                                                         vars)

    return model, network_query_fn

def create_modulated_siren_maml(args):
    embed_fn, input_ch = get_embedder(args.multires, 1)
    output_ch = args.output_ch
    # without skips
    skips = []
    device = get_device(args.GPU)
    model = ModulatedSiren(D=args.netdepth, W=args.netwidth, input_ch=input_ch, output_ch=output_ch,
                      w0=args.w0, latent_dim=args.latent_dim, use_latent=True).to(device)
    network_query_fn = lambda inputs, latent_vector, network_fn, vars=None: run_modulated_network(inputs, network_fn, latent_vector, embed_fn,
                                                                         vars)

    return model, network_query_fn

def create_transformer_modulated_siren_maml(args):
    embed_fn, input_ch = get_embedder(args.multires, 1)
    # without skips
    skips = []
    device = get_device(args.GPU)
    model = TransformerModulatedSiren(D=args.netdepth, W=args.netwidth, input_ch=args.input_ch, output_ch=args.output_ch,
                      w0=args.w0, latent_dim=args.latent_dim, use_latent=True, input_transformer_depth=args.input_transformer_depth).to(device)
    network_query_fn = lambda inputs, latent_vector, network_fn, vars=None: run_modulated_network(inputs, network_fn, latent_vector, embed_fn,
                                                                         vars)

    return model, network_query_fn


def reset_parameters(weights, bias):
    torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
    bound = 1 / math.sqrt(fan_in)
    torch.nn.init.uniform_(bias, -bound, bound)


class AdaptiveLinearWithChannel(nn.Module):
    """
        Implementation from https://github.com/pytorch/pytorch/issues/36591

        Evaluate only selective channels
    """

    def __init__(self, input_size, output_size, channel_size):
        super(AdaptiveLinearWithChannel, self).__init__()

        # initialize weights
        self.weight = torch.nn.Parameter(torch.zeros(channel_size,
                                                     input_size,
                                                     output_size))
        self.bias = torch.nn.Parameter(torch.zeros(channel_size,
                                                   1,
                                                   output_size))

        # change weights to kaiming
        reset_parameters(self.weight, self.bias)

    def forward(self, x, indices):
        # mul_output = torch.bmm(x, self.weight[indices, ...])
        return torch.bmm(x, self.weight[indices, ...]) + self.bias[indices, ...]


class AdaptiveMultiReLULayer(nn.Module):
    """
        Implements ReLU activations with multiple channel input.
    """

    def __init__(self, in_features, out_features, n_channels):
        super().__init__()
        self.in_features = in_features
        self.linear = AdaptiveLinearWithChannel(in_features,
                                                out_features,
                                                n_channels)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, input, indices):
        return self.relu(self.linear(input, indices))


# nn.Sequential handle multiple input
class MultiSequential(nn.Sequential):
    """
        https://github.com/pytorch/pytorch/issues/19808#
    """
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input


class AdaptiveMultiMLP(nn.Module):
    """
        多个MLP同时训练
    """
    def __init__(self, in_features, hidden_features, hidden_layers,
                 out_features, n_channels):
        super().__init__()

        self.nonlin = AdaptiveMultiReLULayer
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, n_channels))
        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features,
                                        hidden_features,
                                        n_channels))

        self.net.append(self.nonlin(hidden_features, out_features,
                                    n_channels))

        self.net = nn.ModuleList(self.net)

    def forward(self, inp, indices):
        output = inp[indices, ...]

        for mod in self.net:
            output = mod(output, indices)
        return output


# 单例测试
def main():
    # net = AdaptiveMultiMLP(3, 64, 2, 1, 10)
    # # channels batch in_features
    # input = torch.ones([10, 1, 3])
    # indices = torch.tensor([0,1,2,3,4,5,6,7,8,9])
    # print(net(input, indices))
    pass


if __name__ == '__main__':
    main()

