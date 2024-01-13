# SIREN(h) = sin(ω0(Wh + b))
import onnx
import torch
from utils import *


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0*x)


class Sinc(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        denom = self.w0*x
        numer = torch.cos(denom)

        return numer/(1 + abs(denom).pow(2))


class Relu(nn.Module):
    """
        Drop in replacement for SineLayer but with ReLU nonlinearity
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return F.relu(x)


class Tanh(nn.Module):
    """
        Drop in replacenment for SineLayer but with Tanh nonlinearity
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return torch.tanh(x)
    
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
    
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()
    #

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
            #
        #
    #

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    #
#

class ResidualSineLayer(nn.Module):
    def __init__(self, features, bias=True, ave_first=False, ave_second=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0

        self.features = features
        self.linear_1 = nn.Linear(features, features, bias=bias)
        self.linear_2 = nn.Linear(features, features, bias=bias)

        self.weight_1 = .5 if ave_first else 1
        self.weight_2 = .5 if ave_second else 1

        self.init_weights()
    #

    def init_weights(self):
        with torch.no_grad():
            self.linear_1.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0, 
                                           np.sqrt(6 / self.features) / self.omega_0)
            self.linear_2.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0, 
                                           np.sqrt(6 / self.features) / self.omega_0)
        #
    #
    def forward(self, input):
        sine_1 = torch.sin(self.omega_0 * self.linear_1(self.weight_1*input))
        sine_2 = torch.sin(self.omega_0 * self.linear_2(sine_1))
        return self.weight_2*(input+sine_2)


class Siren(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, skips=[4], output_ch=1, nonlinearity='sine', use_bias=True, w0=30., multires=5):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.output_ch = output_ch
        self.omega_0 = w0
        self.use_bias = use_bias
        if nonlinearity == 'sine':
            # See siren paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            self.activation_function = Sine(w0)
        elif nonlinearity == 'tanh':
            self.activation_function = Tanh
        elif nonlinearity == 'sinc':
            self.activation_function = Sinc
        else:
            self.activation_function = Relu

        embed_kwargs = {
        'include_input': True,
        'input_dims': 4,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
        self.embedder = Embedder(**embed_kwargs)
        self.pts_linear = nn.ModuleList(
            [nn.Linear(input_ch, W, bias=use_bias)] + [nn.Linear(W, W, bias=use_bias) if i not in self.skips
                                        else nn.Linear(W + input_ch, W, bias=use_bias) for i in range(D - 1)])
        self.output_linear = nn.Linear(W, output_ch, bias=use_bias)
        # 添加网络初始化，important
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            for i, linear in enumerate(self.pts_linear):
                if i == 0:
                    # 第一层
                    self.pts_linear[0].weight.uniform_(-1 / self.pts_linear[0].in_features,
                                                       1 / self.pts_linear[0].in_features)

                    if self.use_bias:
                        self.pts_linear[0].bias.uniform_(-1 / self.pts_linear[0].in_features,
                                                         1 / self.pts_linear[0].in_features)
                else:
                    linear.weight.uniform_(-np.sqrt(6 / linear.in_features) / self.omega_0,
                                           np.sqrt(6 / linear.in_features) / self.omega_0)

                    if self.use_bias:
                        linear.bias.uniform_(-np.sqrt(6 / linear.in_features) / self.omega_0,
                                             np.sqrt(6 / linear.in_features) / self.omega_0)

            self.output_linear.weight.uniform_(-np.sqrt(6 / self.W) / self.omega_0,
                                         np.sqrt(6 / self.W) / self.omega_0)

            if self.use_bias:
                self.output_linear.bias.uniform_(-np.sqrt(6 / self.W) / self.omega_0,
                                                 np.sqrt(6 / self.W) / self.omega_0)

    def forward(self, x, vars=None):
        if vars is None:
            input_pts = self.embedder(x)
            h = input_pts
            for i, l in enumerate(self.pts_linear):
                h = self.pts_linear[i](h)
                # relu激活函数
                h = self.activation_function.forward(h)
                if i in self.skips:
                    h = torch.cat([input_pts, h], -1)

            outputs = self.output_linear(h)
            outputs += 0.5
            return outputs
        # 渐变更新 用在元学习
        else:
            input_pts = self.embedder(x)
            h = input_pts
            idx = 0
            for i, l in enumerate(self.pts_linear):
                w, b = vars[idx], vars[idx + 1]
                h = F.linear(h, w, b)
                # relu激活函数
                h = self.activation_function.forward(h)
                if i in self.skips:
                    h = torch.cat([input_pts, h], -1)
                idx += 2
            w, b = vars[idx], vars[idx + 1]
            outputs = F.linear(h, w, b)
            # https://github.com/EmilienDupont/coinpp/blob/main/coinpp/models.py
            # We assume target data is in [0, 1], so adding 0.5 allows us to learn
            # zero-centered features
            outputs += 0.5
            return outputs
        
class ResidualSiren(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, skips=[4], output_ch=1, nonlinearity='sine', use_bias=True, w0=30.):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        self.output_ch = output_ch
        self.omega_0 = w0
        self.use_bias = use_bias
        if nonlinearity == 'sine':
            # See siren paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            self.activation_function = Sine(w0)
        elif nonlinearity == 'tanh':
            self.activation_function = Tanh
        elif nonlinearity == 'sinc':
            self.activation_function = Sinc
        else:
            self.activation_function = Relu

        self.pts_linear = nn.ModuleList(
            [nn.Linear(input_ch, W, bias=use_bias)] + [ResidualSineLayer(W, bias=use_bias) if i not in self.skips
                                        else nn.Linear(W + input_ch, W, bias=use_bias) for i in range(D - 1)])
        self.output_linear = nn.Linear(W, output_ch, bias=use_bias)
        # 添加网络初始化，important
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            # 第一层
            self.pts_linear[0].weight.uniform_(-1 / self.pts_linear[0].in_features,
                                                1 / self.pts_linear[0].in_features)       
            self.output_linear.weight.uniform_(-np.sqrt(6 / self.W) / self.omega_0,
                                         np.sqrt(6 / self.W) / self.omega_0)

            if self.use_bias:
                self.pts_linear[0].bias.uniform_(-1 / self.pts_linear[0].in_features,
                                                    1 / self.pts_linear[0].in_features)
                self.output_linear.bias.uniform_(-np.sqrt(6 / self.W) / self.omega_0,
                                                 np.sqrt(6 / self.W) / self.omega_0)

    def forward(self, x, vars=None):
        if vars is None:
            input_pts = x
            h = input_pts
            for i, l in enumerate(self.pts_linear):
                h = self.pts_linear[i](h)

                # relu激活函数
                h = self.activation_function.forward(h)
                if i in self.skips:
                    h = torch.cat([input_pts, h], -1)

            outputs = self.output_linear(h)
            outputs += 0.5
            return outputs
        # 渐变更新 用在元学习
        else:
            input_pts = x
            h = input_pts
            idx = 0
            for i, l in enumerate(self.pts_linear):
                w, b = vars[idx], vars[idx + 1]
                h = F.linear(h, w, b)
                # relu激活函数
                h = self.activation_function.forward(h)
                if i in self.skips:
                    h = torch.cat([input_pts, h], -1)
                idx += 2
            w, b = vars[idx], vars[idx + 1]
            outputs = F.linear(h, w, b)
            # https://github.com/EmilienDupont/coinpp/blob/main/coinpp/models.py
            # We assume target data is in [0, 1], so adding 0.5 allows us to learn
            # zero-centered features
            outputs += 0.5
            return outputs

class LatentToModulation(nn.Module):
    """Maps a latent vector to a set of modulations.
    Args:
        latent_dim (int):
        num_modulations (int):
        dim_hidden (int):
        num_layers (int):
    """

    def __init__(self, latent_dim, num_modulations, dim_hidden, num_layers):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modulations = num_modulations
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        if num_layers == 1:
            self.net = nn.Linear(latent_dim, num_modulations)
        else:
            layers = [nn.Linear(latent_dim, dim_hidden), nn.ReLU()]
            if num_layers > 2:
                for i in range(num_layers - 2):
                    layers += [nn.Linear(dim_hidden, dim_hidden), nn.ReLU()]
            layers += [nn.Linear(dim_hidden, num_modulations)]
            self.net = nn.Sequential(*layers)

    def forward(self, latent):
        return self.net(latent)

class ModulatedSiren(Siren):
    def __init__(
            self,
            D,
            W,
            input_ch,
            output_ch,
            w0=30.0,
            w0_initial=30.0,
            use_bias=True,
            modulate_scale=False,
            modulate_shift=True,
            use_latent=False,
            latent_dim=64,
            modulation_net_dim_hidden=64,
            modulation_net_num_layers=1,
    ):
        super().__init__(
            D,
            W,
            input_ch,
            skips=[],
            output_ch=output_ch,
            nonlinearity='sine',
            use_bias=use_bias,
        )
        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial

        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated

        num_modulations = W * D
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2
        self.num_modulations = num_modulations

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

    def modulated_forward(self, x, latent):
        """
        Forward pass of modulated SIREN model.
        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.
        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        h = self.embedder.embed(x)

        # Shape (batch_size, num_modulations)
        modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        mid_idx = (
            self.num_modulations // 2
            if (self.modulate_scale and self.modulate_shift)
            else 0
        )
        idx = 0
        for module in self.pts_linear:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, idx: idx + self.W].unsqueeze(1) + 1.0
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[mid_idx + idx: mid_idx + idx + self.W]
            else:
                shift = 0.0

            h = module(h)
            h = scale * h + shift  # Broadcast scale and shift across num_points
            h = self.activation_function.forward(h)  # (batch_size, num_points, dim_hidden)

            idx = idx + self.W

        # Shape (batch_size, num_points, dim_out)
        h = self.output_linear(h)
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return h
    
class ResidualModulatedSiren(ResidualSiren):
    def __init__(
            self,
            D,
            W,
            input_ch,
            output_ch,
            skips = [],
            nonlinearity='sine',
            w0=30.0,
            w0_initial=30.0,
            use_bias=True,
            modulate_scale=False,
            modulate_shift=True,
            use_latent=False,
            latent_dim=64,
            modulation_net_dim_hidden=64,
            modulation_net_num_layers=1,
    ):
        super().__init__(D, W, input_ch, skips, output_ch, nonlinearity, use_bias, w0,)

        assert modulate_scale or modulate_shift

        self.modulate_scale = modulate_scale
        self.modulate_shift = modulate_shift
        self.w0 = w0
        self.w0_initial = w0_initial

        # We modulate features at every *hidden* layer of the base network and
        # therefore have dim_hidden * (num_layers - 1) modulations, since the
        # last layer is not modulated

        num_modulations = W * D 
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2
        self.num_modulations = num_modulations

        if use_latent:
            self.modulation_net = LatentToModulation(
                latent_dim,
                num_modulations,
                modulation_net_dim_hidden,
                modulation_net_num_layers,
            )
        else:
            self.modulation_net = Bias(num_modulations)

        # Initialize scales to 1 and shifts to 0 (i.e. the identity)
        if not use_latent:
            if self.modulate_shift and self.modulate_scale:
                self.modulation_net.bias.data = torch.cat(
                    (
                        torch.ones(num_modulations // 2),
                        torch.zeros(num_modulations // 2),
                    ),
                    dim=0,
                )
            elif self.modulate_scale:
                self.modulation_net.bias.data = torch.ones(num_modulations)
            else:
                self.modulation_net.bias.data = torch.zeros(num_modulations)

    def modulated_forward(self, x, latent):
        """
        Forward pass of modulated SIREN model.
        Args:
            x (torch.Tensor): Shape (batch_size, *, dim_in), where * refers to
                any spatial dimensions, e.g. (height, width), (height * width,)
                or (depth, height, width) etc.
            latent (torch.Tensor): Shape (batch_size, latent_dim). If
                use_latent=False, then latent_dim = num_modulations.
        Returns:
            Output features of shape (batch_size, *, dim_out).
        """
        h = x
        modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        mid_idx = (
            self.num_modulations // 2
            if (self.modulate_scale and self.modulate_shift)
            else 0
        )
        idx = 0
        for module in self.pts_linear:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, idx: idx + self.W].unsqueeze(1) + 1.0
            else:
                scale = 1.0
            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[mid_idx + idx: mid_idx + idx + self.W]
            else:
                shift = 0.0

            h = module(h)
            h = scale * h + shift  # Broadcast scale and shift across num_points
            h = self.activation_function.forward(h)  # (batch_size, num_points, dim_hidden)

            idx = idx + self.W

        # Shape (batch_size, num_points, dim_out)
        h = self.output_linear(h)
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return h

class TransformerModulatedSiren(ModulatedSiren):
    def __init__(self,
            D,
            W,
            input_ch,
            output_ch,
            w0=30.0,
            w0_initial=30.0,
            use_bias=True,
            modulate_scale=False,
            modulate_shift=True,
            use_latent=False,
            latent_dim=64,
            modulation_net_dim_hidden=64,
            modulation_net_num_layers=1,
            use_input_transformer=True,
            input_transformer_depth=2,
            use_output_transformer=True,
            output_transformer_depth=2,):
        self.use_former_transformer = use_input_transformer
        self.use_latter_transformer = use_output_transformer
        self.input_transformer_depth = input_transformer_depth
        self.output_transformer_depth = output_transformer_depth

        if use_input_transformer:
            layer_size = [4,8,64,256]
        super().__init__(D, W, input_ch, output_ch, w0, w0_initial, use_bias, modulate_scale, modulate_shift, use_latent, latent_dim, modulation_net_dim_hidden, modulation_net_num_layers) 

        if use_input_transformer:
            self.input_transformer = nn.ModuleList()
            self.input_transformer.append(nn.Linear(input_ch, layer_size[0], bias=use_bias))
            for i in range(1,input_transformer_depth):
                self.input_transformer.append(nn.Linear(layer_size[i-1], layer_size[i], bias=use_bias)) 
        
        if use_output_transformer:
            self.output_transformer = nn.ModuleList([nn.Linear(W, W, bias=use_bias) if i != output_transformer_depth-1 
                                                     else nn.Linear(W , output_ch, bias=use_bias) for i in range(output_transformer_depth)])

    # update base network
    def forward(self, x, vars=None):
        if vars is None:
            input_pts = x
            h = input_pts
            for i, l in enumerate(self.pts_linear):
                h = self.pts_linear[i](h)
                h = self.activation_function.forward(h)
                if i in self.skips:
                    h = torch.cat([input_pts, h], -1)

            outputs = self.output_linear(h)
            # outputs += 0.5
            return outputs
        # 渐变更新 用在元学习
        else:
            input_pts = x
            h = input_pts
            idx = 0
            for i, l in enumerate(self.pts_linear):
                w, b = vars[idx], vars[idx + 1]
                h = F.linear(h, w, b)
                h = self.activation_function.forward(h)
                if i in self.skips:
                    h = torch.cat([input_pts, h], -1)
                idx += 2
            w, b = vars[idx], vars[idx + 1]
            outputs = F.linear(h, w, b)
            # outputs += 0.5
            return outputs

    # inner step update modulation, outer step update base network
    def modulated_forward(self, x, latent):
        input_pts = self.embedder.embed(x)
        h = input_pts

        # Shape (batch_size, num_modulations)
        modulations = self.modulation_net(latent)

        # Split modulations into shifts and scales and apply them to hidden
        # features.
        mid_idx = (
            self.num_modulations // 2
            if (self.modulate_scale and self.modulate_shift)
            else 0
        )
        idx = 0
        for module in self.pts_linear:
            if self.modulate_scale:
                # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                # modulations remain zero centered
                scale = modulations[:, idx: idx + self.W].unsqueeze(1) + 1.0
            else:
                scale = 1.0

            if self.modulate_shift:
                # Shape (batch_size, 1, dim_hidden)
                shift = modulations[mid_idx + idx: mid_idx + idx + self.W]
            else:
                shift = 0.0

            h = module(h)
            h = scale * h + shift  # Broadcast scale and shift across num_points
            h = self.activation_function.forward(h)  # (batch_size, num_points, dim_hidden)

            idx = idx + self.W

        # Shape (batch_size, num_points, dim_out)
        h = self.output_linear(h)
        # Reshape (batch_size, num_points, dim_out) -> (batch_size, *, dim_out)
        return h
    
    def transformer_forward(self, x, vars=None):
        if vars == None:
            for module in self.input_transformer:
                x = module(x)
            input_pts = self.embedder.embed(x)
            h = input_pts
            for i, l in enumerate(self.pts_linear):
                h = self.pts_linear[i](h)
                # relu激活函数
                h = self.activation_function.forward(h)
                if i in self.skips:
                    h = torch.cat([input_pts, h], -1)
            for module in self.output_transformer:
                h = module(h)
            outputs = h
        else:
            idx = 0
            for module in self.input_transformer:
                w,b = vars[idx], vars[idx+1]
                x = F.linear(x,w,b)
                idx += 2
            input_pts = self.embedder.embed(x)
            h = input_pts
            for i, l in enumerate(self.pts_linear):
                h = self.pts_linear[i](h)
                # relu激活函数
                h = self.activation_function.forward(h)
                if i in self.skips:
                    h = torch.cat([input_pts, h], -1)
            for module in self.output_transformer:
                w,b = vars[idx], vars[idx+1]
                h = F.linear(h,w,b)
                idx += 2
            outputs = h
        return outputs
    
    def MTF_forward(self, x, MTF_parameters=None):
        if MTF_parameters is None:
            for module in self.input_transformer:
                x = module(x)
                x = self.activation_function(x)
            # x = self.embedder.embed(x)

            h = x
            for i, l in enumerate(self.pts_linear):
                h = self.pts_linear[i](h)
                # relu激活函数
                h = self.activation_function.forward(h)
                if i in self.skips:
                    h = torch.cat([x, h], -1)

            for module in self.output_transformer:
                h = module(h)
            outputs = h
            return  outputs           
        else:
            # Shape (batch_size, num_modulations)
            # latent_vector = MTF_parameters[-1]
            # modulations = self.modulation_net(latent_vector)

            # Split modulations into shifts and scales and apply them to hidden
            # features.
            # mid_idx = (
            #     self.num_modulations // 2
            #     if (self.modulate_scale and self.modulate_shift)
            #     else 0
            # )

            idx = 0
            # m_idx = 0
            for i, module in enumerate(self.input_transformer):
                    w,b = MTF_parameters[idx], MTF_parameters[idx+1]
                    x = F.linear(x,w,b)
                    if i == (self.input_transformer_depth - 1):
                        x = self.activation_function(x)
                    idx += 2
            # input_pts = self.embedder.embed(x)
            h = x
            for i, module in enumerate(self.pts_linear):
                if i == 0:
                    # m_idx += self.W
                    continue
                # if self.modulate_scale:
                #     # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                #     # modulations remain zero centered
                #     scale = modulations[:, m_idx: m_idx + self.W].unsqueeze(1) + 1.0
                # else:
                #     scale = 1.0

                # if self.modulate_shift:
                #     # Shape (batch_size, 1, dim_hidden)
                #     shift = modulations[mid_idx + m_idx: mid_idx + m_idx + self.W]
                # else:
                #     shift = 0.0

                h = module(h)
                # h = scale * h + shift  # Broadcast scale and shift across num_points
                h = self.activation_function.forward(h)
                if i in self.skips:
                    h = torch.cat([x, h], -1)
                # m_idx += self.W
            for module in self.output_transformer:
                w,b = MTF_parameters[idx], MTF_parameters[idx+1]
                h = F.linear(h,w,b)
                h = F.relu(h)
                idx += 2
            outputs = h
        return outputs

class FilmSiren(nn.Module):
    def __init__(self,
                 D,
                 W,
                 input_ch,
                 output_ch,
                 skips,
                 w0=30.0,
                 use_bias=True,
                 scale=False,
                 shift=True,
                 ):
        super().__init__()
        self.siren = Siren(
            D,
            W,
            input_ch,
            skips=skips,
            output_ch=output_ch,
            nonlinearity='sine',
            use_bias=use_bias,
            w0=w0)

        assert scale or shift
        self.W = W
        num_modulations = D * W
        self.modulate_scale = scale
        self.modulate_shift = shift
        if self.modulate_scale and self.modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            num_modulations *= 2

        self.BiasNet = Bias(num_modulations)

        if self.modulate_shift and self.modulate_scale:
            self.BiasNet.bias.data = torch.cat(
                (
                    torch.ones(num_modulations // 2),
                    torch.zeros(num_modulations // 2),
                ),
                dim=0,
            )
        elif self.modulate_scale:
            self.BiasNet.bias.data = torch.ones(num_modulations)
        else:
            self.BiasNet.bias.data = torch.zeros(num_modulations)

        self.num_modulations = num_modulations

    def forward(self, x, vars=None):
        if vars is None:
            modulations = self.BiasNet(None)
            h = x
            # Split modulations into shifts and scales and apply them to hidden
            # features.
            mid_idx = (
                self.num_modulations // 2
                if (self.modulate_scale and self.modulate_shift)
                else 0
            )
            idx = 0
            for module in self.siren.pts_linear:
                if self.modulate_scale:
                    # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                    # modulations remain zero centered
                    scale = modulations[idx: idx + self.W].unsqueeze(1) + 1.0
                else:
                    scale = 1.0

                if self.modulate_shift:
                    # Shape (batch_size, 1, dim_hidden)
                    shift = modulations[mid_idx + idx: mid_idx + idx + self.W].unsqueeze(1)
                else:
                    shift = 0.0

                h = module(h)
                h = scale * h + shift.T  # Broadcast scale and shift across num_points
                h = self.siren.activation_function.forward(h)  # (batch_size, num_points, dim_hidden)

                idx = idx + self.W

            h = self.siren.output_linear(h)
            h += 0.5
            return h
        else:
            # Split modulations into shifts and scales and apply them to hidden
            # features.
            mid_idx = (
                self.num_modulations // 2
                if (self.modulate_scale and self.modulate_shift)
                else 0
            )

            modulations = vars[-1]
            idx = 0
            idx_net = 0
            input_pts = x
            h = input_pts
            for i, module in enumerate(self.siren.pts_linear):
                if self.modulate_scale:
                    # Shape (batch_size, 1, dim_hidden). Note that we add 1 so
                    # modulations remain zero centered
                    scale = modulations[idx: idx + self.W].unsqueeze(1) + 1.0
                else:
                    scale = 1.0

                if self.modulate_shift:
                    # Shape (batch_size, 1, dim_hidden)
                    shift = modulations[mid_idx + idx: mid_idx + idx + self.W].unsqueeze(1)
                else:
                    shift = 0.0

                h = module(h)
                h = scale * h + shift.T  # Broadcast scale and shift across num_points
                h = self.siren.activation_function.forward(h)  # (batch_size, num_points, dim_hidden)
                idx_net += 4
               
            h = self.siren.output_linear(h)
            h += 0.5

            return h


class Bias(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(size), requires_grad=True)
        # Add latent_dim attribute for compatibility with LatentToModulation model
        self.latent_dim = size

    # 两用
    def forward(self, x):
        if x is None:
            return self.bias
        else:
            return x + self.bias


class OurNet(nn.Module):
    def __init__(self, H, W, ):
        super().__init__()

    def forward(self, x):
        # 整个网络的forward
        pass

    def modulated_forward(self, x):
        # 只有modulated部分有梯度
        pass


if __name__ == "__main__":
    # dim_in, dim_hidden, dim_out, num_layers = 2, 5, 3, 4
    # batch_size, latent_dim = 3, 40
    # model = ModulatedSiren(
    #     num_layers,
    #     dim_hidden,
    #     dim_in,
    #     dim_out,
    #     modulate_scale=True,
    #     use_latent=False,
    #     latent_dim=latent_dim,
    # )
    # print(model)
    # latent = torch.rand(batch_size, latent_dim)
    # x = torch.rand(batch_size, 5, 5, 2)
    # out = model(x)
    # out = model.modulated_forward(x, latent)
    # print(out.shape)
    net = FilmSiren(2, 64, 3, 1, skips=[])
    parm = list(net.parameters())
    print(parm)
