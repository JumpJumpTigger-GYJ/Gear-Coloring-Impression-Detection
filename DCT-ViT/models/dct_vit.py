import torch
from torch import nn, pixel_shuffle
from timm.models.layers import DropPath, trunc_normal_

class HWCConv(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.padding = nn.ConstantPad2d(padding, 0.)
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, stride=stride, groups=groups)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(self.padding(x))
        x = x.permute(0, 2, 3, 1)
        return x

    def flops(self):
        #flops for each output pixel
        flops = 0.
        flops += self.in_chans // self.groups * (self.kernel_size ** 2)
        flops *= self.out_chans
        return flops

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def flops(self):
        #flops for each pixel
        flops = 0.
        flops += self.in_features * self.hidden_features
        flops += self.hidden_features * self.out_features
        return flops

class GroupNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=dim, num_channels=dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads):
        super().__init__()
        self.dim = dim
        self.win_size = win_size if isinstance(win_size, list) else [win_size]
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Sequential(
            HWCConv(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.Linear(dim, dim),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        shape = x.shape
        assert self.dim == shape[-1]

        x = self.qkv(x)
        out = self.window_partition(x)

        for i, qkv in enumerate(out):
            q, k, v = qkv[0], qkv[1], qkv[2]
            out[i] = self.self_attention(q, k, v)

        x = self.window_reverse(out, shape)
        x = self.proj(x)
        return x

    def self_attention(self, q, k, v):
        scale = q.shape[-1] ** -0.5
        attn = (q * scale) @ k.transpose(-2, -1)
        attn = self.softmax(attn)
        out = attn @ v
        return out

    def window_partition(self, x):
        out = []
        B, H, W, _ = x.shape
        B_scale = self.num_heads // len(self.win_size)
        head_dim = self.dim // self.num_heads
        x = x.reshape(B, H, W, 3, self.num_heads, head_dim)
        for i, x_scale in enumerate(torch.split(x, B_scale, dim=4)):
            win_size = self.win_size[i]
            x_scale = x_scale.reshape(B, H // win_size, win_size, W // win_size, win_size, 3, B_scale, head_dim)
            x_scale = x_scale.permute(5, 0, 1, 3, 6, 7, 2, 4).reshape(3, -1, head_dim, win_size ** 2)
            out.append(x_scale)
        return out

    def window_reverse(self, out, shape):
        B, H, W, C = shape
        B_scale = self.num_heads // len(self.win_size)
        head_dim = self.dim // self.num_heads
        for i, x in enumerate(out):
            win_size = self.win_size[i]
            x = x.reshape(B, H // win_size, W // win_size, B_scale, head_dim, win_size, win_size)
            x = x.permute(0, 1, 5, 2, 6, 3, 4).reshape(B, H, W, B_scale, head_dim)
            out[i] = x
        x = torch.cat(out, dim=3)
        x = x.reshape(B, H, W, C)
        return x

    def flops(self):
        feature_size = self.win_size[-1]
        pixels = feature_size ** 2
        n_token = self.dim // self.num_heads
        B_scale = self.num_heads // len(self.win_size)
        ttl_flops = 0.
        for i, win_size in enumerate(self.win_size):
            token_dim = win_size ** 2
            n_win = (feature_size // win_size) ** 2
            flops = 0.
            flops += n_token * token_dim * n_token
            flops += n_token * n_token * token_dim
            flops *= B_scale * n_win
            ttl_flops += flops
        ttl_flops += self.dim * 3 * self.dim * pixels
        ttl_flops += self.dim * self.dim * pixels
        ttl_flops += self.proj[0].flops() * pixels
        return ttl_flops

class ChannelBlock(nn.Module):
    def __init__(self, dim, win_size, num_heads, mlp_ratio=4., norm_layer=nn.LayerNorm,
                 drop_path_rate=0.1):
        super().__init__()
        win_size = win_size if isinstance(win_size, list) else [win_size]
        self.cpe = HWCConv(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.attn = ChannelAttention(dim, win_size, num_heads)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.path_drop = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x = self.cpe(x)
        x = x + self.path_drop(self.attn(self.norm1(x)))
        x = x + self.path_drop(self.mlp(self.norm2(x)))
        return x

    def flops(self):
        feature_size = self.attn.win_size[-1]
        pixels = feature_size ** 2
        flops = 0.
        flops += self.cpe.flops() * pixels
        flops += self.attn.flops()
        flops += self.mlp.flops() * pixels
        return flops

class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads):
        super().__init__()
        self.dim = dim
        self.win_size = win_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Sequential(
            HWCConv(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.Linear(dim, dim),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        shape = x.shape
        assert self.dim == shape[-1]

        qkv = self.partition(self.qkv(x), shape)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = q.shape[-1] ** -0.5
        q = q * scale
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)

        x = self.reverse(attn @ v, shape)
        x = self.proj(x)
        return x

    def partition(self, x, shape):
        B, H, W, C = shape
        win_size = self.win_size
        num_heads = self.num_heads

        x = x.reshape(B, H // win_size, win_size, W // win_size, win_size, 3, num_heads, C // num_heads)
        x = x.permute(5, 0, 1, 3, 6, 2, 4, 7).reshape(3, -1, win_size * win_size, C // num_heads)
        return x

    def reverse(self, x, shape):
        B, H, W, C = shape
        win_size = self.win_size
        num_heads = self.num_heads

        x = x.reshape(B, H // win_size, W // win_size, num_heads, win_size, win_size, C // num_heads)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, H, W, C)
        return x

    def flops(self):
        #flops for each window
        n_token = self.win_size ** 2
        token_dim = self.dim // self.num_heads
        flops = 0.
        flops += n_token * token_dim * n_token
        flops += n_token * n_token * token_dim
        flops *= self.num_heads
        flops += self.dim * 3 * self.dim * n_token
        flops += self.dim * self.dim * n_token
        flops += self.proj[0].flops() * n_token
        return flops

class WindowBlock(nn.Module):
    def __init__(self, dim, win_size, num_heads, mlp_ratio=4., norm_layer=nn.LayerNorm, drop_path_rate=0.1):
        super().__init__()
        self.cpe = HWCConv(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.attn = WindowAttention(dim, win_size, num_heads)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.path_drop = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x = self.cpe(x)
        x = x + self.path_drop(self.attn(self.norm1(x)))
        x = x + self.path_drop(self.mlp(self.norm2(x)))
        return x

    def flops(self, feature_size):
        pixels = feature_size ** 2
        n_win = (feature_size // self.attn.win_size) ** 2
        flops = 0.
        flops += self.cpe.flops() * pixels
        flops += self.attn.flops() * n_win
        flops += self.mlp.flops() * pixels
        return flops

class BasicBlock(nn.Module):
    def __init__(self, dim, sp_win_size, sp_num_heads, ch_win_size, ch_num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, drop_path_rate=0.1):
        super().__init__()
        self.window_block = WindowBlock(dim=dim, win_size=sp_win_size, num_heads=sp_num_heads, mlp_ratio=mlp_ratio,
                                        norm_layer=norm_layer, drop_path_rate=drop_path_rate)
        self.channel_block = ChannelBlock(dim=dim, win_size=ch_win_size, num_heads=ch_num_heads, mlp_ratio=mlp_ratio,
                                          norm_layer=norm_layer, drop_path_rate=drop_path_rate)
    def forward(self, x):
        x = self.window_block(x)
        x = self.channel_block(x)
        return x

    def flops(self):
        feature_size = self.channel_block.attn.win_size[-1]
        flops = 0.
        flops += self.window_block.flops(feature_size)
        flops += self.channel_block.flops()
        return flops

class BasicLayer(nn.Module):
    def __init__(self, n_blk, dim, sp_win_size, sp_num_heads, ch_win_size, ch_num_heads,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, drop_path_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            BasicBlock(dim=dim, sp_win_size=sp_win_size, sp_num_heads=sp_num_heads,
                       ch_win_size=ch_win_size, ch_num_heads=ch_num_heads,
                       mlp_ratio=mlp_ratio, norm_layer=norm_layer,
                       drop_path_rate= drop_path_rate[i] if isinstance(drop_path_rate, list)
                                                         else drop_path_rate)
            for i in range(n_blk)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def flops(self):
        flops = 0.
        for layer in self.layers:
            flops += layer.flops()
        return flops

class DCT_ViT(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, dims=[96, 192, 384, 768],
                 depths=[1, 1, 3, 1], sp_win_size=[7, 7, 7, 7], sp_num_heads=[3, 6, 12, 24],
                 ch_win_size = [[14, 28, 56], [14, 28], [14], [7]], ch_num_heads = [3, 4, 6, 12],
                 mlp_ratio=4., norm_layer=nn.LayerNorm, drop_path_rate=0.1,
                 head_init_scale=1.):
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.dims = dims
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            HWCConv(in_chans, dims[0], kernel_size=7, stride=4, padding=3),
            norm_layer(dims[0])
        )
        self.downsample_layers.append(stem)
        for i in range(len(depths) - 1):
            downsample_layer = nn.Sequential(
                norm_layer(dims[i]),
                HWCConv(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList([
            BasicLayer(n_blk=depths[i], dim=dims[i], sp_win_size=sp_win_size[i], sp_num_heads=sp_num_heads[i],
                       ch_win_size=ch_win_size[i], ch_num_heads=ch_num_heads[i], mlp_ratio=mlp_ratio,
                       norm_layer=norm_layer, drop_path_rate=dp_rates[sum(depths[:i]):sum(depths[:i + 1])])
            for i in range(len(depths))])

        self.norm = norm_layer(dims[-1])  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def _forward_features(self, x):
        for i in range(len(self.stages)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([1, 2]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self._forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        feature_size = self.img_size
        flops = 0.
        for i in range(len(self.downsample_layers)):
            for layer in self.downsample_layers[i]:
                if isinstance(layer, HWCConv):
                    feature_size /= layer.stride
                    flops += layer.flops() * (feature_size ** 2)
            flops += self.stages[i].flops()
        flops += self.dims[-1] * self.num_classes
        return flops

    def info(self):
        params = sum([torch.numel(param) for param in self.parameters() if param.requires_grad])
        flops = self.flops()
        print(f"params: {params / 1e6:.1f}M, flops: {flops / 1e9:.1f}G")
