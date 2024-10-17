import torch
import winnt

from header import *

class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.cfg = None
        self.save = None

    def forward(self, inputs):
        x = inputs
        outputs = []
        for name, module in self.named_children():
            for i, layer_cfg in enumerate(self.cfg[name]):
                # print(name, i)
                fr_i = layer_cfg[0]
                if fr_i == -1:
                    x = module[i](x)
                elif fr_i is None:
                    x = module[i](inputs)
                else:
                    fr_i = fr_i if isinstance(fr_i, list) else [fr_i]
                    args = []
                    for arg_i in fr_i:
                        if isinstance(arg_i, list):
                            k, v = arg_i
                            args.append(inputs[v] if k is None else x[v] if k == -1 else outputs[k][v])
                        else:
                            args.append(inputs if arg_i is None else x if arg_i == -1 else outputs[arg_i])

                    x = module[i](*args)

                outputs.append(x if len(outputs) in self.save else None)

        return x

    def parse_model(self, path):
        with open(path, "r") as f:
            self.cfg = cfg = yaml.safe_load(f)

        save = []
        for name, module_cfg in cfg.items():
            module_lst = []
            for layer_cfg in module_cfg:
                # print(name)
                if len(layer_cfg) >= 3:
                    layer = eval(layer_cfg[1])(*layer_cfg[2])
                else:
                    layer = eval(layer_cfg[1])()
                module_lst.append(layer)

                fs = [layer_cfg[0]] if isinstance(layer_cfg[0], (int, type(None))) else layer_cfg[0]
                for f in fs:
                    if isinstance(f, list) and f[0] is not None and f[0] != -1:
                        save.append(f[0])
                    elif isinstance(f, int) and f != -1:
                        save.append(f)

            self.add_module(name, nn.Sequential(*module_lst))

        save.sort()
        self.save = save

#----------------ResNet------------------------
class ResBlk(nn.Module):
    """
    resnet block
    """

    def __init__(self, ch_in, ch_out, stride=1):
        """

        :param ch_in:
        :param ch_out:
        """
        super(ResBlk, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.1),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.1)
        )
        if stride == 1 and ch_in == ch_out:
            net = nn.Sequential()
        elif stride == 1:
            net = Conv2d(ch_in, ch_out, 1, 1, 0)
        else:
            net = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                Conv2d(ch_in, ch_out, 1, 1, 0)
                )
        self.shortcut = net

    def forward(self, x):
        """
        :param x: [b, c, h, w]
        :return:
        """

        out = self.net(x) + self.shortcut(x)
        return out


#-----------------cFastGAN--------------------
class Concat(nn.Module):
    def __init__(self, ch_in, ch_out, dim=1, parametric=True):
        super(Concat, self).__init__()
        self.dim = dim
        self.parametric = parametric
        if parametric:
            self.net = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(0.1)
            )

    def forward(self, *args):
        out = torch.cat([*args], dim=self.dim)
        return self.net(out) if self.parametric else out


class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, negative_slope=0.1):
        super(ResBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(negative_slope=negative_slope)
        )
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(negative_slope=negative_slope)
        )

    def forward(self, input):
        out = self.net(input) + self.shortcut(input)
        return out


class SLE(nn.Module):
    def __init__(self, ch_in, ch_ex, negative_slope=0.1):
        super(SLE, self).__init__()
        self.ch_in = ch_in
        self.ch_ex = ch_ex
        self.net = nn.Sequential(
            #[b, ch_ex, 4, 4]
            nn.AdaptiveAvgPool2d([4, 4]),
            nn.Conv2d(ch_ex, ch_ex, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(ch_ex, ch_in, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, input, excitation):
        scalers = self.net(excitation)
        out = input * scalers
        return out


class GLU(nn.Module):
    def __init__(self, parametric=False, ch_in=None, ch_out=None):
        super(GLU, self).__init__()
        self.parametric = parametric
        self.activation = nn.GLU()
        if parametric:
            self.conv_in = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)
            self.conv_ex = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        if self.parametric:
            conv_in = self.conv_in(input)
            conv_ex = self.conv_ex(input)
            actv_in = torch.cat([conv_in, conv_ex], dim=-1)
        else:
            actv_in = torch.cat([input, input], dim=-1)
        out = self.activation(actv_in)
        return out


class ConvTranspose(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvTranspose, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(ch_out),
            GLU()
        )

    def forward(self, input):
        out = self.net(input)
        return out


class BasicBlock(nn.Module):
    def __init__(self, ch_in, ch_out, parametric=False):
        super(BasicBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.net = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2.),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            GLU(parametric, ch_out, ch_out)
        )

    def forward(self, input):
        out = self.net(input)
        return out


class Decoder(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Decoder, self).__init__()
        self.net = nn.Sequential()
        for i in range(3):
            self.net.append(nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2.),
                nn.Conv2d(ch_in, (ch_in // 2) if i != 2 else ch_out,
                          kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d((ch_in // 2) if i != 2 else ch_out),
                GLU()
            ))
            ch_in = ch_in // 2

    def forward(self, input):
        out = self.net(input)
        return out


class Gather(nn.Module):
    def __init__(self):
        super(Gather, self).__init__()

    def forward(self, *args):
        return [arg for arg in args]


class Crop(nn.Module):
    def __init__(self, rate):
        super(Crop, self).__init__()
        self.rate = rate

    def forward(self, features, imgs):
        _, _, h, w = imgs.shape
        _, _, h_f, w_f = features.shape
        s_h, s_w = h // h_f, w // w_f
        h_, w_ = math.floor(h_f * self.rate), math.floor(w_f * self.rate)
        y, x = random.randint(0, h_f - h_), random.randint(0, w_f - w_)
        y_, x_, sz_h, sz_w = y * s_h, x * s_w, h_ * s_h, w_ * s_w
        out = [features[..., y:y+h_, x:x+w_], imgs[..., y_:y_+sz_h, x_:x_+sz_w]]
        return out


class Stack(nn.Module):
    def __init__(self, dim=0):
        super(Stack, self).__init__()
        self.dim = dim

    def forward(self, *args):
        return torch.stack(args, dim=self.dim)


class Split(nn.Module):
    def __init__(self, dim):
        super(Split, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.split(inputs, inputs.shape[self.dim] // 2, self.dim)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.args = args

    def forward(self, inputs):
        return inputs.view(*self.args)


class Conv2d(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding):
        super(Conv2d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size, stride, padding),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, inputs):
        return self.net(inputs)


class ImgReshape(nn.Module):
    def __init__(self, size, n):
        super(ImgReshape, self).__init__()
        self.n = n
        self.psz = size // n
        indices = []
        for i in range(n):
            indices.append(torch.arange(i, i + self.psz * n, n))
        self.i = torch.cat(indices, dim=0)

    def forward(self, inputs):
        n, psz = self.n, self.psz
        b, c, _, _ = inputs.shape
        inputs = inputs.view(b, c, n, -1, psz)
        inputs = inputs[..., self.i, :]
        return inputs.view(b, -1, psz, psz)


class CondBroadcast(nn.Module):
    def __init__(self, size, alpha=1.):
        super(CondBroadcast, self).__init__()
        self.size = size
        self.alpha = alpha

    def forward(self, inputs):
        sz = self.size
        b, c = inputs.shape
        inputs = inputs.view(b, c, 1, 1) * self.alpha
        return torch.broadcast_to(inputs, [b, c, sz, sz])


class CondBroadcast_(nn.Module):
    def __init__(self, size):
        super(CondBroadcast_, self).__init__()
        self.size = size

    def forward(self, inputs):
        sz = self.size
        b, c = inputs.shape
        mask = inputs.bool()
        cond = torch.ones([b, c], device=inputs.device) * (-1. / (c - 1))
        cond = torch.masked_fill(cond, mask, 1.)
        cond = cond.view(b, c, 1, 1)
        return torch.broadcast_to(cond, [b, c, sz, sz])


# class CondBroadcast_D(nn.Module):
#     def __init__(self, size):
#         super(CondBroadcast_D, self).__init__()
#         self.size = size
#
#     def forward(self, inputs):
#         sz = self.size
#         imgs, mask = inputs[0], inputs[1].bool()
#         b, c = mask.shape
#         cond = torch.ones([b, c], device=mask.device) * (-1. / (c - 1))
#         cond = torch.masked_fill(cond, mask, 1.).view(b, c, 1, 1)
#         cond = torch.broadcast_to(cond, [b, c, sz, sz])
#         return torch.cat([imgs, cond], dim=1)

class CondBroadcast_D(nn.Module):
    def __init__(self, size):
        super(CondBroadcast_D, self).__init__()
        self.size = size

    def forward(self, inputs):
        sz = self.size
        imgs, cond = inputs[0], inputs[1]
        b, c = cond.shape
        cond = cond.view(b, c, 1, 1)
        cond = torch.broadcast_to(cond, [b, c, sz, sz])
        return torch.cat([imgs, cond], dim=1)


# class CondBroadcast_G(nn.Module):
#     def __init__(self):
#         super(CondBroadcast_G, self).__init__()
#
#     def forward(self, inputs):
#         z, mask = inputs[0], inputs[1].bool()
#         b, c = mask.shape
#         cond = torch.ones([b, c], device=mask.device) * (-1. / (c - 1))
#         cond = torch.masked_fill(cond, mask, 1.)
#         return torch.cat([z, cond], dim=1).view(b, -1, 1, 1)


class CondBroadcast_G(nn.Module):
    def __init__(self):
        super(CondBroadcast_G, self).__init__()

    def forward(self, inputs):
        z, cond = inputs[0], inputs[1]
        b = z.shape[0]
        return torch.cat([z, cond], dim=1).view(b, -1, 1, 1)


class Cond_D(nn.Module):
    def __init__(self, size, ch_in, ch_out):
        super(Cond_D, self).__init__()
        self.net = nn.Sequential(
            CondBroadcast_D(size),
            Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0),
            Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        )
        self.shortcut = nn.Sequential(
            Conv2d(3, ch_out, kernel_size=5, stride=1, padding=2)
        )

    def forward(self, inputs):
        return self.net(inputs) + self.shortcut(inputs[0])


class SPPF(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SPPF, self).__init__()
        ch = ch_in // 2
        self.conv = Conv2d(ch_in, ch, 3, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.cat = Concat(ch * 4, ch_out)

    def forward(self, inputs):
        f_lst = [self.conv(inputs)]
        for _ in range(3):
            f_lst.append(self.pool(f_lst[-1]))
        return self.cat(*f_lst)


class UpSample(nn.Module):
    def __init__(self, ch_in, ch_out, parametric=False):
        super(UpSample, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.net = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2.),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.LeakyReLU(0.1)
        )

    def forward(self, input):
        out = self.net(input)
        return out

class DivHead(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DivHead, self).__init__()
        self.conv = Conv2d(6, 64, 3, 1, 1)

    def forward(self, inputs):
        img0, img1 = torch.split(inputs, inputs.shape[1] // 2, dim=1)
        zeros = torch.zeros([1, 1, 1, 1], device=inputs.device)
        zeros = torch.broadcast_to(zeros, img0.shape)
        f = self.conv(inputs)
        f0 = self.conv(torch.cat([img0, zeros], dim=1))
        f1 = self.conv(torch.cat([zeros, img1], dim=1))
        return f + f0 + f1

if __name__ == "__main__":
    m = SPPF(32, 32)
    inputs = torch.randn([2, 32, 8, 8])
    o = m(inputs)
    print(o.shape)