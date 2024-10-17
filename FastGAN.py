import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from torchvision import transforms, datasets
import yaml
import random
from ResNet import seed_everything, ResNet18
import PIL
from pathlib import Path
import numpy as np

device = torch.device("cuda")

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
    def __init__(self, ch_in, ch_out):
        super(BasicBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.net = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2.),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            GLU()
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


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.cfg = None
        self.net = nn.ModuleList()

    def forward(self, input):
        outputs = []
        for i, layer_cfg in enumerate(self.cfg):
            if i == 0:
                out = self.net[0](input)
            else:
                arg_is = layer_cfg[0] if isinstance(layer_cfg[0], list) else [layer_cfg[0]]
                args = [outputs[arg_i] for arg_i in arg_is]
                out = self.net[i](*args)
            outputs.append(out)
        return outputs[-1]

    def parse_model(self, path):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        self.cfg = data["FastGAN"]

        for layer_cfg in self.cfg:
            if layer_cfg[1] == "Conv":
                layer = nn.Conv2d(*layer_cfg[2])
            elif layer_cfg[1] == "Tanh":
                layer = nn.Tanh()
            else:
                layer = eval(layer_cfg[1])(*layer_cfg[2])
            self.net.append(layer)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.cfg = None
        self.crop = None
        self.net = nn.ModuleList()
        self.decoder_0 = Decoder(256, 3)
        self.decoder_1 = Decoder(512, 3)

    def forward(self, input, crop=None):
        outputs = []
        for i, layer_cfg in enumerate(self.cfg):
            if i == 0:
                out = self.net[0](input)
            else:
                arg_is = layer_cfg[0] if isinstance(layer_cfg[0], list) else [layer_cfg[0]]
                args = [outputs[arg_i] for arg_i in arg_is]
                out = self.net[i](*args)
            outputs.append(out)

        if crop is not None:
            self.crop = [outputs[i] for i in crop]

        return outputs[-1]

    def parse_model(self, path):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        self.cfg = data["FastGAN"]

        for layer_cfg in self.cfg:
            if layer_cfg[1] == "Conv":
                layer = nn.Conv2d(*layer_cfg[2])
            elif layer_cfg[1] == "LeakyReLU":
                layer = nn.LeakyReLU(negative_slope=layer_cfg[2])
            elif layer_cfg[1] == "BatchNorm":
                layer = nn.BatchNorm2d(layer_cfg[2])
            elif layer_cfg[1] == "AdaptivePool":
                layer = nn.AdaptiveAvgPool2d(layer_cfg[2])
            elif layer_cfg[1] == "Flatten":
                layer = nn.Flatten(*layer_cfg[2])
            elif layer_cfg[1] == "Linear":
                layer = nn.Linear(*layer_cfg[2])
            else:
                layer = eval(layer_cfg[1])(*layer_cfg[2])
            self.net.append(layer)


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True

def norm(input):
    batch = input.shape[0]
    res = torch.zeros([]).to(device)
    for i in range(batch):
        res = res + torch.norm(input[i])
    res = res / batch
    return res


def train():
    BATCHSIZE = 32
    LEARNING_RATE = 5e-4
    EPOCH = 200
    ROOT = Path(r".\checkpoints\tmp")

    my_trans = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor()
    ])
    database = datasets.ImageFolder(r".\dataset\X-SDD\datas", transform=my_trans)

    train = DataLoader(database, batch_size=BATCHSIZE // 2, shuffle=True)

    G, D = Generator(), Discriminator()
    G.parse_model(".\models\FastGAN\Generator.yaml")
    D.parse_model(".\models\FastGAN\Discriminator.yaml")
    G.to(device)
    D.to(device)

    optimizer = optim.Adam([*G.parameters(), *D.parameters()], lr=LEARNING_RATE)
    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=0.)
    resize = transforms.Resize([64, 64], antialias=True)
    ToImg = transforms.ToPILImage()


    for epoch in range(EPOCH):
        D.train(), G.train()

        for batch_i, (x_data, label_cls) in enumerate(train):
            x_data, label_cls = x_data.to(device), label_cls.to(device)
            x_data = x_data * 2. - 1.

            #disriminator
            freeze(G)
            z = torch.randn([x_data.shape[0], 256, 1, 1]).to(device)
            x_g = G(z)

            logits_fake = D(x_g)
            logits_real = D(x_data, crop=[7, 8])
            (f1, f2), D.crop = D.crop, None

            #Regularization
            h, w = [random.randint(0, 8) for _ in range(2)]
            f1_crop = f1[..., h:h+8, w:w+8]
            I, I_part = resize(x_data), x_data[...,h*8:h*8+64, w*8:w*8+64]
            I_, I_part_ = D.decoder_1(f2), D.decoder_0(f1_crop)

            #loss
            loss_real = -torch.mean(torch.min(torch.zeros([]), -1. + logits_real))
            loss_fake = -torch.mean(torch.min(torch.zeros([]), -1. - logits_fake))
            loss_recons = norm(I - I_) + norm(I_part - I_part_)
            loss = loss_real + loss_fake + loss_recons

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            unfreeze(G)

            #generator
            freeze(D)

            z = torch.randn([BATCHSIZE, 256, 1, 1]).to(device)
            logits = D(G(z))
            loss = -torch.mean(logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            unfreeze(D)

            if batch_i % 20 == 0:
                print(f"loss_real:{loss_real}, loss_fake:{loss_fake}, loss_recons:{loss_recons}")

        scheduler_lr.step()

        D.eval(), G.eval()

        with torch.no_grad():
            z = torch.randn([9, 256, 1, 1]).to(device)
            x_g = (G(z) + 1.) / 2.
            _, _, h, w = x_g.shape
            img = PIL.Image.new("RGB", (h * 3, w * 3))
            for i in range(9):
                h_i, w_i = i // 3, i % 3
                img_ = ToImg(x_g[i])
                img.paste(img_, (w * w_i, h * h_i))
            img.save(rf".\runs\FastGAN\imgs\epoch{epoch}.jpg")

        if epoch != 0 and epoch % 10 == 0:
            new_dir = ROOT.joinpath(f"epoch{epoch}")
            new_dir.mkdir(parents=True, exist_ok=True)
            torch.save(G.state_dict(), new_dir.joinpath("G.pt"))
            torch.save(D.state_dict(), new_dir.joinpath("D.pt"))
            torch.save(optimizer.state_dict(), new_dir.joinpath("optimizer.pt"))
            torch.save(scheduler_lr.state_dict(), new_dir.joinpath("scheduler_lr.pt"))

    torch.save(G.state_dict(), r".\checkpoints\tmp\G.pt")
    torch.save(D.state_dict(), r".\checkpoints\tmp\D.pt")


#record dataset info
def record_dataset(dataset, state_dict):
    classes = len(dataset.classes)
    num_classes = [0] * classes
    for _, cls in dataset:
        num_classes[cls] += 1

    l = [(dataset.classes[i], num_classes[i]) for i in range(classes)]
    state_dict.update({"dataset_info": l})
    return state_dict


#get embeddings of generated imgs
def get_test():
    state_dict = {}

    my_trans = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor()
    ])
    database = datasets.ImageFolder(r".\dataset\X-SDD\datas", transform=my_trans)
    record_dataset(database, state_dict)

    G, D = Generator(), Discriminator()
    G.parse_model(".\models\FastGAN\Generator.yaml")
    D.parse_model(".\models\FastGAN\Discriminator.yaml")
    G.load_state_dict(torch.load(r".\checkpoints\FastGAN_G.pt"))
    D.load_state_dict(torch.load(r".\checkpoints\FastGAN_D.pt"))
    G.to(device), D.to(device)
    cls_num = len(state_dict["dataset_info"])
    classifier_0 = ResNet18(input_shape=[128, 128], class_num=cls_num, chs=[3, 64, 128, 256, 512])
    classifier_0.load_state_dict(torch.load(r".\checkpoints\ResNet18_epoch88_acc98.16.pt"))
    classifier_1 = classifier_0.classifier[-1]
    classifier_0.classifier.pop(-1)
    classifier_0.to(device), classifier_1.to(device)
    Encoder = torch.load(r".\checkpoints\Encoder.pt")
    Encoder.to(device)

    #sample
    ToImg = transforms.ToPILImage()
    n = 4000
    z = torch.randn([n, 256, 1, 1]).to(device)
    sample = DataLoader(z, batch_size=100, shuffle=False)

    G.eval(), classifier_0.eval(), classifier_1.eval(), Encoder.eval()
    pred_cls_num = [0] * cls_num
    pred_cls_num_9 = [0] * cls_num
    embedding_lst = [[] for _ in range(cls_num)]
    for batchi, x in enumerate(sample):
        with torch.no_grad():
            x_g = (G(x) + 1.) / 2.
            embeddings = classifier_0(x_g)
            logits = classifier_1(embeddings)
            pred = F.softmax(logits, dim=1)

            for i, cls in enumerate(torch.argmax(pred, dim=1).tolist()):
                if pred[i, cls] >= 0.9:
                    pred_cls_num_9[cls] += 1
                    embedding = embeddings[i].view(1, -1)
                    embedding = Encoder(embedding).view(-1).tolist()
                    embedding_lst[cls].append(embedding)
                pred_cls_num[cls] += 1

            if False:
                batchsz, _, h, w = x_g.shape
                imgsz = 6
                img = PIL.Image.new("RGB", (w * imgsz, h * imgsz))
                arg_i = torch.randint(low=0, high=batchsz - 1, size=[imgsz**2]).tolist()
                for i, arg in enumerate(arg_i):
                    h_i, w_i = i // imgsz, i % imgsz
                    img_ = ToImg(x_g[arg])
                    img.paste(img_, (w * w_i, h * h_i))
                img.save(rf".\runs\FastGAN\test_imgs\{batchi}.jpg")

    state_dict.update({"pred": pred_cls_num, "pred_.9": pred_cls_num_9})
    print(state_dict)
    torch.save(state_dict, r".\runs\test_info.pt")

    for i in range(cls_num):
        print(len(embedding_lst[i]))
        embedding_lst[i] = np.array(embedding_lst[i]).reshape(-1, 2)
    torch.save({"embeddings": embedding_lst}, r".\runs\data_fake_embeddings.pt")


def analyze_plot():
    import matplotlib.pyplot as plt

    dict_real = torch.load(r".\runs\FastGAN\data_real_embeddings.pt")
    dict_fake = torch.load(r".\runs\FastGAN\data_fake_embeddings.pt")
    test_info = torch.load(r".\runs\FastGAN\test_info.pt")

    classes = test_info["dataset_info"]
    cls_num = len(classes)
    pred, pred_9 = test_info["pred"], test_info["pred_.9"]
    ttl_pred, ttl_pred_9 = sum(pred), sum(pred_9)
    print(test_info)

    embed_real = dict_real["embeddings"]
    embed_fake = dict_fake["embeddings"]

    plt.figure(figsize=(4.8*3, 3.6*3))
    for i in range(cls_num):
        print(f"{classes[i][0]}, ttl:{embed_fake[i].shape[0]}")
        x_real, y_real = embed_real[i][:, 0], embed_real[i][:, 1]
        x_fake, y_fake = embed_fake[i][:, 0], embed_fake[i][:, 1]
        plt.subplot(3, 3, i + 1)
        plt.xticks([]), plt.yticks([])
        plt.scatter(x_fake, y_fake, c="red", marker=".", alpha=0.2, label="fake", zorder=1)
        plt.scatter(x_real, y_real, c="blue", marker=".", alpha=0.7, label="real", zorder=2)
        plt.title(f"{classes[i][0]}, ttl:{pred[i] / ttl_pred * 100:.1f}%, "
                                                f"≥.9:{pred_9[i] / ttl_pred_9 * 100:.1f}%")
        plt.legend(loc="upper right")
    plt.show()



if __name__ == "__main__":
    seed_everything(0)
    # train()
    # get_test()
    analyze_plot()

