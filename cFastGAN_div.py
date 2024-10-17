import torch
from torch.utils.data import Subset
from header import *
from utills import *
from Modules import *
from ResNet import ResNet18
from SN import SN, SNClassifier, ClassSampler

G_YAML = r".\models\cFastGAN\cGenerator.yaml"
D_YAML = r".\models\cFastGAN\cDiscriminator_6.yaml"
DIV_YAML = r".\models\SN\diversifier.yaml"

class Discriminator(BasicModule):
    def __init__(self, yml):
        super(Discriminator, self).__init__()
        self.yml = yml
        self.parse_model(self.yml)

    def forward(self, inputs):
        return super(Discriminator, self).forward(inputs)

    def loss_real(self, inputs):
        return -torch.mean(torch.min(torch.zeros([]).to(device), -1. + inputs))

    def loss_fake(self, inputs):
        return -torch.mean(torch.min(torch.zeros([]).to(device), -1. - inputs))

    def loss_recons(self, inputs):
        return torch.norm(inputs[0] - inputs[1]) / inputs.shape[1]

    def loss_classifier(self, inputs, label):
        return F.cross_entropy(inputs, label, label_smoothing=0.1)


class Generator(BasicModule):
    def __init__(self, yml):
        super(Generator, self).__init__()
        self.yml = yml
        self.parse_model(yml)

    def forward(self, inputs):
        return super(Generator, self).forward(inputs)


class Diversifier(BasicModule):
    #diverse -> 1, similar -> 0
    def __init__(self, yml):
        super(Diversifier, self).__init__()
        self.yml = yml
        self.parse_model(yml)

    def forward(self, inputs):
        return super(Diversifier, self).forward(inputs)

    def loss(self, inputs, similar=None):
        if similar is None:
            return torch.mean(torch.max(torch.zeros([]).to(device)), -1. - inputs, -1. + inputs)
        elif similar:
            return -torch.mean(torch.min(torch.zeros([]).to(device), -1. - inputs))
        else:
            return -torch.mean(torch.min(torch.zeros([]).to(device), -1. + inputs))

    def loss_classifier(self, inputs, label):
        return F.cross_entropy(inputs, label, label_smoothing=0.1)


class Diversifier6(BasicModule):
    def __init__(self, yml):
        super(Diversifier6, self).__init__()
        self.yml = yml
        self.parse_model(yml)

    def forward(self, inputs):
        idx = [0, 3, 1, 4, 2, 5]
        inputs = inputs[:, idx, ...]
        f_lst = torch.split(inputs, 2)
        v_lst = [self.extractor(f) for f in f_lst]
        return self.net(torch.cat(v_lst, dim=1))

    def loss(self, inputs, similar:bool):
        if similar:
            return -torch.mean(torch.min(torch.zeros([]).to(device), -1. - inputs))
        else:
            return -torch.mean(torch.min(torch.zeros([]).to(device), -1. + inputs))


def train(root, dirname="tmp", ckpt_dir=None, restart=False):
    ROOT = Path(root) / dirname
    ROOT.mkdir(parents=True, exist_ok=True)
    ROOT.joinpath("imgs").mkdir(parents=True, exist_ok=True)

    my_trans = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor()
    ])
    database = datasets.ImageFolder(r".\dataset\X-SDD\datas", transform=my_trans)

    CLS_NUM = len(database.classes)
    BATCHSIZE = 32
    LEARNING_RATE = 1.5e-4
    EPOCH = 1

    # sampler = get_WeightedRandomSampler(Subset(database, list(range(len(database)))))
    # train = DataLoader(database, batch_size=BATCHSIZE // 2, sampler=sampler)
    train = DataLoader(database, batch_size=BATCHSIZE // 2, shuffle=True)

    G = Generator(G_YAML).to(device)
    D = Discriminator(D_YAML).to(device)
    Div = Diversifier6(DIV_YAML).to(device)

    params_group = [
        {"params": G.parameters()},
        {"params": D.parameters()},
        {"params": Div.parameters()},
    ]
    optimizer = optim.Adam(params_group, lr=LEARNING_RATE)
    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=0.)

    if restart:
        ckpt_dir = ROOT.joinpath(ckpt_dir)
        G.load_state_dict(torch.load(ckpt_dir.joinpath("G.pt")))
        D.load_state_dict(torch.load(ckpt_dir.joinpath("D.pt")))
        Div.load_state_dict(torch.load(ckpt_dir.joinpath("Div.pt")))
        optimizer.load_state_dict(torch.load(ckpt_dir.joinpath("optimizer.pt")))
        scheduler_lr.load_state_dict(torch.load(ckpt_dir.joinpath("scheduler_lr.pt")))
        EPOCH_START = int(ckpt_dir.name.split("epoch")[-1]) + 1
    else:
        with open(ROOT / "training_config", mode="a") as txt_writer:
            s = f"batchsize:{BATCHSIZE}, learning_rate:{LEARNING_RATE}, epoch:{EPOCH}\n" \
                f"alpha:{alpha}, min_eta:{min_eta}"
            txt_writer.write(s)

    ToImg = transforms.ToPILImage()
    clsSampler = ClassSampler(database)

    for epoch in range(EPOCH_START if restart else 0, EPOCH):
        train_(G, D, Div)

        for batch_i, (img_real, label_real) in enumerate(train):
            img_real, label_real = img_real.to(device), label_real.to(device)
            img_real = img_real * 2. - 1.
            batchsz = img_real.shape[0]
            onehot_real = F.one_hot(label_real, num_classes=CLS_NUM).to(torch.float32)

            #Div
            img_div = clsSampler.cls_sample(label_real).to(device)
            img_div = img_div * 2. - 1.

            with torch.no_grad():
                z0 = torch.randn([batchsz, 256], device=device)
                z1 = torch.randn([batchsz, 256], device=device)
                img_fake0, img_fake1 = G([z0, onehot_real]), G([z1, onehot_real])

            #0
            out_div = Div(torch.cat([img_real, img_div], dim=1), paired=False)
            out_sm = Div(torch.cat([img_fake0, img_fake1], dim=1), paired=False)

            loss_div = Div.loss(out_div, similar=False)
            loss_sm = Div.loss(out_sm, similar=True)

            loss = loss_sm + loss_div

            optimizer.zero_grad()
            loss.backward()

            if batch_i % 20 == 0:
                print(f"Div: div:{loss_div:.2f}, sm:{loss_sm:.2f}")

            #1
            out_div = Div(torch.cat([img_real, img_div], dim=1), paired=True)
            loss_div = Div.loss(out_div, similar=False)
            loss_div.backward()

            out_sm = Div(torch.cat([img_fake0, img_fake1], dim=1), paired=True)
            loss_sm = Div.loss(out_sm, similar=True)
            loss_sm.backward()

            optimizer.step()

            if batch_i % 20 == 0:
                print(f"Div: div:{loss_div:.2f}, sm:{loss_sm:.2f}")

            #disriminator
            out_fake = D([img_fake0, onehot_real])
            out_real = D([img_real, onehot_real])

            loss_D_real = D.loss_real(out_real[0])
            loss_D_fake = D.loss_fake(out_fake[0])
            loss_D_cls = D.loss_classifier(out_real[1], label_real)
            loss_D_recons = D.loss_recons(out_real[2]) + D.loss_recons(out_real[3])

            loss_D = loss_D_real + loss_D_fake + loss_D_cls + loss_D_recons

            optimizer.zero_grad()
            loss_D.backward()
            optimizer.step()

            if batch_i % 20 == 0:
                print(f"D: real:{loss_D_real:.2f}, fake:{loss_D_fake:.2f}, "
                      f"recons:{loss_D_recons:.2f}, cls:{loss_D_cls:.2f}")

            #generator
            freeze(D, Div)

            z0 = torch.randn([batchsz, 256], device=device)
            z1 = torch.randn([batchsz, 256], device=device)

            img_fake0, img_fake1 = G([z0, onehot_real]), G([z1, onehot_real])
            out0 = D([img_fake0, onehot_real])
            out1 = D([img_fake1, onehot_real])

            out_div = Div(torch.cat([img_fake0, img_fake1], dim=1), paired=False)

            loss_G_D = -(torch.mean(out0[0]) + torch.mean(out1[0])) / 2.
            loss_G_cls = (D.loss_classifier(out0[1], label_real) + D.loss_classifier(out1[1], label_real)) / 2.
            loss_G_Div = -torch.mean(out_div)

            loss_G = loss_G_D + loss_G_cls + loss_G_Div

            optimizer.zero_grad()
            loss_G.backward()
            optimizer.step()

            unfreeze(D, Div)

            if batch_i % 20 == 0:
                print(f"G: loss_D:{loss_G_D:.2f}, cls:{loss_G_cls:.2f}, Div:{loss_G_Div:.2f}")

        scheduler_lr.step()

        eval_(G)
        with torch.no_grad():
            z = torch.randn([16, 256], device=device)
            c = torch.arange(0, CLS_NUM, device=device).view(-1, 1)
            c = torch.expand_copy(c, size=[CLS_NUM, 2]).view(-1)
            c = torch.cat([c, torch.randint(0, CLS_NUM - 1, size=[16 - c.shape[0]], device=device)], dim=0)
            onehot_c = F.one_hot(c, num_classes=CLS_NUM).to(torch.float32)
            img_fake = (G([z, onehot_c]) + 1.) / 2.
            _, _, h, w = img_fake.shape
            img = PIL.Image.new("RGB", (h * 4, w * 4))
            for i in range(16):
                h_i, w_i = i // 4, i % 4
                img_ = ToImg(img_fake[i])
                img.paste(img_, (w * w_i, h * h_i))
            img.save(ROOT.joinpath(rf"imgs\epoch{epoch}.jpg"))

        if epoch != 0 and epoch % 20 == 0:
            new_dir = ROOT.joinpath(f"epoch{epoch}")
            new_dir.mkdir(parents=True, exist_ok=True)
            torch.save(G.state_dict(), new_dir.joinpath("G.pt"))
            torch.save(D.state_dict(), new_dir.joinpath("D.pt"))
            torch.save(Div.state_dict(), new_dir.joinpath("Div.pt"))
            torch.save(optimizer.state_dict(), new_dir.joinpath("optimizer.pt"))
            torch.save(scheduler_lr.state_dict(), new_dir.joinpath("scheduler_lr.pt"))

    torch.save(G.state_dict(), ROOT.joinpath("G.pt"))
    torch.save(D.state_dict(), ROOT.joinpath("D.pt"))
    torch.save(Div.state_dict(), ROOT.joinpath("Div.pt"))

    test_plot(ROOT, CLS_NUM)


def test_plot(root, num_classes):
    ROOT = Path(root)
    DIR = ROOT.joinpath("test_imgs")
    DIR.mkdir(parents=True, exist_ok=True)

    G = Generator(G_YAML).to(device)
    G.load_state_dict(torch.load(ROOT.joinpath("G.pt")))

    ToImg = transforms.ToPILImage()
    cls_sz = 720

    eval_(G)
    for cls in range(num_classes):
        z = torch.randn([cls_sz, 256], device=device)
        sampler = DataLoader(z, batch_size=36, shuffle=False)
        for batchi, x in enumerate(sampler):
            c = torch.tensor([cls], device=device)
            c_onehot = F.one_hot(c.expand(x.shape[0]), num_classes=num_classes)
            c_onehot = c_onehot.view(-1, num_classes).to(torch.float32)
            with torch.no_grad():
                img_fake = (G([x, c_onehot]) + 1.) * .5
                batchsz, _, h, w = img_fake.shape
                imgsz = 6
                img = PIL.Image.new("RGB", (w * imgsz, h * imgsz))
                for i in range(36):
                    h_i, w_i = i // 6, i % 6
                    img_ = ToImg(img_fake[i])
                    img.paste(img_, (w * w_i, h * h_i))
                img.save(DIR.joinpath(rf"cls_{cls}_{batchi}.jpg"))

if __name__ == "__main__":
    seed_everything(0)
    # train(root=r".\checkpoints\tmp7", ckpt_dir="epoch10", restart=False)
    # test_plot(r".\checkpoints\tmp7\p_cond_D_2", 7)
    # os.system("shutdown -s -t 10")
    m = nn.Linear(25, 100)
    optimizer = optim.Adam(m.parameters(), lr=1.)
    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=160, eta_min=1e-5)
    for i in range(160):
        print(i, scheduler_lr.get_lr())
        scheduler_lr.step()
