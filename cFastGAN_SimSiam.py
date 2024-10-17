import torch
from torch.utils.data import Subset
from header import *
from utills import *
from Modules import *
from SN import SNClassifier, ClassSampler
from SimSiam import SimSiam

G_YAML = r".\models\cFastGAN\cGenerator.yaml"
D_YAML = r".\models\cFastGAN\cDiscriminator_6.yaml"
SNC_YAML = r".\models\SN\classifier.yaml"
SimSiam_YAML = r".\models\SN\SimSiam.yaml"

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


def train(root, dirname="tmp", ckpt_dir=None, restart=False, *, epochs=160, alpha=1., eta_min=0.):
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
    EPOCH = epochs

    # sampler = get_WeightedRandomSampler(Subset(database, list(range(len(database)))))
    # train = DataLoader(database, batch_size=BATCHSIZE // 2, sampler=sampler)
    train = DataLoader(database, batch_size=BATCHSIZE // 2, shuffle=True)

    sn = SimSiam(SimSiam_YAML, None).to(device)
    sn.load_state_dict(torch.load(Path(root) / "SimSiam.pt"))
    sn = sn.backbone
    eval_(sn)
    freeze(sn)

    G = Generator(G_YAML).to(device)
    D = Discriminator(D_YAML).to(device)
    snc = SNClassifier(SNC_YAML).to(device)

    params_group = [
        {"params": G.parameters()},
        {"params": D.parameters()},
        {"params": snc.parameters(), "lr": alpha * LEARNING_RATE}
    ]
    optimizer = optim.Adam(params_group, lr=LEARNING_RATE)
    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=eta_min)

    if restart:
        ckpt_dir = ROOT.joinpath(ckpt_dir)
        G.load_state_dict(torch.load(ckpt_dir.joinpath("G.pt")))
        D.load_state_dict(torch.load(ckpt_dir.joinpath("D.pt")))
        snc.load_state_dict(torch.load(ckpt_dir.joinpath("SNC.pt")))
        optimizer.load_state_dict(torch.load(ckpt_dir.joinpath("optimizer.pt")))
        scheduler_lr.load_state_dict(torch.load(ckpt_dir.joinpath("scheduler_lr.pt")))
        EPOCH_START = int(ckpt_dir.name.split("epoch")[-1]) + 1
    else:
        with open(ROOT / "training_config", mode="a") as txt_writer:
            s = f"batchsize:{BATCHSIZE}\nlearning_rate:{LEARNING_RATE}\nepoch:{epochs}\n" \
                f"alpha:{alpha}\neta_min:{eta_min}"
            txt_writer.write(s)

    ToImg = transforms.ToPILImage()
    clsSampler = ClassSampler(database)

    for epoch in range(EPOCH_START if restart else 0, EPOCH):
        train_(G, D, snc)

        for batch_i, (img_real, label_real) in enumerate(train):
            img_real, label_real = img_real.to(device), label_real.to(device)
            img_real = img_real * 2. - 1.
            batchsz = img_real.shape[0]
            onehot_real = F.one_hot(label_real, num_classes=CLS_NUM).to(torch.float32)

            #snc
            img_div = clsSampler.cls_sample(label_real)
            img_div = img_div * 2. - 1.

            with torch.no_grad():
                z0 = torch.randn([batchsz, 256], device=device)
                z1 = torch.randn([batchsz, 256], device=device)
                img_fake0, img_fake1 = G([z0, onehot_real]), G([z1, onehot_real])

                t0_div, t1_div = sn(img_real), sn(img_div)
                t0_sm, t1_sm = sn(img_fake0), sn(img_fake1)

            logits_div = snc(t0_div, t1_div)
            logits_sm = snc(t0_sm, t1_sm)

            loss_snc_div = snc.loss4G(logits_div, similar=False)
            loss_snc_sm = snc.loss4G(logits_sm, similar=True)

            loss_snc = loss_snc_sm + loss_snc_div

            optimizer.zero_grad()
            loss_snc.backward()
            optimizer.step()

            if batch_i % 20 == 0:
                print(f"SNC: div:{loss_snc_div:.2f}, sm:{loss_snc_sm:.2f}")

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
            freeze(D, snc)

            z0 = torch.randn([batchsz, 256], device=device)
            z1 = torch.randn([batchsz, 256], device=device)
            img_fake0, img_fake1 = G([z0, onehot_real]), G([z1, onehot_real])

            out0 = D([img_fake0, onehot_real])
            out1 = D([img_fake1, onehot_real])

            t0, t1 = sn(img_fake0), sn(img_fake1)
            logits = snc(t0, t1)

            loss_G_D = -(torch.mean(out0[0]) + torch.mean(out1[0])) / 2.
            loss_G_cls = (D.loss_classifier(out0[1], label_real) + D.loss_classifier(out1[1], label_real)) / 2.
            loss_G_snc = -torch.mean(logits)

            loss_G = loss_G_D + loss_G_cls + loss_G_snc

            optimizer.zero_grad()
            loss_G.backward()
            optimizer.step()

            unfreeze(D, snc)

            if batch_i % 20 == 0:
                print(f"G: loss_D:{loss_G_D:.2f}, cls:{loss_G_cls:.2f}, sn:{loss_G_snc:.2f}")

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
            torch.save(snc.state_dict(), new_dir.joinpath("SNC.pt"))
            torch.save(optimizer.state_dict(), new_dir.joinpath("optimizer.pt"))
            torch.save(scheduler_lr.state_dict(), new_dir.joinpath("scheduler_lr.pt"))

    torch.save(G.state_dict(), ROOT.joinpath("G.pt"))
    torch.save(D.state_dict(), ROOT.joinpath("D.pt"))
    torch.save(snc.state_dict(), ROOT.joinpath("SNC.pt"))

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
    root = r".\checkpoints\SimSiam"
    train(root=root, dirname="0", epochs=200, alpha=1., eta_min=1e-5)

