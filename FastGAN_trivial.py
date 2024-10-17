import torch
from torch.utils.data import Subset
from header import *
from utills import *
from Modules import *
from ResNet import ResNet18

G0_YAML = r".\models\cFastGAN\Generator_t0.yaml"
G1_YAML = r".\models\cFastGAN\Generator_t1.yaml"
D0_YAML = r".\models\cFastGAN\Discriminator_t0.yaml"
D1_YAML = r".\models\cFastGAN\Discriminator_t1.yaml"
RESNET_YAML = r".\models\ResNet18\ResNet18.yaml"
ENCODER_YAML = r".\models\ResNet18\encoder.yaml"


class Discriminator(BasicModule):
    def __init__(self, yml):
        super(Discriminator, self).__init__()
        self.yml = yml
        self.parse_model(self.yml)

    def forward(self, input):
        return super(Discriminator, self).forward(input)

    def loss_real(self, input):
        return -torch.mean(torch.min(torch.zeros([]).to(device), -1. + input))

    def loss_fake(self, input):
        return -torch.mean(torch.min(torch.zeros([]).to(device), -1. - input))

    def loss_recons(self, input):
        return torch.norm(input[0] - input[1]) / input.shape[1]

    def loss_classifier(self, input, label):
        return F.cross_entropy(input, label)

    def loss_embeddings(self, input):
        return torch.mean(torch.norm(input[0] - input[1], dim=1))


class Generator(BasicModule):
    def __init__(self, yml):
        super(Generator, self).__init__()
        self.yml = yml
        self.parse_model(yml)

    def forward(self, input):
        return super(Generator, self).forward(input)


def get_embds(database, E):
    train = DataLoader(database, batch_size=32, shuffle=False)
    CLS_NUM = len(database.classes)

    embds_real = [None] * CLS_NUM

    eval_(E)
    for batchi, (img_real, label_real) in enumerate(train):
        img_real, label_real = img_real.to(device), label_real.to(device)
        img_real = img_real * 2. - 1.
        batchsz = img_real.shape[0]

        with torch.no_grad():
            embd_real = E(img_real)

        for i in range(batchsz):
            c = label_real[i]
            if embds_real[c] is None:
                embds_real[c] = embd_real[i:i + 1]
            else:
                embds_real[c] = torch.cat([embds_real[c], embd_real[i:i + 1]], dim=0)

    res = []
    for i in range(CLS_NUM):
        mean_real = torch.mean(embds_real[i], dim=0)
        std_real = torch.mean(torch.norm(embds_real[i] - mean_real, dim=1))
        print(f"cls:{i}, std_real:{std_real:.2f}")
        res.append(mean_real)
    res = torch.stack(res, dim=0)
    res.to(device)

    return res

def samplez(G, s):
    z = torch.randn(s.shape, device=device, requires_grad=True)
    optimizer = optim.Adam([z], lr=0.1)
    for i in range(50):
        s_ = G(z)
        loss = torch.mean(torch.norm(s - s_, dim=1))
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    z.requires_grad = False
    return z


def train(root, ckpt_dir=None, restart=False):
    ROOT = Path(root)

    my_trans = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor()
    ])
    database = datasets.ImageFolder(r".\dataset\X-SDD\datas", transform=my_trans)

    CLS_NUM = len(database.classes)
    BATCHSIZE = 32
    LEARNING_RATE = 1e-4
    EPOCH = 200

    sampler = get_WeightedRandomSampler(Subset(database, list(range(len(database)))))
    train = DataLoader(database, batch_size=BATCHSIZE // 2, sampler=sampler)

    G0 = Generator(G0_YAML).to(device)
    D0 = Discriminator(D0_YAML).to(device)
    G1 = Generator(G1_YAML).to(device)
    D1 = Discriminator(D1_YAML).to(device)
    E = ResNet18(ENCODER_YAML).to(device)

    G1.load_state_dict(torch.load(r".\checkpoints\G1.pt"))
    D1.load_state_dict(torch.load(r".\checkpoints\D1.pt"))
    E.load_state_dict(torch.load(r".\checkpoints\encoder_epoch76_acc96.13.pt"))
    encoder = list(E.children())[0]
    encoder = nn.Sequential(*encoder.children())
    classifier = list(E.children())[1][0]

    eval_(G1, D1, encoder, classifier)
    freeze(G1, D1, encoder, classifier)

    # means = get_embds(database, encoder)

    params_group = [
        {"params": G0.parameters(), "weight_decay": 1e-3},
        {"params": D0.parameters(), "weight_decay": 1e-2}
    ]
    optimizer = optim.Adam(params_group, lr=LEARNING_RATE)
    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=0.)
    ToImg = transforms.ToPILImage()

    if restart:
        ckpt_dir = ROOT.joinpath(ckpt_dir)
        G0.load_state_dict(torch.load(ckpt_dir.joinpath("G0.pt")))
        D0.load_state_dict(torch.load(ckpt_dir.joinpath("D0.pt")))
        optimizer.load_state_dict(torch.load(ckpt_dir.joinpath("optimizer.pt")))
        scheduler_lr.load_state_dict(torch.load(ckpt_dir.joinpath("scheduler_lr.pt")))
        EPOCH_START = int(ckpt_dir.name.split("epoch")[-1]) + 1

    for epoch in range(EPOCH_START if restart else 0, EPOCH):
        train_(G0, D0)
        for batch_i, (img_real, label_real) in enumerate(train):
            img_real, label_real = img_real.to(device), label_real.to(device)
            img_real = img_real * 2. - 1.
            batchsz = img_real.shape[0]

            #disriminator0
            freeze(G0)

            z = torch.randn([batchsz, 256], device=device)
            with torch.no_grad():
                img_fake = G0(G1(z))

            out_fake = D0(img_fake)
            out_real = D0(img_real)

            loss_D_real = D0.loss_real(out_real[0])
            loss_D_fake = D0.loss_fake(out_fake[0])
            loss_D_recons = D0.loss_recons(out_real[1]) + D0.loss_recons(out_real[2])
            loss_D_cls = D0.loss_classifier(out_real[4], label_real)

            loss = loss_D_real + loss_D_fake + loss_D_recons + loss_D_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_i % 20 == 0:
                print(f"D0: real:{loss_D_real:.2f}, fake:{loss_D_fake:.2f}, recons:{loss_D_recons:.2f}, "
                      f"cls:{loss_D_cls:.2f}")

            unfreeze(G0)

            #generator0
            freeze(D0)

            z1 = torch.randn([batchsz, 256], device=device)
            with torch.no_grad():
                s = encoder(img_real)
            z0 = samplez(G1, s)
            z = torch.cat([z0, z1], dim=0)
            img_fake = G0(G1(z))
            s_ = encoder(img_fake[:batchsz])
            out = D0(img_fake)

            # z1 = torch.randn([batchsz, 256], device=device)
            # with torch.no_grad():
            #     s = G1(z1)
            #     logits = classifier(s)
            #     p = torch.softmax(logits, dim=1)
            #     label_fake = torch.argmax(p, dim=1)
            # img_fake = G0(G1(z1))
            # s_ = encoder(img_fake)
            # out = D0(img_fake)

            # with torch.no_grad():
            #     s = encoder(img_real)
            # z = samplez(G1, s)
            # img_fake = G0(G1(z))
            # out = D0(img_fake)

            loss_G = -torch.mean(out[0])
            loss_G_cls = D0.loss_classifier(out[4][:batchsz], label_real)
            loss_G_recons = torch.norm(img_real - img_fake[:batchsz]) / batchsz \
                            + torch.mean(torch.norm(s - s_, dim=1))

            loss = loss_G + loss_G_cls + 0.1 * loss_G_recons

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_i % 20 == 0:
                print(f"G0: {loss_G:.2f}, cls:{loss_G_cls:.2f}, recons:{loss_G_recons:.2f}")

            unfreeze(D0)

        scheduler_lr.step()

        eval_(G0, D0)
        with torch.no_grad():
            z = torch.randn([9, 256], device=device)
            img_fake = (G0(G1(z)) + 1.) * .5
            _, _, h, w = img_fake.shape
            img = PIL.Image.new("RGB", (h * 3, w * 3))
            for i in range(9):
                h_i, w_i = i // 3, i % 3
                img_ = ToImg(img_fake[i])
                img.paste(img_, (w * w_i, h * h_i))
            img.save(ROOT.joinpath(rf"imgs\epoch{epoch}.jpg"))

        if epoch != 0 and epoch % 20 == 0:
            new_dir = ROOT.joinpath(f"epoch{epoch}")
            new_dir.mkdir(parents=True, exist_ok=True)
            torch.save(G0.state_dict(), new_dir.joinpath("G0.pt"))
            torch.save(D0.state_dict(), new_dir.joinpath("D0.pt"))
            torch.save(optimizer.state_dict(), new_dir.joinpath("optimizer.pt"))
            torch.save(scheduler_lr.state_dict(), new_dir.joinpath("scheduler_lr.pt"))

    torch.save(G0.state_dict(), ROOT.joinpath("G0.pt"))
    torch.save(D0.state_dict(), ROOT.joinpath("D0.pt"))

#performance of D on real
def check_classifier(database, D, state_dict):
    num_cls = len(database.classes)
    correct = [0] * num_cls
    sampler = DataLoader(database, batch_size=64)

    with torch.no_grad():
        for img, label in sampler:
            img, label = img.to(device), label.to(device)
            label_onehot = F.one_hot(label, num_classes=num_cls).view(-1, num_cls, 1, 1).to(torch.float32)
            img = img * 2. - 1.
            out = D([img, label_onehot])
            p = F.softmax(out[1], dim=1)
            pred = torch.argmax(p, dim=1)

            for i in range(p.shape[0]):
                if pred[i] == label[i]:
                    correct[pred[i]] += 1

    state_dict.update({"acc_D_real": correct})


def test(root):
    ROOT = Path(root)

    state_dict = {}

    #record dataset info
    my_trans = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor()
    ])
    database = datasets.ImageFolder(r".\dataset\X-SDD\datas", transform=my_trans)
    CLS_NUM = len(database.classes)

    record_dataset(database, state_dict)

    G = Generator(".\models\cFastGAN\Generator.yaml").to(device)
    D = Discriminator(".\models\cFastGAN\Discriminator.yaml").to(device)
    G.load_state_dict(torch.load(ROOT.joinpath("G.pt")))
    D.load_state_dict(torch.load(ROOT.joinpath("D.pt")))
    classifier = ResNet18(r".\models\ResNet18\ResNet18.yaml").to(device)
    classifier.load_state_dict(torch.load(r".\checkpoints\ResNet18_epoch83_acc93.68.pt"))

    #performance of D's classifier on real imgs
    check_classifier(database, D, state_dict)

    #sample
    ToImg = transforms.ToPILImage()
    cls_sz = 720
    n = cls_sz * CLS_NUM
    state_dict.update({"cls_sz": cls_sz})

    pred_cls_num = [0] * CLS_NUM
    pred_cls_num_9 = [0] * CLS_NUM
    points_9 = [None] *CLS_NUM
    G.eval(), D.eval(), classifier.eval()
    for cls in range(CLS_NUM):
        z = torch.randn([cls_sz, 256, 1, 1]).to(device)
        c = torch.tensor([cls]).to(device)
        c_onehot = F.one_hot(c.expand(cls_sz), num_classes=CLS_NUM).view(-1, CLS_NUM, 1, 1).to(torch.float32)
        z = torch.cat([z, c_onehot], dim=1)

        sampler = DataLoader(z, batch_size=36, shuffle=False)
        for batchi, x in enumerate(sampler):
            with torch.no_grad():
                img_fake = (G(x) + 1.) / 2.

                out = D([img_fake * 2. - 1., c_onehot[0:img_fake.shape[0]]])
                logits = out[1]
                p = F.softmax(logits, dim=1)
                correct = torch.sum(torch.eq(torch.argmax(p, dim=1), c.expand(p.shape[0])))
                # print(f"D's classifier on generated imgs!!!  cls:{cls}, ttl:{x.shape[0]}, correct:{correct}")

                logits, _, points = classifier(img_fake)
                p = F.softmax(logits, dim=1)
                pred = torch.argmax(p, dim=1)

                arg_9 = []
                for i in range(pred.shape[0]):
                    if pred[i] == cls:
                        pred_cls_num[cls] += 1
                        if p[i][cls] >= 0.8:
                            pred_cls_num_9[cls] += 1
                            arg_9.append(i)

                if points_9[cls] is not None:
                    points_9[cls] = np.concatenate((points_9[cls], points[arg_9].cpu().numpy()), axis=0)
                else:
                    points_9[cls] = points[arg_9].cpu().numpy()

                if True:
                    batchsz, _, h, w = img_fake.shape
                    imgsz = 6
                    img = PIL.Image.new("RGB", (w * imgsz, h * imgsz))
                    for i, i_ in enumerate(arg_9):
                        h_i, w_i = i // imgsz, i % imgsz
                        img_ = ToImg(img_fake[i_])
                        img.paste(img_, (w * w_i, h * h_i))
                    img.save(ROOT.joinpath(rf"test_imgs\cls_{cls}_{batchi}.jpg"))

    state_dict.update({"pred": pred_cls_num, "pred_.9": pred_cls_num_9})
    print_dict(state_dict)

    for i in range(CLS_NUM):
        print(f"cls:{i}, shape:{points_9[i].shape}")

    torch.save(state_dict, ROOT.joinpath("test_info.pt"))
    torch.save({"points": points_9}, ROOT.joinpath("fake_points.pt"))

#plot results
def analyze_plot(root):
    import matplotlib.pyplot as plt

    ROOT = Path(root)

    dict_real = torch.load(ROOT.joinpath("real_points.pt"))
    dict_fake = torch.load(ROOT.joinpath("fake_points.pt"))
    test_info = torch.load(ROOT.joinpath("test_info.pt"))
    print_dict(test_info)

    classes = test_info["dataset_info"]
    cls_num = len(classes)
    cls_sz = test_info["cls_sz"]
    pred, pred_9 = test_info["pred"], test_info["pred_.9"]

    embed_real = dict_real["points"]
    embed_fake = dict_fake["points"]

    plt.figure(figsize=(4.8*3, 3.6*3))
    for i in range(cls_num):
        print(f"{classes[i][0]}, ttl:{embed_fake[i].shape}")

        x_real, y_real = embed_real[i][:, 0], embed_real[i][:, 1]

        plt.subplot(3, 3, i + 1)
        plt.scatter(x_real, y_real, c="blue", marker=".", alpha=0.7, label="real", zorder=2)

        flag = embed_fake[i].shape[0] > 0
        if flag:
            x_fake, y_fake = embed_fake[i][:, 0], embed_fake[i][:, 1]
            plt.scatter(x_fake, y_fake, c="red", marker=".", alpha=0.2, label="fake", zorder=1)
            plt.title(f"{classes[i][0]}, ttl:{pred[i] / cls_sz * 100:.1f}%, "
                                                    f"≥.8:{pred_9[i] / pred[i] * 100:.1f}%")
        else:
            plt.title(f"{classes[i][0]}, ttl:{pred[i] / cls_sz * 100:.1f}%, ≥.8:0.0%")

        plt.xticks([]), plt.yticks([])
        plt.legend(loc="upper left")

    plt.show()


if __name__ == "__main__":
    seed_everything(0)
    train(root=r".\checkpoints\tmp8", ckpt_dir=None, restart=False)
    # test(root=r".\checkpoints\tmp1")
    # analyze_plot(root=r".\checkpoints\tmp")
    os.system("shutdown -s -t 10")
    # E = ResNet18(ENCODER_YAML).to(device)
    # E.load_state_dict(torch.load(r".\checkpoints\Encoder_epoch93_acc96.17.pt"))
    # E = list(E.children())[0]
    # E = nn.Sequential(*E.children())
    # print(E)
    # E(torch.randn([3, 3, 128, 128], device=device))
