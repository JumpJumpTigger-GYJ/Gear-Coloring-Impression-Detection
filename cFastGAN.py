import torch
from torch.utils.data import Subset
from header import *
from utills import *
from Modules import *
from ResNet import ResNet18

G_YAML = r".\models\cFastGAN\Generator.yaml"
D_YAML = r".\models\cFastGAN\Discriminator_patch.yaml"

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
        return F.cross_entropy(input, label, label_smoothing=0.1)

    def loss_embeddings(self, input):
        return torch.mean(torch.norm(input[0] - input[1], dim=1))


class Generator(BasicModule):
    def __init__(self, yml):
        super(Generator, self).__init__()
        self.yml = yml
        self.parse_model(yml)

    def forward(self, input):
        return super(Generator, self).forward(input)


def train(root, ckpt_dir=None, restart=False):
    ROOT = Path(root)

    my_trans = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor()
    ])
    database = datasets.ImageFolder(r".\dataset\X-SDD\datas", transform=my_trans)

    CLS_NUM = len(database.classes)
    BATCHSIZE = 32
    LEARNING_RATE = 2e-4
    EPOCH = 200

    # sampler = get_WeightedRandomSampler(Subset(database, list(range(len(database)))))
    # train = DataLoader(database, batch_size=BATCHSIZE // 2, sampler=sampler)
    train = DataLoader(database, batch_size=BATCHSIZE // 2, shuffle=True)

    G = Generator(G_YAML).to(device)
    D = Discriminator(D_YAML).to(device)
    params_group = [
        {"params": G.parameters()},
        {"params": D.parameters()}
    ]
    optimizer = optim.Adam(params_group, lr=LEARNING_RATE)
    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=0.)

    if restart:
        ckpt_dir = ROOT.joinpath(ckpt_dir)
        G.load_state_dict(torch.load(ckpt_dir.joinpath("G.pt")))
        D.load_state_dict(torch.load(ckpt_dir.joinpath("D.pt")))
        optimizer.load_state_dict(torch.load(ckpt_dir.joinpath("optimizer.pt")))
        scheduler_lr.load_state_dict(torch.load(ckpt_dir.joinpath("scheduler_lr.pt")))

        EPOCH_START = int(ckpt_dir.name.split("epoch")[-1]) + 1

    ToImg = transforms.ToPILImage()

    for epoch in range(EPOCH_START if restart else 0, EPOCH):
        D.train(), G.train()

        for batch_i, (img_real, label_real) in enumerate(train):
            img_real, label_real = img_real.to(device), label_real.to(device)
            img_real = img_real * 2. - 1.
            batchsz = img_real.shape[0]
            onehot_real = F.one_hot(label_real, num_classes=CLS_NUM).to(torch.float32)

            #disriminator
            freeze(G)

            label_fake = torch.randint(0, CLS_NUM - 1, size=[batchsz], device=device)
            onehot_fake = F.one_hot(label_fake, num_classes=CLS_NUM).to(torch.float32)
            z = torch.randn([batchsz, 256], device=device)
            z = torch.cat([z, onehot_fake], dim=1)
            img_fake = G(z)

            out_fake = D([img_fake, onehot_fake])
            out_real = D([img_real, onehot_real])

            loss_D_real = D.loss_real(out_real[0])
            loss_D_fake = D.loss_fake(out_fake[0])
            loss_D_cls = D.loss_classifier(out_real[1], label_real)
            loss_D_recons = D.loss_recons(out_real[2]) + D.loss_recons(out_real[3])

            loss = loss_D_real + loss_D_fake + loss_D_cls + loss_D_recons

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            unfreeze(G)

            if batch_i % 20 == 0:
                print(f"D: loss_real:{loss_D_real:.2f}, loss_fake:{loss_D_fake:.2f}, "
                      f"loss_recons:{loss_D_recons:.2f}, loss_cls:{loss_D_cls:.2f}")

            #generator
            freeze(D)

            label_fake = torch.randint(0, CLS_NUM - 1, size=[BATCHSIZE], device=device)
            onehot_fake = F.one_hot(label_fake, num_classes=CLS_NUM).to(torch.float32)
            z = torch.randn([BATCHSIZE, 256], device=device)
            z = torch.cat([z, onehot_fake], dim=1)

            img_fake = G(z)
            out = D([img_fake, onehot_fake])

            loss_G = -torch.mean(out[0])
            loss_G_cls = D.loss_classifier(out[1], label_fake)

            loss = loss_G + loss_G_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            unfreeze(D)

            if batch_i % 20 == 0:
                print(f"G: loss_G:{loss_G:.2f}, loss_cls:{loss_G_cls:.2f}")

        scheduler_lr.step()

        D.eval(), G.eval()
        with torch.no_grad():
            c = torch.arange(0, CLS_NUM, device=device).view(-1, 1)
            c = torch.expand_copy(c, size=[CLS_NUM, 2]).view(-1)
            c = torch.cat([c, torch.randint(0, CLS_NUM - 1, size=[16 - c.shape[0]], device=device)], dim=0)
            onehot_c = F.one_hot(c, num_classes=CLS_NUM).to(torch.float32)
            z = torch.randn([16, 256], device=device)
            z = torch.cat([z, onehot_c], dim=1)
            img_fake = (G(z) + 1.) / 2.
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
            torch.save(optimizer.state_dict(), new_dir.joinpath("optimizer.pt"))
            torch.save(scheduler_lr.state_dict(), new_dir.joinpath("scheduler_lr.pt"))

    torch.save(G.state_dict(), ROOT.joinpath("G.pt"))
    torch.save(D.state_dict(), ROOT.joinpath("D.pt"))

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
    train(root=r".\checkpoints\tmp7", ckpt_dir="epoch120", restart=True)
    # test(root=r".\checkpoints\tmp1")
    # analyze_plot(root=r".\checkpoints\tmp")
    # os.system("shutdown -s -t 10")

