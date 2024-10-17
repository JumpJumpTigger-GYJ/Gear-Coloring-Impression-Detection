import torch
import winnt
from torch.utils.data import Subset
from header import *
from utills import *
from Modules import *

SimSiam_YAML = r".\models\SN\SimSiam.yaml"

class SimSiam(BasicModule):
    def __init__(self, yml, trans):
        super(SimSiam, self).__init__()
        self.yml = yml
        self.parse_model(yml)
        self.trans = trans

    def forward(self, inputs):
        trans = self.trans
        x0, x1 = trans(inputs), trans(inputs)
        z0, z1 = self.backbone(x0), self.backbone(x1)
        z0, z1 = self.projection(z0), self.projection(z1)
        p0, p1 = self.prediction(z0), self.prediction(z1)
        return (p0, z0), (p1, z1)

    def loss(self, p, z):
        z = z.detach()
        p = p / torch.norm(p, dim=1, keepdim=True)
        z = z / torch.norm(z, dim=1, keepdim=True)
        return -torch.mean(torch.sum(p * z, dim=1))


def train(root, dirname="tmp", ckpt_dir=None, restart=False):
    ROOT = Path(root) / dirname
    ROOT.mkdir(parents=True, exist_ok=True)

    my_trans = transforms.Compose([
        transforms.RandomResizedCrop(128, (.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(.4, .4, .4, .1)], p=.8),
        transforms.GaussianBlur(3 ,(.1, 2.)),
    ])

    database = datasets.ImageFolder(r".\dataset\X-SDD\datas",
                                    transform=transforms.Compose([
                                                transforms.Resize([128, 128]),
                                                transforms.ToTensor()]))

    BATCHSIZE = 16
    LEARNING_RATE = 0.05 * BATCHSIZE / 256
    EPOCH = 160

    train = DataLoader(database, batch_size=BATCHSIZE, shuffle=True)

    m = SimSiam(SimSiam_YAML, my_trans).to(device)

    optimizer = optim.Adam(m.parameters(), lr=LEARNING_RATE)
    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=0.)

    if restart:
        ckpt_dir = ROOT.joinpath(ckpt_dir)
        m.load_state_dict(torch.load(ckpt_dir.joinpath("SimSiam.pt")))
        optimizer.load_state_dict(torch.load(ckpt_dir.joinpath("optimizer.pt")))
        scheduler_lr.load_state_dict(torch.load(ckpt_dir.joinpath("scheduler_lr.pt")))
        EPOCH_START = int(ckpt_dir.name.split("epoch")[-1]) + 1

    for epoch in range(EPOCH_START if restart else 0, EPOCH):
        train_(m)

        for batch_i, (imgs, labels) in enumerate(train):
            imgs, labels = imgs.to(device), labels.to(device)

            (p0, z0), (p1, z1) = m(imgs)

            loss = (m.loss(p0, z1) + m.loss(p1, z0)) / 2.

            if batch_i % 20 == 0:
                print(f"epoch:{epoch}, batch:{batch_i}, loss:{loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler_lr.step()

        if epoch != 0 and epoch % 20 == 0:
            new_dir = ROOT.joinpath(f"epoch{epoch}")
            new_dir.mkdir(parents=True, exist_ok=True)
            torch.save(m.state_dict(), new_dir.joinpath("SimSiam.pt"))
            torch.save(optimizer.state_dict(), new_dir.joinpath("optimizer.pt"))
            torch.save(scheduler_lr.state_dict(), new_dir.joinpath("scheduler_lr.pt"))

    torch.save(m.state_dict(), ROOT.joinpath("SimSiam.pt"))


def test_plot(root, dirname="tmp", ckpt_dir=""):
    print("-" * 10 + "test begin" + "-" * 10)

    ROOT = Path(root) / dirname / ckpt_dir

    database = datasets.ImageFolder(r".\dataset\X-SDD\datas",
                                    transform=transforms.Compose([
                                        transforms.Resize([128, 128]),
                                        transforms.ToTensor()]))

    CLSNUM = len(database.classes)
    BATCHSIZE = 512
    LEARNING_RATE = 0.02
    EPOCH = 90

    m = SimSiam(SimSiam_YAML, transforms.ToTensor()).to(device)
    m.load_state_dict(torch.load(ROOT / "SimSiam.pt"))
    encoder = m.backbone

    with torch.no_grad():
        embeddings = [(encoder(sample[0]), sample[1]) for sample in database]

    train_set, test_set = torch.utils.data.random_split(embeddings, [0.7, 0.3])
    train = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True)
    test = DataLoader(test_set, batch_size=BATCHSIZE, shuffle=False)

    classifier = nn.Sequential(
        nn.Linear(256, CLSNUM)
    )
    classifier.to(device)

    optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=0.)

    for epoch in range(EPOCH):
        train_(classifier)

        for batch_i, (x, labels) in enumerate(train):
            x, labels = x.to(device), labels.to(device)
            logits = classifier(x)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler_lr.step()

        eval_(classifier)
        correct = torch.zeros([CLSNUM], device=device)
        total = torch.zeros([CLSNUM], device=device)
        with torch.no_grad():
            for x, labels in test:
                x, labels = x.to(device), labels.to(device)
                logits = classifier(x)
                pred = torch.argmax(logits, dim=1)

                for i in range(labels.shape[0]):
                    label = labels[i]
                    total[label] += 1
                    if label == pred[i]:
                        correct[label] += 1

        acc = correct / total * 100.
        print(f"epoch:{epoch}, ", *[f"{database.classes[i]}:{acc[i]:.2f}%" for i in range(CLSNUM)])


if __name__ == "__main__":
    seed_everything(0)
    # train(root=r".\checkpoints", dirname="SimSiam", ckpt_dir=None, restart=False)
