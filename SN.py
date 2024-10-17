import torch
from torch.utils.data import Subset
from header import *
from utills import *
from Modules import *
from torch.utils.data import SubsetRandomSampler

SN_YAML = r".\models\SN\SN_Patch.yaml"
SNCls_YAML = r".\models\SN\classifier.yaml"


class SN(BasicModule):
    def __init__(self, yml):
        super(SN, self).__init__()
        self.yml = yml
        self.parse_model(self.yml)

    def forward(self, inputs):
        embd = super(SN, self).forward(inputs)
        return embd

    def contrastive_loss(self, embd0, embd1, similar:bool, margin=2.):
        d = F.pairwise_distance(embd0, embd1)
        if similar:
            loss = d
        else:
            loss = torch.max(torch.zeros([], device=device), margin - d)
        return torch.mean(loss)


class SNClassifier(BasicModule):
    #similar -> 0, dissimilar -> 1

    def __init__(self, yml):
        super(SNClassifier, self).__init__()
        self.yml = yml
        self.parse_model(yml)

    def forward(self, embd0, embd1):
        x = torch.cat([embd0, embd1], dim=1)
        return super(SNClassifier, self).forward(x)

    def loss4G(self, logits, similar:bool):
        if similar:
            return -torch.mean(torch.min(torch.zeros([], device=device), -1. - logits))
        else:
            return -torch.mean(torch.min(torch.zeros([], device=device), -1. + logits))

    def loss(self, logits, similar:bool):
        if similar:
            labels = torch.zeros([logits.shape[0], 1], device=device)
        else:
            labels = torch.ones([logits.shape[0], 1], device=device)
        return F.binary_cross_entropy_with_logits(logits, labels)


class ClassSampler:
    def __init__(self, database):
        self.database = database
        num_classes = np.bincount(database.targets)
        st_classes = np.zeros_like(num_classes)
        self.n = num_classes.shape[0]
        for i in range(1, st_classes.shape[0]):
            st_classes[i] = st_classes[i - 1] + num_classes[i - 1]
        self.num_classes = num_classes.tolist()
        self.st_classes = st_classes.tolist()
        self.samplers = [SubsetRandomSampler(list(range(self.st_classes[i], self.st_classes[i] + self.num_classes[i])))
                         for i in range(len(self.num_classes))]

    def __sample(self, c):
        while True:
            for i in self.samplers[c]:
                return self.database[i][0]

    def cls_sample(self, classes):
        return torch.stack([self.__sample(c) for c in classes], dim=0)

    def cls_inv_sample(self, classes):
        classes = classes + torch.randint(1, self.n - 1, [classes.shape[0]], device=device)
        classes = classes % self.n
        return self.cls_sample(classes)


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
    EPOCH = 100

    sampler = get_WeightedRandomSampler(Subset(database, list(range(len(database)))))
    train = DataLoader(database, batch_size=BATCHSIZE // 2, sampler=sampler)

    MARGIN = 4.
    sn = SN(SN_YAML).to(device)
    snc = SNClassifier(SNCls_YAML).to(device)

    params_group = [
        {"params": sn.parameters(), "weight_decay": 1e-3},
        {"params": snc.parameters(), "weight_decay": 1e-2}
    ]
    optimizer = optim.Adam(params_group, lr=LEARNING_RATE)
    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=0.)

    if restart:
        ckpt_dir = ROOT.joinpath(ckpt_dir)
        sn.load_state_dict(torch.load(ckpt_dir.joinpath("SN.pt")))
        snc.load_state_dict(torch.load(ckpt_dir.joinpath("SNC.pt")))
        optimizer.load_state_dict(torch.load(ckpt_dir.joinpath("optimizer.pt")))
        scheduler_lr.load_state_dict(torch.load(ckpt_dir.joinpath("scheduler_lr.pt")))

        EPOCH_START = int(ckpt_dir.name.split("epoch")[-1]) + 1

    ToImg = transforms.ToPILImage()
    clsSampler = ClassSampler(database)
    clsSampler_test = ClassSampler(database)

    for epoch in range(EPOCH_START if restart else 0, EPOCH):
        train_(sn, snc)

        for batch_i, (img, label) in enumerate(train):
            img, label = img.to(device), label.to(device)
            img = img * 2. - 1.
            batchsz = img.shape[0]

            #SN
            img_sm = clsSampler.cls_sample(label)
            img_sm = img_sm.to(device)
            img_sm = img_sm * 2. - 1.

            t = sn(img)
            t_sm = sn(img_sm)
            loss_sm = sn.contrastive_loss(t, t_sm, similar=True, margin=MARGIN)

            img_div = clsSampler.cls_inv_sample(label)
            img_div = img_div.to(device)
            img_div = img_div * 2. - 1.

            t_div = sn(img_div)
            loss_div = sn.contrastive_loss(t, t_div, similar=False, margin=MARGIN)

            optimizer.zero_grad()
            loss_sm.backward(retain_graph=True)
            loss_div.backward()
            optimizer.step()

            if batch_i % 20 == 0:
                print(f"SN: sm:{loss_sm.item():.2f}, div:{loss_div.item():.2f}")

            #Classifier
            t, t_sm, t_div = t.detach(), t_sm.detach(), t_div.detach()

            logits_sm = snc(t, t_sm)
            logits_div = snc(t, t_div)
            loss_c_sm = snc.loss(logits_sm, similar=True)
            loss_c_div = snc.loss(logits_div, similar=False)

            optimizer.zero_grad()
            loss_c_sm.backward(retain_graph=True)
            loss_c_div.backward()
            optimizer.step()

            if batch_i % 20 == 0:
                print(f"SNClassifier: sm:{loss_c_sm.item():.2f}, div:{loss_c_div.item():.2f}")

        scheduler_lr.step()

        eval_(sn, snc)
        with torch.no_grad():
            ttl = BATCHSIZE * 2
            c = torch.randint(0, CLS_NUM - 1, [ttl], device=device)
            img = clsSampler_test.cls_sample(c)
            img = img.to(device)
            img = img * 2. - 1.
            img_sm = clsSampler_test.cls_sample(c)
            img_sm = img_sm.to(device)
            img_sm = img_sm * 2. - 1.
            img_div = clsSampler_test.cls_inv_sample(c)
            img_div = img_div.to(device)
            img_div = img_div * 2. - 1.

            t, t_sm, t_div = sn(img), sn(img_sm), sn(img_div)
            logits_sm, logits_div = snc(t, t_sm), snc(t, t_div)
            acc_sm = torch.sum(logits_sm < 0.) / ttl
            acc_div = torch.sum(logits_div > 0.) / ttl

            print(f"epoch {epoch}: acc_sm:{acc_sm * 100:.2f}%, acc_div:{acc_div * 100:.2f}%")

        if epoch != 0 and epoch % 20 == 0:
            new_dir = ROOT.joinpath(f"epoch{epoch}")
            new_dir.mkdir(parents=True, exist_ok=True)
            torch.save(sn.state_dict(), new_dir.joinpath("SN.pt"))
            torch.save(snc.state_dict(), new_dir.joinpath("SNC.pt"))
            torch.save(optimizer.state_dict(), new_dir.joinpath("optimizer.pt"))
            torch.save(scheduler_lr.state_dict(), new_dir.joinpath("scheduler_lr.pt"))

    torch.save(sn.state_dict(), ROOT.joinpath("SN.pt"))
    torch.save(snc.state_dict(), ROOT.joinpath("SNC.pt"))


def _test(sn_yaml, snc_yaml, path, database):
    ROOT = Path(path)
    sn = SN(sn_yaml).to(device)
    snc = SNClassifier(snc_yaml).to(device)
    sn.load_state_dict(torch.load(ROOT.joinpath("SN.pt")))
    snc.load_state_dict(torch.load(ROOT.joinpath("SNC.pt")))
    eval_(sn, snc)

    n = len(database)
    epochs = 3
    ttl = n * epochs

    loader = DataLoader(database, batch_size=32, shuffle=False)
    sampler = ClassSampler(database)

    correct_sm = 0
    correct_div = 0

    for epoch in range(epochs):
        for img, label in loader:
            img, label = img.to(device), label.to(device)
            img = img * 2. - 1.

            img_sm = sampler.cls_sample(label).to(device)
            img_sm = img_sm * 2. -1.
            img_div = sampler.cls_inv_sample(label).to(device)
            img_div = img_div * 2. - 1.

            with torch.no_grad():
                t, t_sm, t_div = sn(img), sn(img_sm), sn(img_div)
                logits_sm = snc(t, t_sm)
                logits_div = snc(t, t_div)

                correct_sm += torch.sum(logits_sm < 0.).item()
                correct_div += torch.sum(logits_div > 0.).item()

    print(f"path:{path}, acc_sm:{correct_sm / ttl * 100:.2f}%, acc_div:{correct_div / ttl * 100:.2f}%")


def test():
    my_trans = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor()
    ])
    database = datasets.ImageFolder(r".\dataset\X-SDD\datas", transform=my_trans)

    yamls = [
        (r".\models\SN\SN.yaml", r".\models\SN\classifier.yaml"),
        (r".\models\SN\SN.yaml", r".\models\SN\classifier.yaml"),
        (r".\models\SN\SN_Patch.yaml", r".\models\SN\classifier.yaml")
    ]

    paths = [
        r".\checkpoints\SN",
        r".\checkpoints\SN0",
        r".\checkpoints\SN_P"
    ]

    for (sn_yaml, snc_yaml), path in zip(yamls, paths):
        _test(sn_yaml, snc_yaml, path, database)


if __name__ == "__main__":
    seed_everything(0)
    # train(root=r".\checkpoints\SN_P", ckpt_dir=None, restart=False)
    test()
    # analyze_plot(root=r".\checkpoints\tmp")
    # os.system("shutdown -s -t 10")