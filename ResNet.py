import numpy as np
import torch

from header import *
from Modules import *
from utills import *


class ResNet18(BasicModule):
    def __init__(self, yml):
        super(ResNet18, self).__init__()
        self.yml = yml
        self.parse_model(yml)

    def forward(self, inputs):
        out = super(ResNet18, self).forward(inputs)
        return out

    def classify_forward(self, inputs):
        out = self.forward(inputs)
        return out[0]

    def embeddings_forward(self, inputs):
        out = self.forward(inputs)
        return out[1]

    def encode_forward(self, inputs):
        out = self.forward(inputs)
        return out[2]


class Criterion:
    def __init__(self):
        super(Criterion, self).__init__()

    def loss_cls(self, inputs, targets):
        return F.cross_entropy(inputs, targets)

    def loss_recons(self, inputs):
        return torch.mean(torch.norm(inputs[0] - inputs[1], dim=1))


def main():
    #load data
    my_transftorm = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomRotation(45),
        transforms.ToTensor()
    ])

    train_data, test_data, classes = load_data(r".\dataset\X-SDD\datas",
                                                             transforms=my_transftorm, show=False)
    train_sample_num = count_sample_num(train_data)
    test_sample_num = count_sample_num(test_data)
    print(f"train_sample_num:{train_sample_num}")
    print(f"test_sample_num:{test_sample_num}")

    sampler = get_WeightedRandomSampler(train_data)
    train = DataLoader(train_data, batch_size=32, sampler=sampler)
    test = DataLoader(test_data, batch_size=32, shuffle=False)

    # x, label = next(iter(train))
    # print("x: ", x.shape, "label:", label.shape, "classes:", classes)

    #train
    model = ResNet18(r".\models\ResNet18\encoder.yaml")
    model.to(device)
    # print(model)

    EPOCH = 100
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = Criterion()
    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=0.)

    best_acc = torch.zeros([])

    # viz = Visdom()
    # viz.line([[2., 0.]], [0.], win="training", opts=dict(title="training",
    #                                                      legend=["training loss", "test_acc"]))

    for epoch in range(EPOCH):
        model.train()

        for batchidx, (x, label) in enumerate(train):
            x, label = x.to(device), label.to(device)
            x = x * 2. - 1.

            out = model(x)
            loss_cls = criterion.loss_cls(out, label)
            # loss_recons = criterion.loss_recons(out[1])
            loss = loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler_lr.step()

        #test
        model.eval()
        test_sample_correct = torch.zeros(classes, dtype=torch.int64)
        total_correct = 0
        total_num = 0
        for x, label in test:
            x, label = x.to(device), label.to(device)
            x = x * 2. - 1.
            logits = model(x)
            pred = logits.argmax(1)
            correct = torch.eq(pred, label).float().sum().item()
            for i in range(logits.shape[0]):
                if pred[i] == label[i]:
                    test_sample_correct[label[i]] += 1
            total_correct += correct
            total_num += x.size()[0]
        acc = total_correct / total_num

        print(f"epoch:{epoch}, loss_cls:{loss_cls:.2f}, test_acc:{acc * 100:.2f}%")
        percent = test_sample_correct / test_sample_num * 100.
        avg_acc = torch.mean(percent, dim=0)
        print("correct:{}, avg_acc:{:.2f}%".format([f"{percent[i]:.2f}%" for i in range(classes)], avg_acc))
        # print(f"epoch:{epoch}, loss_cls:{loss_cls:.2f}, test_acc:{acc * 100:.2f}%")

        if epoch >= 75 and best_acc <= avg_acc:
            best_acc = avg_acc
            torch.save(model.state_dict(), rf".\checkpoints\tmp7\encoder_epoch{epoch}_acc{avg_acc:.2f}.pt")

        # viz.line([[total_loss, acc]], [epoch], win="training", update="append")

if __name__ == "__main__":
    seed_everything(0)
    main()
    # from cFastGAN_1 import *
    # m = Generator(G_YAML).to(device)
    # m.load_state_dict(torch.load(r".\checkpoints\1e-4_200\epoch120\G.pt"))
    # # train_data = datasets.ImageFolder(r".\dataset\X-SDD\datas", transform=transforms.Compose([
    # #     transforms.Resize([128, 128]),
    # #     transforms.ToTensor()
    # # ]))
    # CLS_NUM = 7
    # cls_sz = 200
    # n = cls_sz * CLS_NUM
    # m.eval()
    # for cls in range(CLS_NUM):
    #     z = torch.randn([cls_sz, 256, 1, 1]).to(device)
    #     c = torch.tensor([cls]).to(device)
    #     c_onehot = F.one_hot(c.expand(cls_sz), num_classes=CLS_NUM).view(-1, CLS_NUM, 1, 1).to(torch.float32)
    #     z = torch.cat([z, c_onehot], dim=1)
    #     with torch.no_grad():
    #         f_l = m(z)[1:]
    #         for i, f in enumerate(f_l):
    #             mean = torch.mean(f, dim=0)
    #             var = torch.mean(torch.linalg.norm(f - mean, dim=[2, 3]))
    #             print(f"cls:{cls}, layer:{i}, var:{var}")
