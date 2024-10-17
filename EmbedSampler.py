from torch.utils.data import Subset
from header import *
from utills import *
from Modules import *
from ResNet import ResNet18

RESNET_YAML = r".\models\ResNet18\ResNet18.yaml"
classifier = ResNet18(RESNET_YAML).to(device)
classifier.load_state_dict(torch.load(r".\checkpoints\ResNet18_epoch83_acc93.68.pt"))
classifier.eval(), freeze(classifier)

my_trans = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor()
    ])
database = datasets.ImageFolder(r".\dataset\X-SDD\datas", transform=my_trans)

class EmbedSampler:
    def __init__(self, model, database):
        self.samples = self.get_class_embeddings(model, database)
        subset_class_idx = [database.targets[i] for i in range(len(database))]
        class_weight = np.round(100. / np.bincount(subset_class_idx), decimals=2)
        subset_weight = [class_weight[i] for i in subset_class_idx]
        self.sampler = WeightedRandomSampler(subset_weight, num_samples=len(subset_weight), replacement=True)

    def get_class_embeddings(self, model, database, out_i=-1):
        model.eval()

        loader = DataLoader(database, batch_size=128, shuffle=False)
        res = None
        with torch.no_grad():
            for x, label in loader:
                x = x.to(device)
                embds = model(x)[out_i]
                if res is not None:
                    res = torch.cat([res, embds], dim=0)
                else:
                    res = embds

        return list(zip(torch.split(res, 1, dim=0), database.targets))

    def sample(self, sz):
        indices = []
        while True:
            for x, y in self.sampler:
                yield x, y

if __name__ == "__main__":
    embder = EmbedSampler(classifier, database)
    for i in range(20):
        x, y = embder.sample(4)
        print(y)