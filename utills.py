import torch.utils.data
from torch.utils.data import WeightedRandomSampler
from header import *

device = torch.device("cuda")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_dict(d):
    for key, value in d.items():
        print(f"{key}: {value}")

def freeze(*models):
    for model in models:
        for param in model.parameters():
            param.requires_grad = False

def unfreeze(*models):
    for model in models:
        for param in model.parameters():
            param.requires_grad = True

def train_(*models):
    for model in models:
        model.train()

def eval_(*models):
    for model in models:
        model.eval()

def norm(inputs):
    batch = inputs.shape[0]
    res = torch.zeros([]).to(device)
    for i in range(batch):
        res = res + torch.norm(inputs[i])
    res = res / batch
    return res

#record dataset info
def record_dataset(dataset, state_dict):
    classes = len(dataset.classes)
    num_classes = [0] * classes
    for _, cls in dataset:
        num_classes[cls] += 1

    l = [(dataset.classes[i], num_classes[i]) for i in range(classes)]
    state_dict.update({"dataset_info": l})
    return state_dict

def load_data(path, show=False, transforms=None, length=(0.8, 0.2)):
    database = datasets.ImageFolder(path, transforms)

    if show:
        img = database[0][0]
        _, h, w = img.shape
        img = PIL.Image.new("RGB", (w * 3, h * 3))
        for i in range(9):
            h_i, w_i = i // 3, i % 3
            src_i = random.randint(0, len(database) - 1)
            img_ = database[src_i][0].permute(2, 1, 0).contiguous().numpy() * 255
            img_ = img_.astype(np.uint8)
            img_ = PIL.Image.fromarray(img_, "RGB")
            img.paste(img_, (w * w_i, h * h_i))
        img.show()

    train, test = torch.utils.data.random_split(database, lengths=length)
    return train, test, len(database.classes)

#for imbalanced subset
def get_WeightedRandomSampler(subset):
    subset_class_idx = [subset.dataset.targets[i] for i in subset.indices]
    class_weight = np.round(100. / np.bincount(subset_class_idx), decimals=2)
    subset_weight = [class_weight[i] for i in subset_class_idx]
    return WeightedRandomSampler(subset_weight, num_samples=len(subset_weight), replacement=True)

def count_sample_num(subset):
    subset_class_idx = [subset.dataset.targets[i] for i in subset.indices]
    sample_num = torch.from_numpy(np.bincount(subset_class_idx)).to(torch.int64)
    return sample_num