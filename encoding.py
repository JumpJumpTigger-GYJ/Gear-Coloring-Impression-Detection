import torch

from header import *
from utills import *
from ResNet import ResNet18
import matplotlib.pyplot as plt

if __name__ == "__main__":
    seed_everything(0)

    #load data
    train_data = datasets.ImageFolder(r".\dataset\X-SDD\datas", transform=transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor()
    ]))
    loader = DataLoader(train_data, batch_size=128, shuffle=False)
    classes = len(train_data.classes)
    print(f"sample.shape:{train_data[0][0].shape}, num_classes:{classes}")

    #load model
    model = ResNet18(r".\models\ResNet18\ResNet18.yaml").to(device)
    model.load_state_dict(torch.load(r".\checkpoints\ResNet18_epoch83_acc93.68.pt"))

    points = None
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            if points is not None:
                points = np.concatenate((points, model.encode_forward(x).cpu().numpy()), axis=0)
            else:
                points = model.encode_forward(x).cpu().numpy()
    print(f"points.shape:{points.shape}")

    state_dict = {"points": []}
    sample_nums = np.bincount(train_data.targets, minlength=classes)
    colors = ["gold", "darkcyan", "darkgreen", "black", "red", "darkorange", "blue"]
    zorder = [1, 2, 3, 1, 1, 4, 1]
    plt.figure(figsize=(12, 9))
    for i in range(classes):
        st = sample_nums[i - 1] if i else 0
        ed = sample_nums[i] + (sample_nums[i - 1] if i else 0)
        x, y = points[st:ed, 0], points[st:ed, 1]
        plt.scatter(x, y, c=colors[i], alpha=0.7, label=f"cls:{i}", marker='.', zorder=zorder[i])
        state_dict["points"].append(points[st:ed])
    plt.title("scatter plot")
    plt.legend(loc="upper right")
    plt.xticks([])
    plt.yticks([])
    plt.show()

    for i, point in enumerate(state_dict["points"]):
        print(f"cls:{i}, points.shape:{point.shape}")
    torch.save(state_dict, r".\checkpoints\tmp\real_points.pt")
