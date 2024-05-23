import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import os


np.random.seed(123)

dataDir = "./data/CUB_200_2011/"


# 设置数据转换
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class CUB200(Dataset):
    def __init__(self, root_dir, subset="train", transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            subset (string): 'train', 'test', or 'val' to specify which subset of the data to load.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = pd.read_csv(
            os.path.join(root_dir, "images.txt"), sep=" ", names=["img_id", "filename"]
        )
        self.labels = pd.read_csv(
            os.path.join(root_dir, "image_class_labels.txt"),
            sep=" ",
            names=["img_id", "label"],
        )
        self.split = pd.read_csv(
            os.path.join(root_dir, "train_test_split.txt"),
            sep=" ",
            names=["img_id", "is_train"],
        )

        # Merge all information into a single DataFrame
        self.data = pd.merge(self.images, self.labels, on="img_id")
        self.data = pd.merge(self.data, self.split, on="img_id")

        # Filter data for training, testing or validation
        if subset == "train":
            self.data = self.data[self.data["is_train"] == 1]
        else:
            self.data = self.data[self.data["is_train"] == 0]
            # test_val_data = self.data[self.data["is_train"] == 0]
            # indices = np.arange(test_val_data.shape[0])
            # np.random.shuffle(indices)  # Shuffle indices to randomize test/val split
            # split_point = int(len(indices) / 2)
            # if subset == "test":
            #     test_indices = indices[:split_point]
            #     self.data = test_val_data.iloc[test_indices]
            # elif subset == "val":
            #     val_indices = indices[split_point:]
            #     self.data = test_val_data.iloc[val_indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, "images", self.data.iloc[idx, 1])
        image = Image.open(img_name).convert("RGB")
        label = self.data.iloc[idx, 2] - 1  # Adjust label to be zero-indexed

        if self.transform:
            image = self.transform(image)

        return image, label


train_dataset = CUB200(root_dir=dataDir, subset="train", transform=transform)
test_dataset = CUB200(root_dir=dataDir, subset="test", transform=transform)
val_dataset = CUB200(root_dir=dataDir, subset="val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)


from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.tensorboard import SummaryWriter
import pickle


class BirdClassificationCNN(nn.Module):
    def __init__(self, use_pretrained=True, num_classes=200):
        super(BirdClassificationCNN, self).__init__()
        if use_pretrained:
            print("Load weights from pretrained model")
            self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            print("Load weights from scratch")
            self.resnet = resnet18()

        self.use_pretrained = use_pretrained
        self.resnet.fc = nn.Linear(512, num_classes)
        # self.conv1 = nn.Conv2d(1, 3, kernel_size=1)
        self.loss_record = {
            "train": [],
            "val": [],
        }
        self.accuracy_record = {
            "train": [],
            "val": [],
        }
        self.load_data()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        return self.resnet(x)

    def load_data(self):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

    def train(
        self, epochs=50, lr=0.001, momentum=0.9, weight_decay=0.001, use_cache=False
    ):
        config_specified_name = (
            f"{lr}_{momentum}_{weight_decay}_{int(self.use_pretrained)}"
        )
        cache_path = f"./cache/model_{config_specified_name}.pt"
        if use_cache:
            self.read_model(path=cache_path)

        self.writer = SummaryWriter(f"./runs/{config_specified_name}")
        train_loader = self.train_loader

        device = self.device
        self.to(device)
        loss_function = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0001)
        optimizer = optim.SGD(
            self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        for epoch in range(epochs):
            # running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.long().to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                # running_loss += loss.item()

            torch.save(self.state_dict(), cache_path)

            # self.loss_record.append(running_loss / len(train_loader))
            eval_res = self.evaluate_on_train_and_val()
            eval_train, eval_val = eval_res["train"], eval_res["val"]
            print("Epoch: %d | Loss: %.4f" % (epoch, eval_train["loss"]))
            # running_loss = 0.0

            self.writer.add_scalar("Loss/train", eval_train["loss"], epoch)
            self.writer.add_scalar("Loss/test", eval_val["loss"], epoch)
            self.writer.add_scalar("Accuracy/train", eval_train["accuracy"], epoch)
            self.writer.add_scalar("Accuracy/test", eval_val["accuracy"], epoch)

            self.loss_record["train"].append(eval_train["loss"])
            self.loss_record["val"].append(eval_val["loss"])
            self.accuracy_record["train"].append(eval_train["accuracy"])
            self.accuracy_record["val"].append(eval_val["accuracy"])

        print("Finished Training")
        # save the loss record and accuracy record
        pickle.dump(self.loss_record, open("./cache/loss_record.pkl", "wb"))
        pickle.dump(self.accuracy_record, open("./cache/accuracy_record.pkl", "wb"))

    # 读取模型参数
    def read_model(self, path="./cache/model.pt"):
        self.load_state_dict(torch.load(path))

    # 在数据集上评估模型，返回总样本数和正确分类的样本数
    def evaluate(self, loader):
        device = self.device
        self.to(device)
        loss_function = nn.CrossEntropyLoss()

        correct = 0
        total = 0
        total_loss = 0.0
        with torch.no_grad():
            for data in loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.long().to(device)
                outputs = self(inputs)
                loss = loss_function(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        return accuracy, avg_loss

    # 在训练集和验证集上评估模型，并打印准确率
    def evaluate_on_train_and_val(self):
        train_accuracy, train_loss = self.evaluate(self.train_loader)
        val_accuracy, val_loss = self.evaluate(self.val_loader)
        print(
            "Accuracy of the network on the %d train data: %.2f %%"
            % (len(self.train_loader.dataset), 100 * train_accuracy)
        )
        print(
            "Accuracy of the network on the %d val data: %.2f %%"
            % (len(self.val_loader.dataset), 100 * val_accuracy)
        )
        return {
            "train": {"accuracy": train_accuracy, "loss": train_loss},
            "val": {"accuracy": val_accuracy, "loss": val_loss},
        }


import argparse


def main():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=0.001, help="Weight decay"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parser.add_argument("--use_cache", type=str2bool, default=False, help="Use cache")
    parser.add_argument(
        "--use_pretrained", type=str2bool, default=True, help="Use pretrained"
    )

    args = vars(parser.parse_args())
    print(args)

    use_pretrained = args.pop("use_pretrained")

    model = BirdClassificationCNN(use_pretrained=use_pretrained)

    model.train(**args)


if __name__ == "__main__":
    main()
