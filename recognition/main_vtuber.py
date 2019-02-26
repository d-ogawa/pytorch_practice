import matplotlib as mpl
mpl.use("Agg")
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from skimage import io
import numpy as np
import glob, os, datetime
from tensorboard_logger import configure, log_value
import matplotlib.pylab as plt
import seaborn as sns
sns.set(style="darkgrid")

save_log = False
train_loss = []
train_acc = []
val_loss = []
val_acc = []

# # case 1
# class myDataset(Dataset):
#     """
#     Dataset for Handmade dataset
#     Use csv data and load data every iteration
#     """
#
#     def __init__(self, csv_file_path, transform=None):
#
#         self.dataframe = pd.read_csv(csv_file_path)
#         self.transform = transform
#
#     def __len__(self):
#         return (len(self.dataframe))
#
#     def __getitem__(self, idx):
#
#         label = self.dataframe.at[idx, "label"]
#         img_path = self.dataframe.at[idx, "path_to_image"]
#
#         img = io.imread(img_path)
#         if img.ndim == 2:
#             img = np.expand_dims(img, axis=-1)          # for 2d image
#
#         if self.transform:
#             img = self.transform(img)
#
#         return (img, label)

# # case 2
# class myDataset(Dataset):
#     """
#     Dataset for Handmade dataset
#     Use pt data and load data every iteration
#     Bit faster than case 1
#     """
#
#     def __init__(self, path_to_dir, transform=None):
#
#         files = glob.glob(os.path.join(path_to_dir, "*.pt"))
#         self.files = sorted(files)
#         self.transform = transform
#
#     def __len__(self):
#         return (len(self.files))
#
#     def __getitem__(self, idx):
#
#         img, label = torch.load(self.files[idx])
#
#         if self.transform:
#             img = self.transform(img)
#
#         return (img, label)

# case 3
class myDataset(Dataset):
    """
    Dataset for Handmade dataset
    Use pt data and load every data in advance
    Much faster than case 1 and case 2
    """

    def __init__(self, path_to_dir, transform=None):

        files = sorted(glob.glob(os.path.join(path_to_dir, "*.pt")))
        data = []
        for file in files:
            data.append(torch.load(file))

        self.data = data
        self.transform = transform

    def __len__(self):
        return (len(self.data))

    def __getitem__(self, idx):

        img, label = self.data[idx]

        if self.transform:
            img = self.transform(img)

        return (img, label)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 128x128x3
        self.block1 = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=5, padding=2),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 64x64x32
        self.block2 = nn.Sequential(
                        nn.Conv2d(32, 64, kernel_size=5, padding=2),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 32x32x64
        self.block3 = nn.Sequential(
                        nn.Conv2d(64, 128, kernel_size=5, padding=2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(128, 128, kernel_size=5, padding=2),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 16x16x128
        self.block4 = nn.Sequential(
                        nn.Conv2d(128, 256, kernel_size=5, padding=2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256, 256, kernel_size=5, padding=2),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 8x8x256
        self.block5 = nn.Sequential(
                        nn.Conv2d(256, 512, kernel_size=5, padding=2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(512, 512, kernel_size=5, padding=2),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 4x4x512 = 8192
        self.classifier = nn.Sequential(
                        nn.Linear(4*4*512, 2048),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.2),
                        nn.Linear(2048, 1024),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.2),
                        nn.Linear(1024, 5),
        )

    def forward(self, x):
        # print(x.size())
        x = self.block1(x)
        # print(x.size())
        x = self.block2(x)
        # print(x.size())
        x = self.block3(x)
        # print(x.size())
        x = self.block4(x)
        # print(x.size())
        x = self.block5(x)
        # print(x.size())

        # x = x.view(x.size(0), -1)
        x = x.view(-1, 4*4*512)

        # print(x.size())
        x = self.classifier(x)
        # print(x.size())

        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_idx * len(data), len(train_loader.dataset), 100.0 * batch_idx / len(train_loader), loss.item()))

            if save_log: log_value("loss", loss.item(), (epoch - 1) * 55000 + 10 * batch_idx)

        if batch_idx % 100 == 0:
            train_loss.append(loss.item())


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target,
                                    reduction="sum").item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{}({:.0f}%)\n".format(test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)))

    val_loss.append(test_loss)
    val_acc.append(correct / len(test_loader.dataset))


def draw_graph(data, x, y, filename=None):
    # sns.lineplot(data=data, x="iterations (/100)", y="iterations (/100)")
    ax = sns.lineplot(data=data)
    ax.set(xlabel=x, ylabel=y)
    plt.savefig(filename)
    plt.clf()



def main():
    parser = argparse.ArgumentParser(description="Pytorch MNIST .")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="input batch size for training [64] .")
    parser.add_argument("--test_batch_size", type=int, default=1000,
                        help="input batch size for testing [1000] .")
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of epochs to train [10] .")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate [0.01] .")
    parser.add_argument("--momentum", type=float, default=0.5,
                        help="SGD momentum [0.5] .")
    parser.add_argument("--no_cuda", action="store_true", default=False,
                        help="disables CUDA training [False] .")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed [1] .")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="how many batches to wait before logging training status [10] .")
    parser.add_argument("--log_dir", type=str, default="./log_vtuber",
                        help="log directory for saving model parameters [./log] .")
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if use_cuda else "cpu")

    kwargs = {"num_workers": 2, "pin_memory": True} if use_cuda else {}

    # # for case 1
    # train_loader = DataLoader(
    #                     myDataset(csv_file_path="../data/mnist_train.csv",
    #                               transform=transforms.Compose([
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize((0.1307, ), (0.3081, ))
    #                                 ])),
    #                     batch_size=args.batch_size,
    #                     shuffle=True,
    #                     **kwargs
    #                     )
    #
    # test_loader = DataLoader(
    #                     myDataset(csv_file_path="../data/mnist_test.csv",
    #                               transform=transforms.Compose([
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize((0.1307, ), (0.3081, ))
    #                                 ])),
    #                     batch_size=args.test_batch_size,
    #                     shuffle=False,
    #                     **kwargs
    #                     )

    # for case 2 and case 3
    train_loader = DataLoader(
                        myDataset(path_to_dir="../data/vtuber_train_pt",
                                  transform=transforms.Compose([
                                      # transforms.RandomHorizontalFlip(), # only for image
                                      transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                                    # transforms.Normalize((0.1307, ), (0.3081, ))
                                    ])
                                  ),
                        batch_size=args.batch_size,
                        shuffle=True,
                        **kwargs
                        )

    test_loader = DataLoader(
                        myDataset(path_to_dir="../data/vtuber_test_pt",
                                  transform=transforms.Compose([
                                      transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                                    # transforms.Normalize((0.1307, ), (0.3081, ))
                                    ])
                                  ),
                        batch_size=args.test_batch_size,
                        shuffle=False,
                        **kwargs
                        )

    model = Net().to(device)

    # # load pre-trained model parameters
    # param = torch.load(os.path.join(args.log_dir, "20181117203509_10.pth"))
    # model.load_state_dict(param)

    # optimizer = optim.SGD(model.parameters(),
    #                       lr=args.lr,
    #                       momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(),
                          lr=0.001)

    if save_log: configure(args.log_dir, flush_secs=5)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

        now = datetime.datetime.now()
        torch.save(model.state_dict(),
                   os.path.join(args.log_dir, "{0:%Y%m%d%H%M%S}".format(now) + "_" + str(epoch) + ".pth"))


    draw_graph(np.array(train_loss),
               x="iterations (/100)",
               y="loss value",
               filename=os.path.join(args.log_dir, "train_loss.png"))
    draw_graph(np.array(val_loss),
               x="epochs",
               y="loss value",
               filename=os.path.join(args.log_dir, "val_loss.png"))
    draw_graph(np.array(val_acc),
               x="epochs",
               y="accuracy",
               filename=os.path.join(args.log_dir, "val_acc.png"))


if __name__ == "__main__":
    main()
