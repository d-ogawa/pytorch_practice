import pandas as pd
import glob
import os
import numpy as np
from skimage import io
from torchvision import transforms
import torch

# # case 1
# # mnist_train
# files = glob.glob(os.path.join("../data/mnist_train", "*.jpg"))
# files = sorted(files)
# print(len(files))
#
# train_df = pd.DataFrame(data=None, index=None, columns=["label", "path_to_image"])
#
# for (i, path) in enumerate(files):
#     print(path)
#     label = os.path.splitext(path)[0][-1]   # [../data/mnist_train/0-7, .jpg]
#     train_df.at[i, "label"] = label
#     train_df.at[i, "path_to_image"] = path
#
# train_df.to_csv("../data/mnist_train.csv")
#
#
# # mnist_test
# files = glob.glob(os.path.join("../data/mnist_test", "*.jpg"))
# files = sorted(files)
# print(len(files))
#
# test_df = pd.DataFrame(data=None, index=None, columns=["label", "path_to_image"])
#
# for (i, path) in enumerate(files):
#     print(path)
#     label = os.path.splitext(path)[0][-1]   # [../data/mnist_train/0-7, .jpg]
#     test_df.at[i, "label"] = label
#     test_df.at[i, "path_to_image"] = path
#
# test_df.to_csv("../data/mnist_test.csv")






# case 2
# mnist_train
files = glob.glob(os.path.join("../data/mnist_train", "*.jpg"))
files = sorted(files)
print(len(files))

for (i, path) in enumerate(files):
    print(path)
    img = io.imread(path)
    label = int(os.path.splitext(path)[0][-1])   # [../data/mnist_train/0-7, .jpg]

    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    img_tensor = transforms.ToTensor()(img)
    label_tensor = torch.tensor(label)

    save_path = "../data/mnist_train_pt/" + "{0:05d}".format(i) + ".pt"
    torch.save((img_tensor, label_tensor), save_path)


# mnist_test
files = glob.glob(os.path.join("../data/mnist_test", "*.jpg"))
files = sorted(files)
print(len(files))

for (i, path) in enumerate(files):
    print(path)
    img = io.imread(path)
    label = int(os.path.splitext(path)[0][-1])   # [../data/mnist_train/0-7, .jpg]

    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    img_tensor = transforms.ToTensor()(img)
    label_tensor = torch.tensor(label)

    save_path = "../data/mnist_test_pt/" + "{0:05d}".format(i) + ".pt"
    torch.save((img_tensor, label_tensor), save_path)
