import pandas as pd
import glob
import os
import numpy as np
from skimage import io
from torchvision import transforms
import torch
import random

def divide_train_test(face, train_ratio):
    face_num = len(face)
    divide_idx = int(face_num * train_ratio)

    train, test = face[:divide_idx], face[divide_idx:]

    return train, test


random.seed(1)
OUTPUT_TRAIN_DIR = './vtuber_train_pt'
OUTPUT_TEST_DIR = './vtuber_test_pt'


KizunaAI_face = glob.glob('./face/KizunaAI/*.jpg')
random.shuffle(KizunaAI_face)
print('Num of KizunaAI faces : %d' %(len(KizunaAI_face)))
KizunaAI_train, KizunaAI_test = divide_train_test(KizunaAI_face, train_ratio=0.9)

MiraiAkari_face = glob.glob('./face/MiraiAkari/*.jpg')
random.shuffle(MiraiAkari_face)
print('Num of MiraiAkari faces : %d' %(len(MiraiAkari_face)))
MiraiAkari_train, MiraiAkari_test = divide_train_test(MiraiAkari_face, train_ratio=0.9)

KaguyaLuna_face = glob.glob('./face/KaguyaLuna/*.jpg')
random.shuffle(KaguyaLuna_face)
print('Num of KaguyaLuna faces : %d' %(len(KaguyaLuna_face)))
KaguyaLuna_train, KaguyaLuna_test = divide_train_test(KaguyaLuna_face, train_ratio=0.9)

Siro_face = glob.glob('./face/Siro/*.jpg')
random.shuffle(Siro_face)
print('Num of Siro faces : %d' %(len(Siro_face)))
Siro_train, Siro_test = divide_train_test(Siro_face, train_ratio=0.9)

NekoMas_face = glob.glob('./face/NekoMas/*.jpg')
random.shuffle(NekoMas_face)
print('Num of NekoMas faces : %d' %(len(NekoMas_face)))
NekoMas_train, NekoMas_test = divide_train_test(NekoMas_face, train_ratio=0.9)


# train data
if not os.path.exists(OUTPUT_TRAIN_DIR):
    os.makedirs(OUTPUT_TRAIN_DIR)

num = 0
for (label, files) in enumerate([KizunaAI_train, MiraiAkari_train, KaguyaLuna_train, Siro_train, NekoMas_train]):
    print(label, len(files))
    for file in files:
        base = '{:05}'.format(num)
        img = io.imread(file)
        img_tensor = transforms.ToTensor()(img)
        label_tensor = torch.tensor(label)

        save_path = os.path.join(OUTPUT_TRAIN_DIR, base + ".pt")
        torch.save((img_tensor, label_tensor), save_path)
        num += 1


# test data
if not os.path.exists(OUTPUT_TEST_DIR):
    os.makedirs(OUTPUT_TEST_DIR)

num = 0
for (label, files) in enumerate([KizunaAI_test, MiraiAkari_test, KaguyaLuna_test, Siro_test, NekoMas_test]):
    print(label, len(files))
    for file in files:
        base = '{:05}'.format(num)
        img = io.imread(file)
        img_tensor = transforms.ToTensor()(img)
        label_tensor = torch.tensor(label)

        save_path = os.path.join(OUTPUT_TEST_DIR, base + ".pt")
        torch.save((img_tensor, label_tensor), save_path)
        num += 1











# files = glob.glob(os.path.join("../data/vtuber_train", "*.jpg"))
# files = sorted(files)
# print(len(files))
#
# for (i, path) in enumerate(files):
#     print(path)
#     img = io.imread(path)
#     label = int(os.path.splitext(path)[0][-1])   # [../data/mnist_train/0-7, .jpg]
#
#     if img.ndim == 2:
#         img = np.expand_dims(img, axis=-1)
#     img_tensor = transforms.ToTensor()(img)
#     label_tensor = torch.tensor(label)
#
#     save_path = "../data/vtuber_train_pt/" + "{0:05d}".format(i) + ".pt"
#     torch.save((img_tensor, label_tensor), save_path)
#
#
# # test data
# files = glob.glob(os.path.join("../data/vtuber_test", "*.jpg"))
# files = sorted(files)
# print(len(files))
#
# for (i, path) in enumerate(files):
#     print(path)
#     img = io.imread(path)
#     label = int(os.path.splitext(path)[0][-1])   # [../data/mnist_train/0-7, .jpg]
#
#     if img.ndim == 2:
#         img = np.expand_dims(img, axis=-1)
#     img_tensor = transforms.ToTensor()(img)
#     label_tensor = torch.tensor(label)
#
#     save_path = "../data/vtuber_test_pt/" + "{0:05d}".format(i) + ".pt"
#     torch.save((img_tensor, label_tensor), save_path)
