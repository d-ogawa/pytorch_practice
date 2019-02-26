from __future__ import print_function
import matplotlib as mpl
mpl.use("Agg")
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from skimage import io
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
from PIL import Image
import copy
import matplotlib.pylab as plt
import seaborn as sns
sns.set(style="darkgrid")

class myDataset(Dataset):
    """
    Dataset for Handmade dataset
    Use pt data and load every data in advance
    Much faster than case 1 and case 2
    """

    def __init__(self, path_to_dir, transform=None):

        files = sorted(glob.glob(os.path.join(path_to_dir, "*.jpg")))
        data = []
        print(len(files))
        for file in files:
            # img = io.imread(file)
            img = Image.open(file)#.convert("RGB") img = io.imread(file) ?

            # print(len(img.size))
            # if img.ndim == 2:
            # if len(img.size) == 2:
                # img = np.expand_dims(img, axis=-1)          # for 2d image
            # print(img.shape)
            # img = Image.fromarray(img)
            # img = Image.fromarray(np.uint8(img))
            # img = Image.fromarray(img.astype("uint8"))

            data.append(copy.deepcopy(img))
            # data.append(img.load())

            img.close()

        self.data = data
        self.transform = transform

    def __len__(self):
        return (len(self.data))

    def __getitem__(self, idx):

        # img, label = self.data[idx]
        img = self.data[idx]

        if self.transform:
            img = self.transform(img)

        return (img, )


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.ngpu = args.ngpu
        nz = int(args.nz)
        ngf = int(args.ngf)
        ndf = int(args.ndf)
        nc = int(args.nc)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.ngpu = args.ngpu
        nz = int(args.nz)
        ngf = int(args.ngf)
        ndf = int(args.ndf)
        nc = int(args.nc)
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


def draw_graph(data, x, y, filename=None):
    # sns.lineplot(data=data, x="iterations (/100)", y="iterations (/100)")
    ax = sns.lineplot(data=data)
    ax.set(xlabel=x, ylabel=y)
    plt.savefig(filename)
    plt.clf()


def main():
    parser = argparse.ArgumentParser(description="Pytorch DCGAN .")
    parser.add_argument("--workers", type=int, default=2, 
                        help="number of data loading workers [2] .")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="input batch size for training [64] .")
    parser.add_argument("--image_size", type=int, default=64, 
                        help="the height and width of the input image to network [64] .")
    parser.add_argument("--epochs", type=int, default=25,
                        help="number of epochs to train [10] .")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="learning rate [0.0002] .")
    parser.add_argument("--beta1", type=float, default=0.5, 
                        help="beta1 for adam [0.5] .")
    
    parser.add_argument("--nz", type=int, default=100, 
                        help="size of the latent z vector [100] .")
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--nc", type=int, default=3)
    
    parser.add_argument("--no_cuda", action="store_true", default=False,
                        help="disables CUDA training [False] .")
    parser.add_argument("--ngpu", type=int, default=1, 
                        help="number of GPUs to use [1] .")
    parser.add_argument("--netG", default="", 
                        help="path to netG (to continue training) .")
    parser.add_argument("--netD", default="", 
                        help="path to netD (to continue training) .")
    parser.add_argument("--log_dir", type=str, default="./log",
                        help="log directory for saving model parameters [./log] .")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="how many batches to wait before logging training status [10] .")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed [1] .")
    args = parser.parse_args()
    print(args)
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cudnn.benchmark = True
    
    if torch.cuda.is_available() and args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu")
    kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}


    dataloader = DataLoader(
                    myDataset(path_to_dir="../data/vtuber", 
                              transform=transforms.Compose([
                                  transforms.Resize(args.image_size), 
                                  transforms.ToTensor(), 
                                  # transforms.Normalize((0.1307, ), (0.3081, ))
                                  transforms.Normalize((0.5,0.5,0.5), 
                                                       (0.5,0.5,0.5))
                                  ])
                              ),
                        batch_size=args.batch_size,
                        shuffle=True,
                        **kwargs
                        )

    ngpu = int(args.ngpu)
    nz = int(args.nz)
    ngf = int(args.ngf)
    ndf = int(args.ndf)
    nc = 1
    
    # setup generator
    netG = Generator(args).to(device)
    netG.apply(weights_init)
    if args.netG != "":
        netG.load_state_dict(torch.load(args.netG))
    print(netG)
    
    # setup discriminator
    netD = Discriminator(args).to(device)
    netD.apply(weights_init)
    if args.netD != "":
        netD.load_state_dict(torch.load(args.netD))
    print(netD)
    
    criterion = nn.BCELoss()
    
    fixed_noise = torch.randn(args.batch_size, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0
    
    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), 
                            lr=args.lr, 
                            betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), 
                            lr=args.lr, 
                            betas=(args.beta1, 0.999))
    
    loss_D = []
    loss_G = []
    
    for epoch in range(args.epochs):
        for (i, data) in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            bsize = real_cpu.size(0)
            label = torch.full((bsize, ), real_label, device=device)
            
            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            # train with fake
            noise = torch.randn(bsize, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            errD = errD_real + errD_fake
            optimizerD.step()
            
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label) # fake labels are real for generator cost
            
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            print("[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f" 
                  % (epoch, 
                     args.epochs, 
                     i, 
                     len(dataloader), 
                     errD.item(), 
                     errG.item(), 
                     D_x, 
                     D_G_z1, 
                     D_G_z2))
            
            loss_D.append(errD.item())
            loss_G.append(errG.item())
            
            if i % 100 == 0:
                vutils.save_image(real_cpu, 
                                  "%s/real_samples.png" % args.log_dir, 
                                  normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(), 
                                  "%s/fake_samples_epoch_%03d.png" % (args.log_dir, epoch), 
                                  normalize=True)

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.log_dir, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.log_dir, epoch))

    draw_graph(np.array(loss_D),
               x="iterations",
               y="loss value",
               filename=os.path.join(args.log_dir, "loss_D.png"))
    draw_graph(np.array(loss_G),
               x="iterations",
               y="loss value",
               filename=os.path.join(args.log_dir, "loss_G.png"))


if __name__ == "__main__":
    main()