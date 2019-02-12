import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import Compose, ToTensor

from tensorboardX import SummaryWriter

import cv2
from glob import glob
from math import log10
import argparse

from model import SRCNN
from dataset import DatasetFromFolder, DatasetFromFolderEval

parser = argparse.ArgumentParser(description='SRCNN Example')
parser.add_argument('--cuda', action='store_true', default=False)
opt = parser.parse_args()

train_set = DatasetFromFolder(image_dir='./data/General-100/train', patch_size=96, scale_factor=4, data_augmentation=True, transform=Compose([ToTensor()]))
train_loader = DataLoader(dataset=train_set, batch_size=10, shuffle=True)

val_set = DatasetFromFolderEval(image_dir='./data/General-100/val', scale_factor=4, transform=Compose([ToTensor()]))
val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False)

net = SRCNN()
criterion = nn.MSELoss()

if opt.cuda:
    net = net.cuda()
    criterion = criterion.cuda()

optimizer = optim.Adam(net.parameters(), lr=1e-4)
writer = SummaryWriter('logs/')

for epoch in range(50000):
    epoch_loss = 0
    net.train()
    for batch in train_loader:
        lr, hr = Variable(batch[0]), Variable(batch[1])
        
        if opt.cuda:
            lr = lr.cuda()
            hr = hr.cuda()

        optimizer.zero_grad()        
        sr = net(lr)
        loss = criterion(sr, hr)
        epoch_loss += loss.data[0]
        
        loss.backward()
        optimizer.step()

    writer.add_scalar('train/loss', epoch_loss / len(train_loader), global_step=epoch)
    writer.add_scalar('train/psnr', 10 * log10(1 / (epoch_loss / len(train_loader))), global_step=epoch)
    print('[Epoch {}] Loss: {}'.format(epoch + 1, epoch_loss / len(train_loader)))

    net.eval()
    avg_psnr = 0
    for batch in val_loader:
        lr, hr = Variable(batch[0]), Variable(batch[1])
        
        if opt.cuda:
            lr = lr.cuda()
            hr = hr.cuda()
            
        sr = net(lr)
        mse = criterion(sr, hr)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    writer.add_scalar('val/psnr', avg_psnr / len(val_loader), global_step=epoch)
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(val_loader)))

    img_in = batch[0][:1].permute(0, 2, 3, 1).squeeze(0).numpy() * 255
    img_tar = batch[1][:1].permute(0, 2, 3, 1).squeeze(0).numpy() * 255
    img_sr = sr[:1].data.permute(0, 2, 3, 1).squeeze(0).cpu().numpy() * 255

    cv2.imwrite('input.bmp', img_in[:, :, ::-1])
    cv2.imwrite('target.bmp', img_tar[:, :, ::-1])
    cv2.imwrite('sr.bmp', img_sr[:, :, ::-1])

