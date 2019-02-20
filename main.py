import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from pathlib import Path
from math import log10

from model import SRCNN
from dataset import DatasetFromFolder, DatasetFromFolderEval

import argparse
parser = argparse.ArgumentParser(description='predictionCNN Example')
parser.add_argument('--cuda', action='store_true', default=False)
opt = parser.parse_args()

train_set = DatasetFromFolder(image_dir='./data/General-100/train', patch_size=96, scale_factor=4, data_augmentation=True)
train_loader = DataLoader(dataset=train_set, batch_size=10, shuffle=True)

val_set = DatasetFromFolderEval(image_dir='./data/General-100/val', scale_factor=4)
val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False)

model = SRCNN()
criterion = nn.MSELoss()
if opt.cuda:
    model = model.cuda()
    criterion = criterion.cuda()

optimizer = optim.Adam([{'params': model.conv1.parameters()},
                        {'params': model.conv2.parameters()},
                        {'params': model.conv3.parameters(), 'lr': 1e-5}],
                        lr=1e-4)

writer = SummaryWriter()
log_dir = Path(writer.log_dir)
sample_dir = log_dir / 'sample'
sample_dir.mkdir(exist_ok=True)
weight_dir = log_dir / 'weights'
weight_dir.mkdir(exist_ok=True)

for epoch in range(50000):
    model.train()
    epoch_loss, epoch_psnr = 0, 0
    for batch in train_loader:
        inputs, targets = Variable(batch[0]), Variable(batch[1])
        if opt.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()        
        prediction = model(inputs)
        loss = criterion(prediction, targets)
        epoch_loss += loss.data
        epoch_psnr += 10 * log10(1 / loss.data)
        
        loss.backward()
        optimizer.step()

    writer.add_scalar('train/loss', epoch_loss / len(train_loader), global_step=epoch)
    writer.add_scalar('train/psnr', epoch_psnr / len(train_loader), global_step=epoch)
    print('[Epoch {}] Loss: {:.4f}, PSNR: {:.4f} dB'.format(epoch + 1, epoch_loss / len(train_loader), epoch_psnr / len(train_loader)))

    if (epoch + 1) % 1000 != 0:
        continue

    model.eval()
    val_loss, val_psnr = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch[0], batch[1]
            if opt.cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
                
            prediction = model(inputs)
            loss = criterion(prediction, targets)
            val_loss += loss.data
            val_psnr += 10 * log10(1 / loss.data)

            save_image(prediction, sample_dir / '{}_epoch{:05}.png'.format(batch[2][0], epoch + 1), nrow=1)

    writer.add_scalar('val/loss', val_loss / len(val_loader), global_step=epoch)
    writer.add_scalar('val/psnr', val_psnr / len(val_loader), global_step=epoch)
    print("===> Avg. Loss: {:.4f}, PSNR: {:.4f} dB".format(val_loss / len(val_loader), val_psnr / len(val_loader)))

    torch.save(model.state_dict(), str(weight_dir / 'weight_epoch{:05}.pth'.format(epoch + 1)))

