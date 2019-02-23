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
parser.add_argument('--weight_path', type=str, default=None)
parser.add_argument('--save_dir', type=str, default=None)
opt = parser.parse_args()

test_set = DatasetFromFolderEval(image_dir='./data/General-100/test', scale_factor=4)
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

model = SRCNN()
criterion = nn.MSELoss()
if opt.cuda:
    model = model.cuda()
    criterion = criterion.cuda()

model.load_state_dict(torch.load(opt.weight_path, map_location='cuda' if opt.cuda else 'cpu'))

model.eval()
total_loss, total_psnr = 0, 0
total_loss_b, total_psnr_b = 0, 0
with torch.no_grad():
    for batch in test_loader:
        inputs, targets = batch[0], batch[1]
        if opt.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
            
        prediction = model(inputs)
        loss = criterion(prediction, targets)
        total_loss += loss.data
        total_psnr += 10 * log10(1 / loss.data)

        loss = criterion(inputs, targets)
        total_loss_b += loss.data
        total_psnr_b += 10 * log10(1 / loss.data)

        save_image(prediction, Path(opt.save_dir) / '{}_sr.png'.format(batch[2][0]), nrow=1)
        save_image(inputs, Path(opt.save_dir) / '{}_lr.png'.format(batch[2][0]), nrow=1)
        save_image(targets, Path(opt.save_dir) / '{}_hr.png'.format(batch[2][0]), nrow=1)

print("===> [Bicubic] Avg. Loss: {:.4f}, PSNR: {:.4f} dB".format(total_loss_b / len(test_loader), total_psnr_b / len(test_loader)))
print("===> [SRCNN] Avg. Loss: {:.4f}, PSNR: {:.4f} dB".format(total_loss / len(test_loader), total_psnr / len(test_loader)))