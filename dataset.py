import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from random import randrange

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

def get_patch(img, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img.size
    if ix == -1:
        ix = random.randrange(0, iw - patch_size + 1)
    if iy == -1:
        iy = random.randrange(0, ih - patch_size + 1)

    img_tar = img.crop((iy, ix, iy + patch_size, ix + patch_size))
    img_in = img_tar.resize((int(img_tar.size[0] / scale),int(img_tar.size[1]/scale)), Image.BICUBIC)
    img_in = img_in.resize((img_in.size[0] * scale, img_in.size[1] * scale), Image.BICUBIC)

    return img_in, img_tar

def augment(img_in, img_tar, flip_h=True, rot=True):
    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        
    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            
    return img_in, img_tar
    
class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, patch_size, scale_factor, data_augmentation, transform=None):
        super(DatasetFromFolder, self).__init__()
        self.filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation

    def __getitem__(self, index):
        target = load_img(self.filenames[index])
        input, target= get_patch(target, self.patch_size, self.scale_factor)
        
        if self.data_augmentation:
            input, target = augment(input, target)
        
        if self.transform:
            input = self.transform(input)
            target = self.transform(target)
                
        return input, target

    def __len__(self):
        return len(self.filenames)

class DatasetFromFolderEval(data.Dataset):
    def __init__(self, image_dir, scale_factor, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        self.filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]
        self.scale_factor = scale_factor
        self.transform = transform

    def __getitem__(self, index):
        target = load_img(self.filenames[index])
        input = target.resize((int(target.size[0] / self.scale_factor),int(target.size[1]/self.scale_factor)), Image.BICUBIC)
        input = input.resize((target.size[0], target.size[1]), Image.BICUBIC)
        
        if self.transform:
            target = self.transform(target)
            input = self.transform(input)
            
        return input, target
      
    def __len__(self):
        return len(self.filenames)