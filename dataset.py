import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms import ToTensor, RandomCrop

from PIL import Image, ImageOps
from pathlib import Path
import random

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, patch_size, scale_factor, data_augmentation=True):
        super(DatasetFromFolder, self).__init__()
        self.filenames = [str(filename) for filename in Path(image_dir).glob('*') if filename.suffix in ['.bmp', '.jpg', '.png']]
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.data_augmentation = data_augmentation
        self.crop = RandomCrop(self.patch_size)

    def __getitem__(self, index):
        target_img = Image.open(self.filenames[index]).convert('RGB')
        target_img = self.crop(target_img)
        
        if self.data_augmentation:
            if random.random() < 0.5:
                target_img = ImageOps.flip(target_img)
            if random.random() < 0.5:
                target_img = ImageOps.mirror(target_img)
            if random.random() < 0.5:
                target_img = target_img.rotate(180)

        input_img = target_img.resize((self.patch_size // self.scale_factor,) * 2, Image.BICUBIC)
        input_img = input_img.resize((self.patch_size,) * 2, Image.BICUBIC)

        return ToTensor()(input_img), ToTensor()(target_img)

    def __len__(self):
        return len(self.filenames)

class DatasetFromFolderEval(data.Dataset):
    def __init__(self, image_dir, scale_factor):
        super(DatasetFromFolderEval, self).__init__()
        self.filenames = [str(filename) for filename in Path(image_dir).glob('*') if filename.suffix in ['.bmp', '.jpg', '.png']]
        self.scale_factor = scale_factor

    def __getitem__(self, index):
        target_img = Image.open(self.filenames[index]).convert('RGB')
        
        input_img = target_img.resize((target_img.size[0] // self.scale_factor, target_img.size[1] // self.scale_factor), Image.BICUBIC)
        input_img = input_img.resize(target_img.size, Image.BICUBIC)

        return ToTensor()(input_img), ToTensor()(target_img), Path(self.filenames[index]).stem

    def __len__(self):
        return len(self.filenames)
