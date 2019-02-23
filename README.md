# SRCNN
The PyTorch implementation of SRCNN for single image super resolution.

The following entry was written based on this repo.  
https://buildersbox.corp-sansan.com/entry/2019/02/21/110000

## Requirements
- torch
- torchvision
- pillow
- tensorboardx
- googledrivedownloader


## Data preparation
You can use General-100 dataset through a script. Change your directory to `./data` and run `general100.py`. It will download dataset from Google Drive and split the dataset into train/val/test randomly. The split ratio is `(train, val, test) = (8, 1, 1)`.
```
$ cd ./data
$ python genaral100.py
```

## Training
You can train the model through `train.py`.
```
$ python train.py
```
If you have a GPU, you should set the `--cuda` argument. It does not support multiple GPUs and data parallelization.
```
$ python train.py --cuda
```

## Testing
Run `test.py`. You must specify the location of `.pth` file and a directory to save the results.
```
$ python test.py --weight_path ./runs/your/model/weight.pth --save_dir ./want/to/save/ --cuda
```
