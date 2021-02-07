import os
import sys
import random
import numpy as np
import torch
import pandas as pd
from PIL import Image
import tqdm
from torchvision import transforms
from skimage import restoration
import cv2


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DatasetMNIST(torch.utils.data.Dataset):
    def __init__(self, image_folder, label, transforms):        
        self.image_folder = image_folder   
        self.label = pd.read_csv(label)
        self.transforms = transforms

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):        
        image_fn = self.image_folder +\
            str(self.label.iloc[index,0]).zfill(5) + '.png'
                                              
        image = Image.open(image_fn).convert('RGB')
        
        gray_sample = image.convert('L')
        f_image = restoration.denoise_tv_bregman(gray_sample, 0.6)
        _, bin = cv2.threshold((f_image*255).astype('uint8'), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        image = cv2.bitwise_and(np.array(image), np.array(image), mask=bin)
        
        label = self.label.iloc[index,1:].values.astype('float')

        if self.transforms:            
            image = self.transforms(Image.fromarray(image)) /255.0

        return image, label

# http://incredible.ai/pytorch/2020/04/25/Pytorch-Image-Augmentation/#random-affine
mnist_transforms_value ={
    'mean' : [0.485, 0.456, 0.406],
    'std' : [0.229, 0.224, 0.225],
}

mnist_transforms = {
    'train' : transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=(0.2, 3), contrast=(0.2, 3), saturation=(0.2, 3), hue=(-0.5, 0.5)),
        transforms.RandomPerspective(),   
        transforms.ToTensor(),
        transforms.Normalize(mnist_transforms_value['mean'],
                             mnist_transforms_value['std']),
        ]),
    'valid' : transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mnist_transforms_value['mean'],
                             mnist_transforms_value['std']),
        ]),
    'test' : transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mnist_transforms_value['mean'],
                             mnist_transforms_value['std']),
        ]),
}


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def train(train_loader, model, loss_func, device, optimizer, scheduler=None):
    n = 0
    running_loss = 0.0
    epoch_loss = 0.0

    # model.to(device)
    model.train()

    with tqdm.tqdm(train_loader, total=len(train_loader), desc="Train", file=sys.stdout) as iterator:
        for train_x, train_y in iterator:
            train_x = train_x.float().to(device)
            train_y = train_y.float().to(device)
            output = model(train_x)            
            loss = loss_func(output, train_y)
            
            n += train_x.size(0)
            running_loss += loss.item() * train_x.size(0)

            epoch_loss = running_loss / float(n)

            log = 'loss - {:.6f}'.format(epoch_loss)
            iterator.set_postfix_str(log)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

    if scheduler:
        scheduler.step(epoch_loss)

    return epoch_loss


def validate(valid_loader, model, loss_func, device, scheduler=None):
    n = 0
    running_loss = 0.0   
    epoch_loss = 0.0 

    # model.to(device)
    model.eval()

    with tqdm.tqdm(valid_loader, total=len(valid_loader), desc="Valid", file=sys.stdout) as iterator:
        for train_x, train_y in iterator:
            train_x = train_x.float().to(device)
            train_y = train_y.float().to(device)

            with torch.no_grad():
                output = model(train_x)

            loss = loss_func(output, train_y)

            n += train_x.size(0)
            running_loss += loss.item() * train_x.size(0)

            epoch_loss = running_loss / float(n)

            log = 'loss - {:.6f}'.format(epoch_loss)
            iterator.set_postfix_str(log)

    if(scheduler):
        scheduler.step(epoch_loss)

    return epoch_loss