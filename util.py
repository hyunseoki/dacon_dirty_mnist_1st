import os
import sys
import random
import numpy as np
import torch
import tqdm
import albumentations
import albumentations.pytorch
import cv2


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    #cudnn 시드 고정 : true, false 
    #cudnn 시드 랜덤 : false, true
    # https://www.facebook.com/groups/PyTorchKR/permalink/1010080022465012/
    # https://hoya012.github.io/blog/reproducible_pytorch/
    
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True 


class DatasetMNIST(torch.utils.data.Dataset):
    def __init__(self, image_folder, label_df, transforms):        
        self.image_folder = image_folder   
        self.label_df = label_df
        self.transforms = transforms

    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, index):        
        image_fn = self.image_folder +\
            str(self.label_df.iloc[index,0]).zfill(5) + '.png'
                                              
        image = cv2.imread(image_fn)
        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.label_df.iloc[index,1:].values.astype('float')

        if self.transforms:            
            image = self.transforms(image=image)['image'] / 255.0

        return image, label


mnist_transforms_value ={
    'mean' : [0.485, 0.456, 0.406],
    'std' : [0.229, 0.224, 0.225],
}

mnist_transforms = {
    'train' : albumentations.Compose([
        albumentations.RandomRotate90(),
        albumentations.VerticalFlip(),
        albumentations.RandomBrightnessContrast(), 
        albumentations.Normalize(mnist_transforms_value['mean'],
                                mnist_transforms_value['std']), 
        albumentations.pytorch.ToTensorV2(),
        ]),
    'valid' : albumentations.Compose([        
        albumentations.Normalize(mnist_transforms_value['mean'],
                                mnist_transforms_value['std']),
        albumentations.pytorch.ToTensorV2(),
        ]),
    'test' : albumentations.Compose([        
        albumentations.Normalize(mnist_transforms_value['mean'],
                                mnist_transforms_value['std']),
        albumentations.pytorch.ToTensorV2(),
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
            train_y = train_y.long().to(device)
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
            train_y = train_y.long().to(device)

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