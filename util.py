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
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 


class DatasetMNIST(torch.utils.data.Dataset):
    def __init__(self, image_folder, label_df, transforms):        
        self.image_folder = image_folder   
        self.label_df = label_df
        self.transforms = transforms

    def distance(self, point1,point2):
        return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

    def gaussianLP(self, D0, imgShape):
        base = np.zeros(imgShape[:2])
        rows, cols = imgShape[:2]
        center = (rows/2,cols/2)
        for x in range(cols):
            for y in range(rows):
                base[y,x] = np.exp(((-self.distance((y,x),center)**2)/(2*(D0**2))))
        return base

    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self, index):        
        image_fn = self.image_folder +\
            str(self.label_df.iloc[index,0]).zfill(5) + '.png'
                                              
        image = cv2.imread(image_fn, cv2.IMREAD_GRAYSCALE)        
        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # image = cv2.bilateralFilter(image, 9, 150, 150)
        image = image.reshape([256, 256, 1])

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
            albumentations.OneOf([
                albumentations.GridDistortion(distort_limit=(-0.3, 0.3), border_mode=cv2.BORDER_CONSTANT, p=1),
                albumentations.ShiftScaleRotate(rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=1),        
                albumentations.ElasticTransform(alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, p=1),
            ], p=1),    
            albumentations.Cutout(num_holes=16, max_h_size=15, max_w_size=15, fill_value=0),
            albumentations.pytorch.ToTensorV2(),
        ]),
    'valid' : albumentations.Compose([        
        albumentations.pytorch.ToTensorV2(),
        ]),
    'test' : albumentations.Compose([        
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
    running_corrects = 0

    epoch_loss = 0.0
    epoch_acc = 0.0

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

            output = output > 0.5
            running_corrects += (output == train_y).sum()
            epoch_acc = running_corrects / train_y.size(1) / n

            log = 'loss - {:.5f}, acc - {:.5f}'.format(epoch_loss, epoch_acc)
            
            iterator.set_postfix_str(log)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

    if scheduler:
        scheduler.step(epoch_loss)

    return epoch_loss, epoch_acc


def validate(valid_loader, model, loss_func, device, scheduler=None):
    n = 0
    running_loss = 0.0
    running_corrects = 0

    epoch_loss = 0.0
    epoch_acc = 0.0

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

            output = output > 0.5
            running_corrects += (output == train_y).sum()
            epoch_acc = running_corrects / train_y.size(1) / n

            log = 'loss - {:.5f}, acc - {:.5f}'.format(epoch_loss, epoch_acc)

            iterator.set_postfix_str(log)

    if(scheduler):
        scheduler.step(epoch_loss)

    return epoch_loss, epoch_acc