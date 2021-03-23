from efficientnet_pytorch import EfficientNet
import albumentations.pytorch
import torch
import os
import cv2
from tqdm import tqdm
import pandas as pd
import argparse
import ttach as tta
from glob import glob
from datetime import datetime
import copy


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
                                              
        image = cv2.imread(image_fn, cv2.IMREAD_GRAYSCALE)        
        image = image.reshape([256, 256, 1])

        label = self.label_df.iloc[index,1:].values.astype('float')

        if self.transforms:            
            image = self.transforms(image=image)['image'] / 255.0

        return image, label

base_transforms = {
    'test' : albumentations.Compose([        
        albumentations.pytorch.ToTensorV2(),
        ]),
}

tta_transforms = tta.Compose(
    [
        tta.Rotate90(angles=[0, 90, 180, 270]),
        # tta.Scale(scales=[0.9,1.0,1.1]), ## batch1로 했었음
    ]
)


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="./data/test_dirty_mnist_2nd/")
    parser.add_argument('--label_path', type=str, default="./data/sample_submission.csv")
    parser.add_argument('--weight_path', type=str, default='./save/kfold_202119/')
    parser.add_argument('--out_path', type=str, default='./save/kfold_202119/')

    parser.add_argument('--model', type=str, default='efficientnet-b8')    
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--device', type=str, default=device)

    args = parser.parse_args()

    assert os.path.isdir(args.image_path), 'wrong path'
    assert os.path.isfile(args.label_path), 'wrong path'
    assert os.path.isdir(args.weight_path), 'wrong path' 
    assert os.path.isdir(args.out_path), 'wrong path'

    print('=' * 50)
    print('[info msg] arguments')
    for key, value in vars(args).items():
        print(key, ":", value)
    
    weights = glob(os.path.join(args.weight_path, '*.tar'))

    test_df = pd.read_csv(args.label_path)

    test_set = DatasetMNIST(
        image_folder=args.image_path,
        label_df=test_df,
        transforms=base_transforms['test']
    )

    submission_df = copy.copy(test_df)

    for weight in weights:   
        model = EfficientNet.from_name(args.model, in_channels=1, num_classes=26)
        model.load_state_dict(torch.load(weight, map_location=args.device))
        print('=' * 50)
        print('[info msg] weight {} is loaded'.format(weight))

        test_data_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size = args.batch_size,
                shuffle = False,
            )

        model.to(args.device)
        model.eval()
        tta_model = tta.ClassificationTTAWrapper(model, tta_transforms)

        batch_size = args.batch_size
        batch_index = 0

        print('=' * 50)
        print('[info msg] inference start')

        for i, (images, _) in enumerate(tqdm(test_data_loader)):
            images = images.to(args.device)            
            outputs = tta_model(images).detach().cpu().numpy().squeeze() # soft vote
            # outputs = (outputs > 0.5).astype(int) # hard vote
            batch_index = i * batch_size
            submission_df.iloc[batch_index:batch_index+batch_size, 1:] += outputs

    
    submission_df.iloc[:,1:] = (submission_df.iloc[:,1:] / len(weights) >= 0.35).astype(int)
    
    SAVE_FN = os.path.join(args.out_path, datetime.now().strftime("%m%d%H%M") + '_ensemble_submission.csv')

    submission_df.to_csv(
        SAVE_FN,
        index=False
        )

    print('=' * 50)
    print('[info msg] submission fils is saved to {}'.format(SAVE_FN))


if __name__ == '__main__':
    main()