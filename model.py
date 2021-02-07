from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn


## https://medium.com/analytics-vidhya/how-to-add-additional-layers-in-a-pre-trained-model-using-pytorch-5627002c75a5
## https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
class MultiLabelEfficientNet(nn.Module):
    def __init__(self, ver='efficientnet-b0', in_channel=3):
        super(MultiLabelEfficientNet, self).__init__()

        available_model =(
            'efficientnet-b0', 
            'efficientnet-b1', 
            'efficientnet-b2', 
            'efficientnet-b3',
            'efficientnet-b4', 
            'efficientnet-b5', 
            'efficientnet-b6', 
            'efficientnet-b7',
            'efficientnet-b8',
        )

        available_channel = (1, 3)

        if ver not in available_model:
            print(available_model)
            raise ValueError('invalid model name')

        if in_channel not in available_channel:
            print(available_channel)
            raise ValueError('invalid image channel')     
        
        last_n = None
        
        if ver == 'efficientnet-b0':
            last_n = 1280

        elif ver == 'efficientnet-b7':
            last_n = 2560

        self.in_channel = in_channel

        self.model = EfficientNet.from_pretrained(ver)
        # self.model = EfficientNet.from_name('efficientnet-b0')

        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 3, 3, stride=1),
        ) 

        self.fc = nn.Sequential(
            nn.Linear(last_n, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.Linear(256, 26),
        )

    def forward(self, x):
        if(self.in_channel == 1):
            x = self.conv2d(x)

        x = self.model.extract_features(x)
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.model._dropout(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = EfficientNet.from_pretrained('efficientnet-b0', in_channels=3, num_classes=26, dropout_rate=0.3)

    # model = MultiLabelEfficientNet('efficientnet-b0', in_channel=1)
    input = torch.rand(5, 3, 256, 256)
    print(model(input).shape)

    # model.available_model()

