from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn


## https://medium.com/analytics-vidhya/how-to-add-additional-layers-in-a-pre-trained-model-using-pytorch-5627002c75a5
## https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
class MultiLabelEfficientNet(nn.Module):
    def __init__(self, ver='efficientnet-b0'):
        super(MultiLabelEfficientNet, self).__init__()

        self.available_model =('efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8',)

        if ver not in self.available_model:
            self.__available_model__()
            raise ValueError('invalid model name')

        self.model = EfficientNet.from_pretrained(ver)
        # self.model = EfficientNet.from_name('efficientnet-b0')
        # self.conv2d = nn.Sequential(
        #     nn.Conv2d(1, 3, 3, stride=1),
        # ) 
        
        last_n = None
        
        if ver == 'efficientnet-b0':
            last_n = 1280

        elif ver == 'efficientnet-b7':
            last_n = 2560

        self.fc = nn.Sequential(
            nn.Linear(last_n, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.Linear(256, 26),
        )

    def forward(self, x):
        # x = self.conv2d(x)
        x = self.model.extract_features(x)
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.model._dropout(x)
        x = self.fc(x)
        return x

    def __available_model__(self):
        print(self.available_model)


if __name__ == '__main__':
    model = MultiLabelEfficientNet('efficientn')
    input = torch.rand(5, 3, 256, 256)
    print(model(input).shape)

    # model.available_model()

