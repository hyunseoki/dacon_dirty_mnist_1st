from torchinfo import summary
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

## https://medium.com/analytics-vidhya/how-to-add-additional-layers-in-a-pre-trained-model-using-pytorch-5627002c75a5
## https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py


class MultiLabelEfficientNet(nn.Module):
    def __init__(self, ver='efficientnet-b0'):
        super(MultiLabelEfficientNet, self).__init__()
        self.model = EfficientNet.from_pretrained(ver)
        # self.model = EfficientNet.from_name('efficientnet-b0')
        self.fc = nn.Sequential(
            nn.Linear(2304, 512),
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

    def available_model(self):
        print('efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8',)


if __name__ == '__main__':
    model = MultiLabelEfficientNet()
    print(summary(model, input_size=(1, 3, 256, 256), verbose=0))
    # model.available_model()

