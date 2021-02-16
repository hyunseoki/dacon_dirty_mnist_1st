from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn


## https://medium.com/analytics-vidhya/how-to-add-additional-layers-in-a-pre-trained-model-using-pytorch-5627002c75a5
## https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py

class CustomModel(nn.Module):
    def __init__(self, ver='efficientnet-b8', in_channel=1, dropout=0.3, advprop=True, resume=None):
        super(CustomModel, self).__init__()
        if resume:
            self.model = EfficientNet.from_name(ver, in_channels=in_channel, num_classes=26, dropout_rate=dropout)
            self.model.load_state_dict(torch.load(resume))
            print('[info msg] pre-trained weight is loaded !!\n')        
            print(resume)
            print('=' * 50)
        else:
            # self.modelmodel = EfficientNet.from_pretrained(args.model, in_channels=in_channel, num_classes=26, dropout_rate=dropout, advprop=advprop)
            self.model = EfficientNet.from_name(ver, in_channels=in_channel, num_classes=26, dropout_rate=dropout)
            print('[info msg] {} model is created\n'.format(ver))
            print('=' * 50)

        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = self.activation(x)

        return x


class MultiLabelEfficientNet(nn.Module):
    def __init__(self, ver='efficientnet-b0', in_channel=3, dropout=0.3):
        super(MultiLabelEfficientNet, self).__init__()

        VALID_MODELS =(
            'efficientnet-b0', 
            'efficientnet-b1', 
            'efficientnet-b2', 
            'efficientnet-b3',
            'efficientnet-b4', 
            'efficientnet-b5', 
            'efficientnet-b6', 
            'efficientnet-b7',            
        )

        VALID_CHANNEL = (1, 3)

        if ver not in VALID_MODELS:
            raise ValueError('model_name should be one of: ' + ', '.join(VALID_MODELS))

        if in_channel not in VALID_CHANNEL:
            raise ValueError('model_name should be one of: ' + ', '.join(VALID_CHANNEL))
        
        last_n = None
        
        if ver == 'efficientnet-b0':
            last_n = 1280

        elif ver == 'efficientnet-b7':
            last_n = 2560

        self.model = EfficientNet.from_pretrained(ver, in_channels=in_channel, dropout=dropout)
        # self.model = EfficientNet.from_name('efficientnet-b0')

        self.fc = nn.Sequential(
            nn.Linear(last_n, 512),
            nn.BatchNorm1d(512),

            nn.Linear(512, 26),
        )

        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        x = self.model.extract_features(x)
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.model._dropout(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = CustomModel()
    input = torch.rand(5, 1, 256, 256)
    out = model(input)
    print(out.shape)
