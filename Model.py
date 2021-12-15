from torch import nn

# net
class ERNet(nn.Module):
    def __init__(self):
        super(ERNet, self).__init__()
        # 卷积层
        self.cnn = nn.Sequential(
            # 1 x 48 x 48
            nn.Conv2d(1, 64, 3, 1, 1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Dropout(p=0.3),  
            # 64 x 24 x 24
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Dropout(p=0.3),  
            # 128 x 12 x 12
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Dropout(p=0.3),  
            # 256 x 6 x 6
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Dropout(p=0.3),  
            # 512 x 3 x 3
        )
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(512 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 7),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out