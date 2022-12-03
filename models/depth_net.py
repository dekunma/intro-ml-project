import torch.nn as nn

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x)
        return x

class DepthNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv_layers = nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
            pool1,
            conv3,
            nn.ReLU(),
            conv4,
            nn.ReLU(),
            pool2,
            conv5,
            nn.ReLU(),
            conv6,
            nn.ReLU(),
            conv7,
            nn.ReLU(),
            pool3,
            conv8,
            nn.ReLU(),
            conv9,
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.1),
            nn.Linear(100352, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(4096, 12),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)        
        return x