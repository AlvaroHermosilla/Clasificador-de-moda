import torch
import torch.nn as nn

class CNN_bn(torch.nn.Module):
    def __init__(self,number_classes):
        super(CNN_bn,self).__init__()
        channels = [3, 32, 64, 128, 256, 512, 1024]  # in/out channels
        self.blocks = nn.ModuleList()

        for i in range(len(channels) - 1):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i+1], kernel_size=5, padding=2),
                    nn.BatchNorm2d(channels[i+1]),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2)
                )
            )
        # Aplanamiento
        self.avpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024,2048)
        self.relu_fc = nn.ReLU()
        self.bnfc = nn.BatchNorm1d(2048)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(2048,number_classes)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        x = self.avpool(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bnfc(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x