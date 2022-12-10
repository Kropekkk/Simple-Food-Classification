import torch
from torch import nn

class ModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        #self.conv_block_2 = nn.Sequential(
        #    nn.Conv2d(64, 64, 3),
        #    nn.ReLU(),
        #    nn.Conv2d(64, 64, 3),
        #    nn.ReLU(),
        #    nn.MaxPool2d(2)
        #)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(30*30*32, 3)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        #x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

model = ModelV1()