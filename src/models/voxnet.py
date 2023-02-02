import torch.nn as nn
import torch.nn.functional as F
import torch

class VoxNet(nn.Module):

    def __init__(self):

        super().__init__()

        # original paper model also has a scaling layer at each group

        # first layer group

        self.conv1a = nn.Conv3d(
            in_channels=1,
            out_channels=32,
            kernel_size = 3,
            stride=2,
            padding='valid'
        )

        self.BN1a = nn.BatchNorm3d(
            num_features=32
        )
        self.relu1a = nn.ReLU()
        self.dropout1a = nn.Dropout3d(
            p = 0.2
        )

        # second layer group

        self.conv2a = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding='valid'
        )
        self.BN2a = nn.BatchNorm3d(
            num_features=64
        )
        self.relu2a = nn.ReLU()
        self.dropout2a = nn.Dropout3d(
            p=0.3
        )

        # third layer group

        self.conv3a = nn.Conv3d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding='valid'
        )
        self.BN3a = nn.BatchNorm3d(num_features=128)
        self.relu3a = nn.ReLU()
        self.dropout3a = nn.Dropout3d(p=0.4)

        # fourth layers group

        self.conv4a = nn.Conv3d(
            in_channels=128,
            out_channels=256,
            kernel_size=3
        )
        self.BN4a = nn.BatchNorm3d(num_features=256)
        self.relu4a = nn.ReLU()
        self.pooling4a = nn.MaxPool3d(
            kernel_size=2,
            stride=2,
            padding=0
        )
        self.dropout4a = nn.Dropout3d(p=0.6)

        # FC layers
        # right now, the out is (batch_size,256,4,4,4) for an input of (batch_size,1,30,30,30)

        self.fc1b = nn.Linear(
            in_features=256*4*4*4,
            out_features=128
        )
        self.relu1b = nn.ReLU()
        self.dropout1b = nn.Dropout1d(p=0.4)

        # output layers

        self.out_class = nn.Linear(in_features=128,out_features=10) # number of classes
        self.out_pose = nn.Linear(in_features=128,out_features=40) # 4 * 10, just to try

    
    def forward(self,x):
        
        x = self.dropout1a(self.relu1a(self.BN1a(self.conv1a(x))))
        x = self.dropout2a(self.relu2a(self.BN2a(self.conv2a(x))))
        x = self.dropout3a(self.relu3a(self.BN3a(self.conv3a(x))))
        x = self.dropout4a(self.pooling4a(self.relu4a(self.BN4a(self.conv4a(x)))))
        x = self.dropout1b(self.relu1b(self.fc1b(torch.flatten(x,1)))) # flatten except batch dimension

        out_class = self.out_class(x)
        out_pose = self.out_pose(x)

        return out_pose,out_class