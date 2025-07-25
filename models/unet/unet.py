import os
import sys
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from modules import *
class SegModel(nn.Module):
    def __init__(self, n_channel, n_class):
        super().__init__()
        self.n_channel = n_channel
        self.n_class = n_class

        self.first_conv = DoubleConv(n_channel, 64)
        
        self.down1 = DownScaling(64, 128)
        self.down2 = DownScaling(128, 256)
        self.up3 = UpScaling(256, 128)
        self.up4 = UpScaling(128, 64)

        self.final_conv = OutConv(64, n_class)

    def forward(self, x):
        x1 = self.first_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)

        return F.sigmoid(self.final_conv(x))