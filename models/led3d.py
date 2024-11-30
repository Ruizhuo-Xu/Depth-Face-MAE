import torch
from torch import nn
import torch.nn.functional as F
from .models import register

def init_weight(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=2e-5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
@register('led3d')
class Led3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=509, drop_p=0.9):
        super().__init__()
        self.num_classes = num_classes
        self.drop_p = drop_p
        self.block1 = Conv3x3(in_channels, 32)
        self.block2 = Conv3x3(32, 64)
        self.block3 = Conv3x3(64, 128)
        self.block4 = Conv3x3(128, 256)
        self.block5 = Conv3x3(480, 960)
        self.maxpool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=33, stride=16, padding=16)
        self.maxpool2 = nn.MaxPool2d(kernel_size=17, stride=8, padding=8)
        self.maxpool3 = nn.MaxPool2d(kernel_size=9, stride=4, padding=4)
        self.global_conv = nn.Conv2d(960, 960, kernel_size=8, stride=8, groups=960)
        self.flatten = nn.Flatten()
        self.dorpout = nn.Dropout(p=drop_p)
        self.fc = nn.Linear(960, self.num_classes)

    def forward(self, x, label=None, only_return_feats=False):
        # stage 1 128*128
        s1_relu = self.block1(x)
        s1_pool = self.maxpool0(s1_relu)
        # stage 2 64*64
        s2_relu = self.block2(s1_pool)
        s2_pool = self.maxpool0(s2_relu)
        # stage 3 32*32
        s3_relu = self.block3(s2_pool)
        s3_pool = self.maxpool0(s3_relu)
        # stage 4 16*16
        s4_relu = self.block4(s3_pool)
        s4_pool = self.maxpool0(s4_relu)
        # stage 5 8*8
        s1_global_pool = self.maxpool1(s1_relu)
        s2_global_pool = self.maxpool2(s2_relu)
        s3_global_pool = self.maxpool3(s3_relu)
        s4_global_pool = s4_pool
        feature = torch.cat([s1_global_pool, s2_global_pool,
                            s3_global_pool, s4_global_pool], dim=1)
        # stage 5 8*8
        s5_relu = self.block5(feature)
        s5_global_feature = self.global_conv(s5_relu)
        # s5_global_feature = self.gap(s5_relu)  # test
        global_feature = self.flatten(s5_global_feature)
        out = self.dorpout(global_feature)
        if not only_return_feats:
            output = self.fc(out)
            loss = F.cross_entropy(output, label)
            return loss, output
        else:
            return out
        
if __name__ == '__main__':
    model = Led3D()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    from torchinfo import summary
    summary(model, (2, 1, 128, 128), device=device)