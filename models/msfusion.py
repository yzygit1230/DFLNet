import torch
import torch.nn as nn
from models.Base import Conv3Relu, DropBlock
from models.Base import DropBlock

class MSFusion(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        inter_channels = inplanes // 4
        self.stage1_Conv1 = Conv3Relu(inplanes * 1, inplanes)  
        self.stage2_Conv1 = Conv3Relu(inplanes * 2, inplanes * 2)  
        self.stage3_Conv1 = Conv3Relu(inplanes * 4, inplanes * 4)  
        self.stage4_Conv1 = Conv3Relu(inplanes * 8, inplanes * 8)  
 
        rate, size, step = (0.15, 7, 30)
        self.drop = DropBlock(rate=rate, size=size, step=step)
            
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.concat_Conv = Conv3Relu(inplanes * 7, inplanes)
        self.out_Conv = nn.Sequential(nn.Conv2d(inplanes, inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(inter_channels, momentum=0.0003),
                        nn.ReLU(),
                        nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(inter_channels, momentum=0.0003),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=inter_channels, out_channels= 2, kernel_size=1,
                        stride=1, padding=0, dilation=1, bias=True))
        
    def forward(self, ms_feats):
        fa1, fa2, fa3, fa4 = ms_feats
        [fa1, fa2, fa3, fa4] = self.drop([fa1, fa2, fa3, fa4])  

        feature1 = self.stage1_Conv1(fa1)  
        feature2 = self.stage2_Conv1(fa2)  
        feature3 = self.stage3_Conv1(fa3)  
        feature4 = self.stage4_Conv1(fa4)  
        feature2 = self.up(feature2)
        feature3 = self.up4(feature3)
        feature4 = self.up8(feature4)
        feature = self.concat_Conv(torch.cat([feature1, feature2, feature2, feature2], 1))
        feature = self.out_Conv(feature)

        return feature

