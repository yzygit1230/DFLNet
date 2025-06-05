import torch
import torch.nn as nn

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1) 
        key = self.key_conv(x).view(batch_size, -1, width * height)  
        value = self.value_conv(x).view(batch_size, -1, width * height)  
        attention = torch.bmm(query, key)
        attention = self.softmax(attention)  
        out = torch.bmm(value, attention.permute(0, 2, 1)) 
        out = out.view(batch_size, C, width, height) 

        return out

class SCA(nn.Module):
    def __init__(self, inplanes):
        super(SCA, self).__init__()
        self.sa_1 = SpatialAttention()
        self.sa_2 = SpatialAttention()
        self.sa_3 = SpatialAttention()
        self.sa_4 = SpatialAttention()
        
        self.cross_1 = CrossAttention(inplanes)
        self.cross_2 = CrossAttention(inplanes*2)
        self.cross_3 = CrossAttention(inplanes*4)
        self.cross_4 = CrossAttention(inplanes*8)
        
        self.Translayer_1 = BasicConv2d(inplanes,  inplanes, 1)
        self.Translayer_2 = BasicConv2d(inplanes*2, inplanes*2, 1)
        self.Translayer_3 = BasicConv2d(inplanes*4, inplanes*4, 1)
        self.Translayer_4 = BasicConv2d(inplanes*8, inplanes*8, 1)

    def forward(self, fa1, fa2, fa3, fa4):
        fa1 = self.sa_1(fa1) * fa1
        fa1 = self.cross_1(fa1) * fa1
        fa1 = self.Translayer_1(fa1)

        fa2 = self.sa_2(fa2) * fa2
        fa2 = self.cross_2(fa2) * fa2
        fa2 = self.Translayer_2(fa2)

        fa3 = self.sa_3(fa3) * fa3
        fa3 = self.cross_3(fa3) * fa3
        fa3 = self.Translayer_3(fa3)

        fa4 = self.sa_4(fa4) * fa4
        fa4 = self.cross_4(fa4) * fa4
        fa4 = self.Translayer_4(fa4)

        return fa1, fa2, fa3, fa4
