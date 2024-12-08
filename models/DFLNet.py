import torch
import re
import torch.nn as nn
import torch.nn.functional as F
from models.msfusion import MSFusion
from models.dle import DLE
from models.sca import SCA

class DFLNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.inplanes = opt.inplanes 
        self.dle = DLE()
        if opt.pretrain_pth != 'None':
            checkpoint = torch.load(opt.pretrain_pth, map_location="cpu")
            self.dle.load_state_dict(checkpoint["model"], strict=False)
        self.sca = SCA(self.inplanes)
        self.msfusion = MSFusion(self.inplanes)

    def forward(self, x):
        _, _, h_input, w_input = x.shape
        f1, f2, f3, f4 = self.dle(x)
        f1, f2, f3, f4 = self.sca(f1, f2, f3, f4)
        ms_feats = f1, f2, f3, f4
        output = self.msfusion(ms_feats)
        output = F.interpolate(output, size=(h_input, w_input), mode='bilinear', align_corners=True)
        
        return output
