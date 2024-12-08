import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from .module import ConvNextBlock, LayerNorm, UpSampleConvnext
from .reverse_function import ReverseFunction

class Fusion(nn.Module):
    def __init__(self, level, channels, first_col):
        super().__init__()
        
        self.level = level
        self.first_col = first_col
        self.down = nn.Sequential(
                nn.Conv2d(channels[level-1], channels[level], kernel_size=2, stride=2),
                LayerNorm(channels[level], eps=1e-6, data_format="channels_first"),
            ) if level in [1, 2, 3] else nn.Identity()
        if not first_col:
            self.up = UpSampleConvnext(1, channels[level+1], channels[level]) if level in [0, 1, 2] else nn.Identity()            

    def forward(self, *args):
        c_down, c_up = args
        if self.first_col:
            x = self.down(c_down)
            return x
        if self.level == 3:
            x = self.down(c_down)
        else:
            x = self.up(c_up) + self.down(c_down)
            
        return x

class Level(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0):
        super().__init__()
        countlayer = sum(layers[:level])
        expansion = 4
        self.fusion = Fusion(level, channels, first_col)
        modules = [ConvNextBlock(channels[level], expansion*channels[level], channels[level], kernel_size = kernel_size,  layer_scale_init_value=1e-6, drop_path=dp_rate[countlayer+i]) for i in range(layers[level])]
        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x = self.fusion(*args)
        x = self.blocks(x)

        return x

class SubNet(nn.Module):
    def __init__(self, channels, layers, kernel_size, first_col, dp_rates):
        super().__init__()
        shortcut_scale_init_value = 0.5
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)), 
                                    requires_grad=True) if shortcut_scale_init_value > 0 else None 
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)), 
                                    requires_grad=True) if shortcut_scale_init_value > 0 else None 
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)), 
                                    requires_grad=True) if shortcut_scale_init_value > 0 else None 
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)), 
                                    requires_grad=True) if shortcut_scale_init_value > 0 else None 
        self.level0 = Level(0, channels, layers, kernel_size, first_col, dp_rates)
        self.level1 = Level(1, channels, layers, kernel_size, first_col, dp_rates)
        self.level2 = Level(2, channels, layers, kernel_size, first_col, dp_rates)
        self.level3 = Level(3, channels, layers, kernel_size, first_col, dp_rates)

    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3= args
        c0 = (self.alpha0)*c0 + self.level0(x, c1)
        c1 = (self.alpha1)*c1 + self.level1(c0, c2)
        c2 = (self.alpha2)*c2 + self.level2(c1, c3)
        c3 = (self.alpha3)*c3 + self.level3(c2, None)

        return c0, c1, c2, c3

    def _forward_reverse(self, *args):
        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunction.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3

    def forward(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)
        
        return self._forward_reverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign=data.sign()
            data.abs_().clamp_(value)
            data*=sign

class DLE(nn.Module):
    def __init__(self, kernel_size = 3, drop_path = 0.1):
        super().__init__()
        self.num_subnet = 4
        self.channels = [64, 128, 256, 512]
        self.layers = [2, 2, 4, 2]
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.channels[0], kernel_size=4, stride=4),
            LayerNorm(self.channels[0], eps=1e-6, data_format="channels_first")
        )
        dp_rate = [x.item() for x in torch.linspace(0, drop_path, sum(self.layers))] 
        for i in range(self.num_subnet):
            first_col = True if i == 0 else False
            self.add_module(f'subnet{str(i)}', SubNet(
                self.channels, self.layers, kernel_size, first_col, dp_rates=dp_rate))

        self.apply(self._init_weights)
        
    def forward(self, x):
        c0, c1, c2, c3 = 0, 0, 0, 0
        x = self.stem(x)        
        for i in range(self.num_subnet):
            c0, c1, c2, c3 = getattr(self, f'subnet{str(i)}')(x, c0, c1, c2, c3)       
            
        return [c0, c1, c2, c3]
            
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=.02)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            nn.init.constant_(module.bias, 0)