import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint

def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out

def maxpool(x, dim=-1, keepdim=False):
    out = x.max(dim=dim, keepdim=keepdim).values
    return out

class MultiStagePointNetEncoder(nn.Module):
    def __init__(self, h_dim=128, out_channels=128, num_layers=4, **kwargs):
        super().__init__()

        self.h_dim = h_dim
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.act = nn.LeakyReLU(negative_slope=0.0, inplace=False)

        self.conv_in = nn.Conv1d(3, h_dim, kernel_size=1)
        self.layers, self.global_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(nn.Conv1d(h_dim, h_dim, kernel_size=1))
            self.global_layers.append(nn.Conv1d(h_dim * 2, h_dim, kernel_size=1))
        self.conv_out = nn.Conv1d(h_dim * self.num_layers, out_channels, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2) # [B, N, 3] --> [B, 3, N]
        y = self.act(self.conv_in(x))
        feat_list = []
        for i in range(self.num_layers):
            y = self.act(self.layers[i](y))
            y_global = y.max(-1, keepdim=True).values
            y = torch.cat([y, y_global.expand_as(y)], dim=1)
            y = self.act(self.global_layers[i](y))
            feat_list.append(y)
        x = torch.cat(feat_list, dim=1)
        x = self.conv_out(x)

        x_global = x.max(-1).values

        return x_global

        
class MultiStagePointNetEncoderRGB(nn.Module):
    """
    Multi-Stage PointNet Encoder that supports point cloud with RGB or other features.
    """
    def __init__(self, in_channels=6, h_dim=128, out_channels=128, num_layers=4, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.h_dim = h_dim
        self.out_channels = out_channels
        self.num_layers = num_layers

        cprint(f"[MultiStagePointNetEncoderRGB] in_channels: {self.in_channels}", 'cyan')

        self.act = nn.LeakyReLU(negative_slope=0.0, inplace=False)

        # 这里使用传入的 in_channels (如 6) 替代了写死的 3
        self.conv_in = nn.Conv1d(self.in_channels, h_dim, kernel_size=1)
        
        self.layers, self.global_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(nn.Conv1d(h_dim, h_dim, kernel_size=1))
            self.global_layers.append(nn.Conv1d(h_dim * 2, h_dim, kernel_size=1))
            
        self.conv_out = nn.Conv1d(h_dim * self.num_layers, out_channels, kernel_size=1)

    def forward(self, x):
        # x 输入形状: [B, N, C] 其中 C 是 in_channels (例如 6，代表 XYZ+RGB)
        x = x.transpose(1, 2) # [B, N, C] --> [B, C, N]
        
        y = self.act(self.conv_in(x))
        feat_list = []
        
        for i in range(self.num_layers):
            y = self.act(self.layers[i](y))
            # 提取全局最大特征
            y_global = y.max(-1, keepdim=True).values
            # 拼接局部特征与全局特征
            y = torch.cat([y, y_global.expand_as(y)], dim=1)
            y = self.act(self.global_layers[i](y))
            feat_list.append(y)
            
        # 将多个 stage 的特征拼接在一起
        x = torch.cat(feat_list, dim=1)
        x = self.conv_out(x)

        # 最后的全局最大池化
        x_global = x.max(-1).values

        return x_global

   