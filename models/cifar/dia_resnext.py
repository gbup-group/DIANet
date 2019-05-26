from __future__ import division
""" 
Creates a ResNeXt Model as defined in:
Xie, S., Girshick, R., Dollar, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.
import from https://github.com/prlz77/ResNeXt.pytorch/blob/master/models/model.py
"""
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch

__all__ = ['dia_resnext']

class small_cell(nn.Module):
    def __init__(self, input_size, hidden_size):
        """"Constructor of the class"""
        super(small_cell, self).__init__()
        self.seq = nn.Sequential(nn.Linear(input_size, input_size // 4),
                      nn.ReLU(inplace=True),
                      nn.Linear(input_size // 4, 4 * hidden_size))
    def forward(self,x):
        return self.seq(x)

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers, dropout = 0.1):
        """"Constructor of the class"""
        super(LSTMCell, self).__init__()

        self.nlayers = nlayers
        self.dropout = nn.Dropout(p=dropout)

        ih, hh = [], []
        for i in range(nlayers):
            if i==0:
                # ih.append(nn.Linear(input_size, 4 * hidden_size))
                ih.append(small_cell(input_size, hidden_size))
                # hh.append(nn.Linear(hidden_size, 4 * hidden_size))
                hh.append(small_cell(hidden_size, hidden_size))
            else:
                ih.append(nn.Linear(hidden_size, 4 * hidden_size))
                hh.append(nn.Linear(hidden_size, 4 * hidden_size))
        self.w_ih = nn.ModuleList(ih)
        self.w_hh = nn.ModuleList(hh)

    def forward(self, input, hidden):
        """"Defines the forward computation of the LSTMCell"""
        hy, cy = [], []
        for i in range(self.nlayers):
            hx, cx = hidden[0][i], hidden[1][i]
            gates = self.w_ih[i](input) + self.w_hh[i](hx)
            i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)
            i_gate = torch.sigmoid(i_gate)
            f_gate = torch.sigmoid(f_gate)
            c_gate = torch.tanh(c_gate)
            o_gate = torch.sigmoid(o_gate)
            ncx = (f_gate * cx) + (i_gate * c_gate)
            # nhx = o_gate * torch.tanh(ncx)
            nhx = o_gate * torch.sigmoid(ncx)
            cy.append(ncx)
            hy.append(nhx)
            input = self.dropout(nhx)

        hy, cy = torch.stack(hy, 0), torch.stack(cy, 0)  # number of layer * batch * hidden
        return hy, cy

class Attention(nn.Module):
    def __init__(self, ModuleList, block_idx):
        super(Attention, self).__init__()
        self.ModuleList = ModuleList
        if block_idx == 1:
            # self.lstm = nn.LSTMCell(64, 64)
            # self.sigmoid = nn.Sequential(nn.Linear(64,64), nn.Sigmoid())
            self.lstm = LSTMCell(128, 128, 1)
        elif block_idx == 2:
            # self.lstm = nn.LSTMCell(128, 128)
            # self.sigmoid = nn.Sequential(nn.Linear(128, 128), nn.Sigmoid())
            self.lstm = LSTMCell(256, 256, 1)
        elif block_idx == 3:
            # self.lstm = nn.LSTMCell(256, 256)
            # self.sigmoid = nn.Sequential(nn.Linear(256, 256), nn.Sigmoid())
            self.lstm = LSTMCell(512, 512, 1)

        self.GlobalAvg = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.block_idx = block_idx

    def forward(self, x):
        for idx, layer in enumerate(self.ModuleList):
            x, org = layer(x)  # 64 128 256   BatchSize * NumberOfChannels * 1 * 1
             # BatchSize * NumberOfChannels
            if idx == 0:
                seq = self.GlobalAvg(x)
                seq = seq.view(seq.size(0), seq.size(1))
                ht = torch.zeros(1, seq.size(0), seq.size(1)).cuda()  # 1 mean number of layers
                ct = torch.zeros(1, seq.size(0), seq.size(1)).cuda()
                ht, ct = self.lstm(seq, (ht, ct))  # 1 * batch size * length
                # ht = self.sigmoid(ht)
                x = x * (ht[-1].view(ht.size(1), ht.size(2), 1, 1))
                x += org
                x = self.relu(x)
                out_relu = self.GlobalAvg(x)
                # list = out_relu.view(out_relu.size(0), 1, out_relu.size(1))
            else:
                seq = self.GlobalAvg(x)
                seq = seq.view(seq.size(0), seq.size(1))
                ht, ct = self.lstm(seq, (ht, ct))
                # ht = self.sigmoid(ht)
                x = x * (ht[-1].view(ht.size(1), ht.size(2), 1, 1))
                x += org
                x = self.relu(x)
                out_relu = self.GlobalAvg(x)
                # list = torch.cat((list, out_relu.view(out_relu.size(0), 1, out_relu.size(1))), 1)
        return x#, list

class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """
    def __init__(self, in_channels, out_channels, stride, cardinality, widen_factor):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        D = cardinality * out_channels // widen_factor
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        # return F.relu(residual + bottleneck, inplace=True)
        return bottleneck, residual


class CifarResNeXt(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """
    def __init__(self, cardinality, depth, num_classes, widen_factor=4, dropRate=0):
        """ Constructor
        Args:
            cardinality: number of convolution groups.
            depth: number of layers.
            num_classes: number of classes
            widen_factor: factor to adjust the channel dimensionality
        """
        super(CifarResNeXt, self).__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.widen_factor = widen_factor
        self.num_classes = num_classes
        self.output_size = 64
        self.stages = [32, 32 * self.widen_factor, 64 * self.widen_factor, 128 * self.widen_factor]

        self.conv_1_3x3 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(32)
        self.stage_1 = Attention(self.block('stage_1', self.stages[0], self.stages[1], 1), 1)
        self.stage_2 = Attention(self.block('stage_2', self.stages[1], self.stages[2], 2), 2)
        self.stage_3 = Attention(self.block('stage_3', self.stages[2], self.stages[3], 2), 3)
        self.classifier = nn.Linear(512, num_classes)
        init.kaiming_normal_(self.classifier.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def block(self, name, in_channels, out_channels, pool_stride=2):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            name: string name of the current block.
            in_channels: number of input channels
            out_channels: number of output channels
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        block = nn.ModuleList([])
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.append(ResNeXtBottleneck(in_channels, out_channels, pool_stride, self.cardinality,
                                                          self.widen_factor))
            else:
                block.append(ResNeXtBottleneck(out_channels, out_channels, 1, self.cardinality, self.widen_factor))
        return block

    def forward(self, x):
        x = self.conv_1_3x3.forward(x)
        x = F.relu(self.bn_1.forward(x), inplace=True)
        x = self.stage_1.forward(x) # batch size * 256 * 32 * 32
        x = self.stage_2.forward(x) # batch size * 512 * 16 * 16
        x = self.stage_3.forward(x) # batch size * 1024 * 8 * 8
        
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, 512)

        # out1 = torch.nn.functional.normalize(out1, p=2, dim=2)
        # out2 = torch.nn.functional.normalize(out2, p=2, dim=2)
        # out3 = torch.nn.functional.normalize(out3, p=2, dim=2)

        return self.classifier(x) # , out1, out2, out3

def dia_resnext(**kwargs):
    """Constructs a ResNeXt.
    """
    model = CifarResNeXt(**kwargs)
    return model

if __name__ == '__main__':
    model = CifarResNeXt(8, 101, 100, 4).cuda()

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    a = torch.randn((2, 3, 32, 32)).cuda()
    model(a)