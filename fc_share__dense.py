from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch.nn as nn
import math
import torch
import torch.nn.functional as F

__all__ = ['fc_share__dense']

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers, dropout = 0.1):
        """"Constructor of the class"""
        super(LSTMCell, self).__init__()

        self.nlayers = nlayers
        self.dropout = nn.Dropout(p=dropout)

        ih, hh = [], []
        for i in range(nlayers):
            if i==0:
                ih.append(nn.Linear(input_size, 4 * hidden_size))
                hh.append(nn.Linear(hidden_size, 4 * hidden_size))
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

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.GlobalAvg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        attention = self.GlobalAvg(out)

        out += residual
        out = self.relu(out)

        return out, attention


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.GlobalAvg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # out += residual
        # out = self.relu(out)

        return out, residual

class Attention(nn.Module):
    def __init__(self, ModuleList, block_idx):
        super(Attention, self).__init__()
        self.ModuleList = ModuleList
        self.c2 = 16
        self.c1 = 8
        if block_idx == 1:
            #self.lstm = LSTMCell(64, 64, 1)
            self.sharefc = nn.Sequential(nn.Linear(64,64//self.c), nn.ReLU(inplace=True), nn.Linear(64//self.c,64), nn.Sigmoid())
        elif block_idx == 2:
            self.sharefc = nn.Sequential(nn.Linear(128,128//self.c), nn.ReLU(inplace=True), nn.Linear(128//self.c,128), nn.Sigmoid())
        elif block_idx == 3:
            self.sharefc = nn.Sequential(nn.Linear(256,256//self.c), nn.ReLU(inplace=True), nn.Linear(256//self.c,256), nn.Sigmoid())


            #self.sharefc = nn.Sequential(nn.Linear(64,64//self.c), nn.ReLU(inplace=True), nn.Linear(64//self.c,64), nn.Sigmoid())
            #self.sharefc = nn.Sequential(nn.Linear(128,128//self.c), nn.ReLU(inplace=True), nn.Linear(128//self.c,128), nn.Sigmoid())
            #self.sharefc = nn.Sequential(nn.Linear(256,256//self.c), nn.ReLU(inplace=True), nn.Linear(256//self.c,256), nn.Sigmoid())

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
                attention = self.sharefc(seq)#self.lstm(seq, (ht, ct))  # 1 * batch size * length
                #attention = self.relu(attention)
                # ht = self.sigmoid(ht)
                x = x * (attention.view(attention.size(0), attention.size(1), 1, 1))
                x += org
                x = self.relu(x)
                out_relu = self.GlobalAvg(x)
                list = out_relu.view(out_relu.size(0), 1, out_relu.size(1))
            else:
                seq = self.GlobalAvg(x)
                seq = seq.view(seq.size(0), seq.size(1))
                attention = self.sharefc(seq)
                #attention = self.relu(attention)
                # ht = self.sigmoid(ht)
                x = x * (attention.view(attention.size(0), attention.size(1), 1, 1))
                x += org
                x = self.relu(x)
                out_relu = self.GlobalAvg(x)
                list = torch.cat((list, out_relu.view(out_relu.size(0), 1, out_relu.size(1))), 1)
        return x, list

class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000, block_name='BasicBlock'):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')


        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = Attention(self._make_layer(block, 16, n), 1)
        self.layer2 = Attention(self._make_layer(block, 32, n, stride=2), 2)
        self.layer3 = Attention(self._make_layer(block, 64, n, stride=2), 3)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = nn.ModuleList([])

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x, out1 = self.layer1(x)  # 32x32
        x, out2 = self.layer2(x)  # 16x16
        x, out3 = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        #out1 = torch.nn.functional.normalize(out1, p=2, dim=2)  # batch * 18 * 64
        #out2 = torch.nn.functional.normalize(out2, p=2, dim=2)
        #out3 = torch.nn.functional.normalize(out3, p=2, dim=2)

        return x#, out1, out2, out3


def fc_share__dense(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)

if __name__ == '__main__':
    lstm = LSTMCell(10, 20, 3, 0.1)
    a = torch.randn(7, 10)
    h = torch.randn(3, 7, 20)
    c = torch.randn(3, 7, 20)
    hy, cy = lstm(a,(h,c))