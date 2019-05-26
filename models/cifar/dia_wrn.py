import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['dia_wrn']

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
            self.lstm = LSTMCell(64, 64, 1)
        elif block_idx == 2:
            # self.lstm = nn.LSTMCell(128, 128)
            # self.sigmoid = nn.Sequential(nn.Linear(128, 128), nn.Sigmoid())
            self.lstm = LSTMCell(128, 128, 1)
        elif block_idx == 3:
            # self.lstm = nn.LSTMCell(256, 256)
            # self.sigmoid = nn.Sequential(nn.Linear(256, 256), nn.Sigmoid())
            self.lstm = LSTMCell(256, 256, 1)


        self.GlobalAvg = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.block_idx = block_idx

    def forward(self, x):
        for idx, layer in enumerate(self.ModuleList):
            x, org = layer(x)  # 64 128 256   BatchSize * NumberOfChannels * 1 * 1
             # BatchSize * NumberOfChannels
            if idx == 0:
                seq = self.GlobalAvg(x)
                # list = seq.view(seq.size(0), 1, seq.size(1))
                seq = seq.view(seq.size(0), seq.size(1))
                ht = torch.zeros(1, seq.size(0), seq.size(1)).cuda()  # 1 mean number of layers
                ct = torch.zeros(1, seq.size(0), seq.size(1)).cuda()
                ht, ct = self.lstm(seq, (ht, ct))  # 1 * batch size * length
                # ht = self.sigmoid(ht)
                x = x * (ht[-1].view(ht.size(1), ht.size(2), 1, 1))
                x += org
                # x = self.relu(x)
            else:
                seq = self.GlobalAvg(x)
                # list = torch.cat((list, seq.view(seq.size(0), 1, seq.size(1))), 1)
                seq = seq.view(seq.size(0), seq.size(1))
                ht, ct = self.lstm(seq, (ht, ct))
                # ht = self.sigmoid(ht)
                x = x * (ht[-1].view(ht.size(1), ht.size(2), 1, 1))
                x += org
                # x = self.relu(x)
                # print(self.block_idx, idx, ht)
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.bn_additional = nn.BatchNorm2d(out_planes)
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        out = self.bn_additional(out)
        if not self.equalInOut:
            x = self.convShortcut(x)
        return out, x

# class NetworkBlock(nn.Module):
#     def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
#         super(NetworkBlock, self).__init__()
#         self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
#     def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
#         layers = []
#         for i in range(nb_layers):
#             layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
#         return nn.Sequential(*layers)
#     def forward(self, x):
#         return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = Attention(self.NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate),1)
        # 2nd block
        self.block2 = Attention(self.NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate),2)
        # 3rd block
        self.block3 = Attention(self.NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate),3)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def  NetworkBlock(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        return self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = nn.ModuleList([])
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return layers

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

def dia_wrn(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet(**kwargs)
    return model

if __name__ == '__main__':
    model = WideResNet(52, 100, 4, 0.3).cuda()

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    a = torch.randn((2, 3, 32, 32)).cuda()
    model(a)