"""
Modified based on: https://github.com/open-mmlab/mmskeleton
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import pad


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def weights_init(module_, bs=1):
    if isinstance(module_, nn.Conv2d) and bs == 1:
        nn.init.kaiming_normal_(module_.weight, mode='fan_out')
        nn.init.constant_(module_.bias, 0)
    elif isinstance(module_, nn.Conv2d) and bs != 1:
        nn.init.normal_(module_.weight, 0,
                        math.sqrt(2. / (module_.weight.size(0) * module_.weight.size(1) * module_.weight.size(2) * bs)))
        nn.init.constant_(module_.bias, 0)
    elif isinstance(module_, nn.BatchNorm2d):
        nn.init.constant_(module_.weight, bs)
        nn.init.constant_(module_.bias, 0)
    elif isinstance(module_, nn.Linear):
        nn.init.normal_(module_.weight, 0, math.sqrt(2. / bs))


class Graph3DConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, A, cuda_, temporal_kernel_size=3, temporal_stride=1, temporal_padding=1):
        super(Graph3DConvolution, self).__init__()
        self.cuda_ = cuda_
        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_stride = temporal_stride
        self.temporal_padding = temporal_padding

        self.graph_attn = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.graph_attn, 1)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = 3
        self.g_conv = nn.ModuleList()
        for i in range(self.num_subset):
            self.g_conv.append(nn.Conv2d(in_channels, out_channels, 1))
            weights_init(self.g_conv[i], bs=self.num_subset)

        if in_channels != out_channels:
            self.gcn_residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (temporal_stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
            weights_init(self.gcn_residual[0], bs=1)
            weights_init(self.gcn_residual[1], bs=1)
        else:
            self.gcn_residual = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        weights_init(self.bn, bs=1e-6)
        self.relu = nn.ReLU()

    def forward(self, x):
        N, C, T, V = x.size()

        temporal_patches = pad(x, (0, 0, self.temporal_padding, self.temporal_padding), "constant", 0).unfold(2, self.temporal_kernel_size, self.temporal_stride)
        temporal_patches_count = temporal_patches.shape[2]
        temporal_patches = temporal_patches.reshape(N, C, temporal_patches_count, -1)
        temporal_patches_squeezed = temporal_patches.view(N, C * temporal_patches_count, -1)

        if self.cuda_:
            A = self.A.cuda(x.get_device())
        else:
            A = self.A
        A = A * self.graph_attn
        
        temporal_adjacency = A.repeat(1, self.temporal_kernel_size, 1)
        
        hidden_ = None
        for i in range(self.num_subset):
            z = self.g_conv[i](torch.matmul(temporal_patches_squeezed, temporal_adjacency[i]).view(N, C, temporal_patches_count, -1))
            hidden_ = z + hidden_ if hidden_ is not None else z
        hidden_ = self.bn(hidden_)
        residual = self.gcn_residual(x)
        residual = residual[:, :, :hidden_.shape[2], :]
        hidden_ += residual
        return self.relu(hidden_)


class TemporalConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(TemporalConvolution, self).__init__()

        pad = int((kernel_size - 1) / 2)
        self.t_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                                padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        weights_init(self.t_conv, bs=1)
        weights_init(self.bn, bs=1)

    def forward(self, x):
        x = self.bn(self.t_conv(x))
        return x


class ST_3DGCNT_block(nn.Module):
    def __init__(self, in_channels, out_channels, A, cuda_=False, residual=True, temporal_kernel_size=3, temporal_stride=1, temporal_padding=1):
        super(ST_3DGCNT_block, self).__init__()

        self.gcn = Graph3DConvolution(in_channels, out_channels, A, cuda_, temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=temporal_padding)
        self.tcn = TemporalConvolution(out_channels, out_channels, stride=1)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (temporal_stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConvolution(in_channels, out_channels, kernel_size=1, stride=temporal_stride)

    def forward(self, x):
        features = self.tcn(self.gcn(x))
        residual = self.residual(x)
        result = features + (residual[:, :, :features.shape[2], :] if isinstance(residual, torch.Tensor) else residual)
        return self.relu(result)


class ST3DGCNT(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 cuda_=True, temporal_kernel_size=3, temporal_stride=1, temporal_padding=1):
        super(ST3DGCNT, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        weights_init(self.data_bn, bs=1)

        # self.layers = nn.ModuleDict(
        #     {'layer1': ST_3DGCNT_block(in_channels, 64, A, cuda_, residual=False, temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=temporal_padding),
        #      'layer2': ST_3DGCNT_block(64, 64, A, cuda_, temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=temporal_padding),
        #      'layer3': ST_3DGCNT_block(64, 64, A, cuda_, temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=temporal_padding),
        #      'layer4': ST_3DGCNT_block(64, 64, A, cuda_, temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=temporal_padding),
        #      'layer5': ST_3DGCNT_block(64, 128, A, cuda_, temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=temporal_padding),
        #      'layer6': ST_3DGCNT_block(128, 128, A, cuda_, temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=temporal_padding),
        #      'layer7': ST_3DGCNT_block(128, 128, A, cuda_, temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=temporal_padding),
        #      'layer8': ST_3DGCNT_block(128, 256, A, cuda_, temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=temporal_padding),
        #      'layer9': ST_3DGCNT_block(256, 256, A, cuda_, temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=temporal_padding),
        #      'layer10': ST_3DGCNT_block(256, 256, A, cuda_, temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=temporal_padding)}
        # )
        self.layers = nn.ModuleDict(
            {'layer1': ST_3DGCNT_block(in_channels, 64, A, cuda_, residual=False, temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=temporal_padding),
             'layer2': ST_3DGCNT_block(64, 128, A, cuda_, temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=temporal_padding),
             'layer3': ST_3DGCNT_block(128, 256, A, cuda_, temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=temporal_padding)}
        )

        self.fc = nn.Linear(256, num_class)
        weights_init(self.fc, bs=num_class)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for i in range(len(self.layers)):
           x = self.layers['layer' + str(i+1)](x)
        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        return self.fc(x)
