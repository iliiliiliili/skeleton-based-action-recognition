"""
Modified based on: https://github.com/open-mmlab/mmskeleton
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .variational import (
    VariationalBase,
    VariationalConvolution,
    init_weights as vnn_init_weights,
)
from .vnn_agcn import VariationalGraphConvolution, VariationalTemporalConvolution, VariationalStgcnBlock
from typing import Any, Optional, List

NUM_SUBSET = 3


def import_class(name: str) -> Any:
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def weights_init(module_: nn.Module, bs: float = 1) -> None:
    if isinstance(module_, nn.Conv2d) and bs == 1:
        nn.init.kaiming_normal_(module_.weight, mode="fan_out")
        nn.init.constant_(module_.bias, 0)
    elif isinstance(module_, nn.Conv2d) and bs != 1:
        nn.init.normal_(
            module_.weight,
            0,
            math.sqrt(
                2.0
                / (
                    module_.weight.size(0)
                    * module_.weight.size(1)
                    * module_.weight.size(2)
                    * bs
                )
            ),
        )
        nn.init.constant_(module_.bias, 0)
    elif isinstance(module_, nn.BatchNorm2d):
        nn.init.constant_(module_.weight, bs)
        nn.init.constant_(module_.bias, 0)
    elif isinstance(module_, nn.Linear):
        nn.init.normal_(module_.weight, 0, math.sqrt(2.0 / bs))



class InternalUncertaintyGraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, A, cuda_, coff_embedding=4):
        super(InternalUncertaintyGraphConvolution, self).__init__()
        self.cuda_ = cuda_
        self.graph_attn = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        nn.init.constant_(self.graph_attn, 1e-6)
        self.A = Variable(
            torch.from_numpy(A.astype(np.float32)), requires_grad=False
        )
        self.num_subset = NUM_SUBSET
        self.g_conv = nn.ModuleList()
        self.a_conv = nn.ModuleList()
        self.b_conv = nn.ModuleList()
        for i in range(self.num_subset):
            self.g_conv.append(nn.Conv2d(in_channels, out_channels, 1))
            self.a_conv.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.b_conv.append(nn.Conv2d(in_channels, inter_channels, 1))
            weights_init(self.g_conv[i], bs=self.num_subset)
            weights_init(self.a_conv[i])
            weights_init(self.b_conv[i])

        if in_channels != out_channels:
            self.gcn_residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
            weights_init(self.gcn_residual[0], bs=1)
            weights_init(self.gcn_residual[1], bs=1)
        else:
            self.gcn_residual = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        weights_init(self.bn, bs=1e-6)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(-2)

    def forward(self, x: torch.Tensor, x_var: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.size()
        if self.cuda_:
            A = self.A.cuda(x.get_device())
        else:
            A = self.A
        A = A + self.graph_attn
        hidden_ = None
        for i in range(self.num_subset):
            A1 = (
                self.a_conv[i](x)
                .permute(0, 3, 1, 2)
                .contiguous()
                .view(N, V, self.inter_c * T)
            )
            A2 = self.b_conv[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            x_a = x_var.view(N, C * T, V)
            z = self.g_conv[i](torch.matmul(x_a, A1).view(N, C, T, V))
            hidden_ = z + hidden_ if hidden_ is not None else z
        hidden_ = self.bn(hidden_)
        hidden_ += self.gcn_residual(x)
        return hidden_


class InternalUncertaintyVariationalStgcnBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        cuda_=False,
        stride=1,
        residual=True,
        **kwargs
    ):
        super().__init__()

        self.gcn = InternalUncertaintyGraphConvolution(
            in_channels, out_channels, A, cuda_, **kwargs
        )
        self.tcn = VariationalTemporalConvolution(
            out_channels, out_channels, stride=stride, **kwargs
        )
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = VariationalTemporalConvolution(
                in_channels, out_channels, kernel_size=1, stride=stride
            )

    def forward(self, x: torch.Tensor, x_var: torch.Tensor) -> torch.Tensor:

        result = self.gcn(x, x_var)
        result = self.tcn(result)

        result += self.residual(x)
        result = self.relu(result)

        return result


class IUVAGCN(nn.Module):
    def __init__(
        self,
        num_class: int = 60,
        num_point: int = 25,
        num_person: int = 2,
        graph: Optional[str] = None,
        graph_args: dict = dict(),
        in_channels: int = 3,
        cuda_: bool = True,
        FIX_GAUSSIAN: Optional[float] = None,
        INIT_WEIGHTS: str = "usual",
        samples: int = 4,
        test_samples: int = 4,
        iu_layers: List[int] = [5,10],
    ) -> None:
        super(IUVAGCN, self).__init__()

        self.default_samples = samples
        self.test_samples = test_samples
        self.iu_layers = iu_layers
        self.LAYER_COUNT = 10 # fixed

        VariationalBase.FIX_GAUSSIAN = FIX_GAUSSIAN
        VariationalBase.INIT_WEIGHTS = INIT_WEIGHTS

        if VariationalBase.FIX_GAUSSIAN is not None:
            print("FIX_GAUSSIAN", VariationalBase.FIX_GAUSSIAN)
            print("FIX_GAUSSIAN", VariationalBase.FIX_GAUSSIAN)
            print("FIX_GAUSSIAN", VariationalBase.FIX_GAUSSIAN)
            print("FIX_GAUSSIAN", VariationalBase.FIX_GAUSSIAN)

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        weights_init(self.data_bn, bs=1)

        def get_block(layer_index):
            if layer_index in self.iu_layers:
                return InternalUncertaintyVariationalStgcnBlock
            else:
                return VariationalStgcnBlock


        self.layers = nn.ModuleDict(
            {
                "layer1": get_block(1)(
                    in_channels, 64, A, cuda_, residual=False
                ),
                "layer2": get_block(2)(64, 64, A, cuda_),
                "layer3": get_block(3)(64, 64, A, cuda_),
                "layer4": get_block(4)(64, 64, A, cuda_),
                "layer5": get_block(5)(64, 128, A, cuda_, stride=2),
                "layer6": get_block(6)(128, 128, A, cuda_),
                "layer7": get_block(7)(128, 128, A, cuda_),
                "layer8": get_block(8)(128, 256, A, cuda_, stride=2),
                "layer9": get_block(9)(256, 256, A, cuda_),
                "layer10": get_block(10)(256, 256, A, cuda_),
            }
        )

        self.fc = nn.Linear(256, num_class)
        weights_init(self.fc, bs=num_class)

    def forward(
        self, 
        x: torch.Tensor,
        samples: Optional[int] = None,
        combine_predictions: bool = True
    ) -> torch.Tensor:

        if samples is None:
            if self.training:
                samples = self.default_samples
            else:
                samples = self.test_samples

        # print('data size', x.size())
        N, C, T, V, M = x.size()
        # x = x[:, :3, :, :, :]  # for mediapipe
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = (
            x.view(N, M, V, C, T)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(N * M, C, T, V)
        )

        current_layer_index = 1 # 1-indexed
        iu_x = [x for _ in range(samples)]

        while current_layer_index <= self.LAYER_COUNT:

            iu_layer_index = None
            outputs = []

            for s in range(samples):

                current_x = iu_x[s]
                should_go_to_next = False

                for i in range(current_layer_index, self.LAYER_COUNT + 1):

                    current_layer = self.layers["layer" + str(i)]

                    if isinstance(current_layer, InternalUncertaintyVariationalStgcnBlock):
                        iu_layer_index = i

                        outputs.append(current_x)
                        should_go_to_next = True
                        break
                    else:
                        current_x = current_layer(current_x)
                
                if should_go_to_next:
                    continue

                # N*M,C,T,V
                c_new = current_x.size(1)
                current_x = current_x.view(N, M, c_new, -1)
                current_x = current_x.mean(3).mean(1)
                current_x = self.fc(current_x)
                outputs.append(current_x)
            
            if iu_layer_index is None:
                break
            else:

                iu_x = []

                for s in range(samples):

                    x_var, x = torch.var_mean(torch.stack(outputs, dim=0), dim=0, unbiased=False)
                    current_x = self.layers["layer" + str(iu_layer_index)](x, x_var)

                    current_layer_index = iu_layer_index + 1

                    if current_layer_index > self.LAYER_COUNT:
                        # N*M,C,T,V
                        c_new = current_x.size(1)
                        current_x = current_x.view(N, M, c_new, -1)
                        current_x = current_x.mean(3).mean(1)
                        current_x = self.fc(current_x)

                    iu_x.append(current_x)
                
                if current_layer_index > self.LAYER_COUNT:
                    outputs = iu_x

                
        result_var, result = torch.var_mean(torch.stack(outputs, dim=0), dim=0, unbiased=False)

        return result #, result_var
