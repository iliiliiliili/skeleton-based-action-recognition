from typing import Any, List, Optional, Literal, Tuple, Union
import torch
from torch import nn


def multi_output_variational_gaussian_sample(means, stds, global_std_mode, GLOBAL_STD, FIX_GAUSSIAN, split=None):

    if global_std_mode == "replace":
        stds = [GLOBAL_STD for _ in range(len(means))]
    elif global_std_mode == "multiply":
        stds = [GLOBAL_STD * s for s in stds]


    if split is None:
        split = [False for _ in range(len(means))]

    if FIX_GAUSSIAN is None:
        result = [
            (m, s) if spl else m + s * torch.normal(0, torch.ones_like(m)) for m, s, spl in zip(means, stds, split)
        ]
    else:
        result = [
            (m, s) if spl else m + s * FIX_GAUSSIAN * torch.ones_like(m) for m, s, spl in zip(means, stds, split)
        ]
    
    return result

def multi_output_variational_forward(
    means_module,
    stds_module,
    x,
    global_std_mode,
    LOG_STDS,
    FIX_GAUSSIAN,
    GLOBAL_STD,
    end_batch_norm,
    end_activation,
    split=None,
):
    means = means_module(x)

    if stds_module:
        stds = stds_module(x)
    else:
        stds = 0
        
    if LOG_STDS:
        for s in stds:
            pstds = s

            if isinstance(s, (int, float)):
                pstds = torch.tensor(s * 1.0)

            print(
                "std%:",
                abs(
                    float(torch.mean(pstds).detach())
                    / float(torch.mean(means).detach())
                    * 100
                ),
                "std:",
                float(torch.mean(pstds).detach()),
                "mean",
                float(torch.mean(means).detach()),
            )

    result = multi_output_variational_gaussian_sample(
        means, stds, global_std_mode, GLOBAL_STD, FIX_GAUSSIAN, split)

    if end_batch_norm is not None:
        result = [end_batch_norm(r) for r in result]

    if end_activation is not None:
        result = [end_activation(r) for r in result]

    return result


class VariationalBase(nn.Module):
    GLOBAL_STD: float = 0
    LOG_STDS = False
    INIT_WEIGHTS = "usual"
    FIX_GAUSSIAN = None
    ALL_ACTIVATION_MODES = [
        "mean",
        "std",
        "mean+std",
        "end",
        "mean+end",
        "std+end",
        "mean+std+end",
    ]
    ALL_BATCH_NORM_MODES = [
        "mean",
        "std",
        "mean+std",
        "end",
        "mean+end",
        "std+end",
        "mean+std+end",
    ]
    ALL_GLOBAL_STD_MODES = ["none", "replace", "multiply"]

    def __init__(self) -> None:
        super().__init__()

    def build(
        self,
        means: nn.Module,
        stds: Any,
        batch_norm_module: Any,
        batch_norm_size: int,
        activation: Optional[Union[nn.Module, List[nn.Module]]] = None,
        activation_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "mean",
        use_batch_norm: bool = False,
        batch_norm_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "mean",
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        global_std_mode: Union[
            Literal["none"], Literal["replace"], Literal["multiply"]
        ] = "none",
    ) -> None:
        super().__init__()

        self.end_activation = None
        self.end_batch_norm = None

        self.means = means
        self.stds = stds

        self.global_std_mode = global_std_mode

        if use_batch_norm:
            batch_norm_targets = batch_norm_mode.split("+")

            for i, target in enumerate(batch_norm_targets):
                if target == "mean":
                    self.means = nn.Sequential(
                        self.means,
                        batch_norm_module(
                            batch_norm_size,
                            eps=batch_norm_eps,
                            momentum=batch_norm_momentum,
                        ),
                    )
                elif target == "std":
                    if self.stds is not None:
                        self.stds = nn.Sequential(
                            self.stds,
                            batch_norm_module(
                                batch_norm_size,
                                eps=batch_norm_eps,
                                momentum=batch_norm_momentum,
                            ),
                        )
                elif target == "end":
                    self.end_batch_norm = batch_norm_module(
                        batch_norm_size,
                        eps=batch_norm_eps,
                        momentum=batch_norm_momentum,
                    )
                else:
                    raise ValueError("Unknown batch norm target: " + target)

        if activation is not None:
            activation_targets = activation_mode.split("+")

            for i, target in enumerate(activation_targets):
                if len(activation_targets) == 1:
                    current_activation: nn.Module = activation  # type: ignore
                else:
                    current_activation: nn.Module = activation[i]  # type: ignore

                if target == "mean":
                    self.means = nn.Sequential(
                        self.means,
                        current_activation,
                    )
                elif target == "std":
                    if self.stds is not None:
                        self.stds = nn.Sequential(
                            self.stds,
                            current_activation,
                        )
                elif target == "end":
                    self.end_activation = current_activation
                else:
                    raise ValueError("Unknown activation target: " + target)

        self._init_weights()

    def forward(self, x):
        means = self.means(x)

        if self.stds:
            stds = self.stds(x)
        else:
            stds = 0

        if self.global_std_mode == "replace":
            stds = VariationalBase.GLOBAL_STD
        elif self.global_std_mode == "multiply":
            stds = VariationalBase.GLOBAL_STD * stds

        if self.LOG_STDS:
            pstds = stds

            if isinstance(stds, (int, float)):
                pstds = torch.tensor(stds * 1.0)

            print(
                "std%:",
                abs(
                    float(torch.mean(pstds).detach())
                    / float(torch.mean(means).detach())
                    * 100
                ),
                "std:",
                float(torch.mean(pstds).detach()),
                "mean",
                float(torch.mean(means).detach()),
            )

        if VariationalBase.FIX_GAUSSIAN is None:
            result = means + stds * torch.normal(0, torch.ones_like(means))
        else:
            result = means + stds * VariationalBase.FIX_GAUSSIAN * torch.ones_like(
                means
            )

        if self.end_batch_norm is not None:
            result = self.end_batch_norm(result)

        if self.end_activation is not None:
            result = self.end_activation(result)

        return result

    def _init_weights(self):
        init_weights(self)


class MultiOutputVariationalBase(VariationalBase):

    def __init__(self) -> None:
        super().__init__()

        
    def build(
        self,
        means: nn.Module,
        stds: Any,
        batch_norm_module: Any,
        batch_norm_size: int,
        activation: Optional[Union[nn.Module, List[nn.Module]]] = None,
        activation_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "mean",
        use_batch_norm: bool = False,
        batch_norm_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "mean",
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        global_std_mode: Union[
            Literal["none"], Literal["replace"], Literal["multiply"]
        ] = "none",
    ) -> None:
        
        super().build(
            means,
            stds,
            batch_norm_module,
            batch_norm_size,
            activation=activation,
            activation_mode=activation_mode,
            use_batch_norm=use_batch_norm,
            batch_norm_mode=batch_norm_mode,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
            global_std_mode=global_std_mode,
        )

        run_means = lambda *args, **kwargs: means(*args, **kwargs)
        run_stds = lambda *args, **kwargs: stds(*args, **kwargs)

        if isinstance(self.means, nn.Sequential):

            def new_run_means(*args, **kwargs):

                x = self.means[0](*args, **kwargs)

                if isinstance(x, (tuple, list)):
                    for i in range(1, len(self.means)):
                        x = [self.means[i](elem) for elem in x]
                else:
                    for i in range(1, len(self.means)):
                        x = self.means[i](x)
                
                return x

            run_means = new_run_means

        if isinstance(self.stds, nn.Sequential):

            def new_run_stds(*args, **kwargs):

                x = self.stds[0](*args, **kwargs)

                if isinstance(x, (tuple, list)):
                    for i in range(1, len(self.stds)):
                        x = [self.stds[i](elem) for elem in x]
                else:
                    for i in range(1, len(self.stds)):
                        x = self.stds[i](x)
                
                return x

            run_stds = new_run_stds
        
        self.run_means = run_means
        self.run_stds = run_stds



    def forward(self, x):
        return multi_output_variational_forward(
            self.run_means,
            self.run_stds,
            x,
            self.global_std_mode,
            self.LOG_STDS,
            VariationalBase.FIX_GAUSSIAN,
            VariationalBase.GLOBAL_STD,
            self.end_batch_norm,
            self.end_activation,
        )


class VariationalConvolution(VariationalBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Union[Tuple, int] = 1,
        activation: Optional[Union[nn.Module, List[nn.Module]]] = None,
        activation_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "mean",
        use_batch_norm: bool = False,
        batch_norm_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "end",
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        global_std_mode: Union[
            Literal["none"], Literal["replace"], Literal["multiply"]
        ] = "none",
        bias=True,
        **kwargs,
    ) -> None:
        super().__init__()

        if use_batch_norm:
            bias = False

        means = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            **kwargs,
        )

        if global_std_mode == "replace":
            stds = None
        else:
            stds = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                **kwargs,
            )

        super().build(
            means,
            stds,
            nn.BatchNorm2d,
            out_channels,
            activation=activation,
            activation_mode=activation_mode,
            use_batch_norm=use_batch_norm,
            batch_norm_mode=batch_norm_mode,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
            global_std_mode=global_std_mode,
        )


class VariationalLinear(VariationalBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Optional[Union[nn.Module, List[nn.Module]]] = None,
        activation_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "mean",
        use_batch_norm: bool = False,
        batch_norm_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "mean",
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        global_std_mode: Union[
            Literal["none"], Literal["replace"], Literal["multiply"]
        ] = "none",
        bias=True,
        **kwargs,
    ) -> None:
        super().__init__()

        if use_batch_norm:
            bias = False

        means = nn.Linear(in_features, out_features, bias=bias, **kwargs)

        if global_std_mode == "replace":
            stds = None
        else:
            stds = nn.Linear(in_features, out_features, bias=bias, **kwargs)

        super().build(
            means,
            stds,
            nn.BatchNorm1d,
            out_features,
            activation=activation,
            activation_mode=activation_mode,
            use_batch_norm=use_batch_norm,
            batch_norm_mode=batch_norm_mode,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
            global_std_mode=global_std_mode,
        )


class VariationalConvolutionTranspose(VariationalBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Union[Tuple, int] = 1,
        activation: Optional[Union[nn.Module, List[nn.Module]]] = None,
        activation_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "mean",
        use_batch_norm: bool = False,
        batch_norm_mode: Union[
            Literal["mean"],
            Literal["std"],
            Literal["mean+std"],
            Literal["end"],
            Literal["mean+end"],
            Literal["std+end"],
            Literal["mean+std+end"],
        ] = "end",
        batch_norm_eps: float = 1e-3,
        batch_norm_momentum: float = 0.01,
        global_std_mode: Union[
            Literal["none"], Literal["replace"], Literal["multiply"]
        ] = "none",
        bias=True,
        **kwargs,
    ) -> None:
        super().__init__()

        if use_batch_norm:
            bias = False

        means = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            **kwargs,
        )

        if global_std_mode == "replace":
            stds = None
        else:
            stds = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                **kwargs,
            )

        super().build(
            means,
            stds,
            nn.BatchNorm2d,
            out_channels,
            activation=activation,
            activation_mode=activation_mode,
            use_batch_norm=use_batch_norm,
            batch_norm_mode=batch_norm_mode,
            batch_norm_eps=batch_norm_eps,
            batch_norm_momentum=batch_norm_momentum,
            global_std_mode=global_std_mode,
        )


def init_weights(self, all_submodules=None):
    init_type, *params = VariationalBase.INIT_WEIGHTS.split(":")

    if all_submodules is None:
        all_submodules = [
            lambda x: (
                x[0].weight if isinstance(x, torch.nn.Sequential) else x.weight,
                True,
            ),
            lambda x: (
                x[0].bias if isinstance(x, torch.nn.Sequential) else x.bias,
                False,
            ),
        ]

    if init_type == "usual":
        pass
    elif init_type == "fill":
        fill_what = params[0]
        value_kernel = float(params[1])
        value_bias = float(params[2])

        def fill(target):

            for func_submodule in all_submodules:
                submodule, is_weight = func_submodule(target)

                if submodule is not None:
                    submodule.data.fill_(value_kernel if is_weight else value_bias)

        if "mean" in fill_what:
            fill(self.means)

        if "std" in fill_what and self.stds is not None:
            fill(self.stds)
    elif init_type == "xavier_uniform":
        fill_what = params[0]
        gain_kernel = float(params[1])
        gain_bias = float(params[2])

        def fill(target):

            for func_submodule in all_submodules:
                submodule, is_weight = func_submodule(target)

                if submodule is not None:
                    torch.nn.init.xavier_uniform_(
                        submodule, gain=gain_kernel if is_weight else gain_bias
                    )

        if "mean" in fill_what:
            fill(self.means)

        if "std" in fill_what and self.stds is not None:
            fill(self.stds)
    elif init_type == "xavier_uniform_fb":
        fill_what = params[0]
        gain_kernel = float(params[1])
        gain_bias = float(params[2])

        def fill(target):

            for func_submodule in all_submodules:
                submodule, is_weight = func_submodule(target)

                if submodule is not None:
                    if is_weight:
                        torch.nn.init.xavier_uniform_(submodule, gain=gain_kernel)
                    else:
                        submodule.data.fill_(gain_bias)

        if "mean" in fill_what:
            fill(self.means)

        if "std" in fill_what and self.stds is not None:
            fill(self.stds)
    elif init_type == "xavier_uniform_0b":
        fill_what = params[0]
        gain_kernel = float(params[1])
        gain_bias = float(params[2])

        def fill(target):

            for func_submodule in all_submodules:
                submodule, is_weight = func_submodule(target)

                if submodule is not None:
                    if is_weight:
                        torch.nn.init.xavier_uniform_(submodule, gain=gain_kernel)
                    else:
                        torch.nn.init.zeros_(submodule)

        if "mean" in fill_what:
            fill(self.means)

        if "std" in fill_what and self.stds is not None:
            fill(self.stds)
    elif init_type == "xavier_normal":
        fill_what = params[0]
        gain_kernel = float(params[1])
        gain_bias = float(params[2])

        def fill(target):

            for func_submodule in all_submodules:
                submodule, is_weight = func_submodule(target)

                if submodule is not None:
                    torch.nn.init.xavier_normal_(
                        submodule, gain=gain_kernel if is_weight else gain_bias
                    )

        if "mean" in fill_what:
            fill(self.means)

        if "std" in fill_what and self.stds is not None:
            fill(self.stds)
    elif init_type == "xavier_normal_fb":
        fill_what = params[0]
        gain_kernel = float(params[1])
        gain_bias = float(params[2])

        def fill(target):

            for func_submodule in all_submodules:
                submodule, is_weight = func_submodule(target)

                if submodule is not None:
                    if is_weight:
                        torch.nn.init.xavier_normal_(
                            submodule, gain=gain_kernel if is_weight else gain_bias
                        )
                    else:
                        submodule.data.fill_(gain_bias)

        if "mean" in fill_what:
            fill(self.means)

        if "std" in fill_what and self.stds is not None:
            fill(self.stds)
    elif init_type == "xavier_normal_0b":
        fill_what = params[0]
        gain_kernel = float(params[1])
        gain_bias = float(params[2])

        def fill(target):

            for func_submodule in all_submodules:
                submodule, is_weight = func_submodule(target)

                if submodule is not None:
                    if is_weight:
                        torch.nn.init.xavier_normal_(
                            submodule, gain=gain_kernel if is_weight else gain_bias
                        )
                    else:
                        torch.nn.init.zeros_(submodule)

        if "mean" in fill_what:
            fill(self.means)

        if "std" in fill_what and self.stds is not None:
            fill(self.stds)
    else:
        raise ValueError()
