import torch.nn as nn
import torch.nn.functional as F
from hmquant.ptq.quantization.histogram_calib import HistPlug
from hmquant.ptq.quantization.quant_functions import *
from hmquant.ptq.nn_layers.base_interface import QuantInterface 
from hmquant.quant_param import QuantParam


class RepVGGQuantConv2d(QuantInterface, nn.Conv2d, HistPlug):
    """
    Normal quantization for conv2d
    calibrating weight and output with thier min max value
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        w_bit=8,
        o_bit=8,
        bias_bit=16,
        need_fake_channelwise_coarseweight=False,
        extract_center=True,
        klconv=True,
        o_abs=True,
    ):
        """
        o_abs: is use .abs() for output for MinMax
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.interface_init()
        self.w_bit = w_bit
        self.o_bit = o_bit
        self.bias_bit = bias_bit
        self.w_qmax = (1 << (self.w_bit - 1)) - 1

        self.bias_qmax = 1 << (self.bias_bit - 1)
        self.w_quant_param = None
        self.o_quant_param = None
        self.x_quant_param = None
        self.bias_quant_param = None
        self.metric = None
        self.output_metric = None
        self.bias_correction = None
        self.w_channelwise = True
        self.conv_as_linear = False
        self.o_abs = o_abs
        self.op_class = "RepVGGQuantConv2d"
        self.step1_out_collect_list = ["min", "max"]
        
        self.need_fake_channelwise_coarseweight = need_fake_channelwise_coarseweight
        self.extract_center = extract_center
        self.klconv = klconv

        self.crosslayer_metric = None
        self.legacy_quant_param = False
        HistPlug.__init__(self) 

    def pre_process(self, *args, **kwargs):
        """
        pre process to support conv as linear
        """
        if self.conv_as_linear:
            args = list(args)
            bs, ic = args[0].size()
            new_x = args[0].view(bs, ic, 1, 1)
            if hasattr(args[0], "quant_param"):
                new_x.quant_param = args[0].quant_param
            args[0] = new_x
        return args, kwargs

    def post_process(self, out, *args, **kwargs):
        """
        post process to support conv as linear
        """
        if self.conv_as_linear:
            bs, oc, _, _ = out.size()
            new_out = out.view(bs, oc)
            if hasattr(out, "quant_param"):
                new_out.quant_param = out.quant_param
            out = new_out
        return out

    def raw_forward(self, x):
        out = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return out

    def quant_weight_bias(self, simulate=True):
        
        w_coarse_sim = self.coarse_w_quant_param.quant_tensor(self.weight_coarse, simulate)
        w_fine_sim = self.fine_w_quant_param.quant_tensor(self.weight_fine, simulate)
        if self.bias is not None:
            if self.bias_quant_param is not None:
                if self.bias_correction:
                    bias = self.bias.data + self.bias_correction
                else:
                    bias = self.bias.data
                bias_sim = self.bias_quant_param.quant_tensor(bias, simulate)
                return w_coarse_sim, w_fine_sim, bias_sim
            else:
                return w_coarse_sim, w_fine_sim, self.bias
        else:
            return w_coarse_sim, w_fine_sim, None

    def quant_forward(self, x):

        w_coarse_sim, w_fine_sim, bias_sim = self.quant_weight_bias()
        if hasattr(self, 'x_scale_factor'):
            x = torch.true_divide(x, self.x_scale_factor)
        coarse_out_tmp = F.conv2d(
            x, w_coarse_sim, bias_sim, self.stride, self.padding, self.dilation, self.groups
        )
        fine_out_tmp = F.conv2d(
            x, w_fine_sim, torch.zeros_like(bias_sim), self.stride, self.padding, self.dilation, self.groups
        )
        out_tmp = coarse_out_tmp + fine_out_tmp
        out_sim = self.o_quant_param.quant_tensor(out_tmp)
        self.ptq_out = out_sim
        return out_sim

    def search_w_bias_quant_param(self, x):
        if self.w_channelwise:  # 为了避免concat操作没有对齐，使得x.quant_param.granularity=="dim1"，所以我们使用channelwise，但是粒度使用per-tensor的。
            if x.quant_param.granularity == "tensor":
                if self.extract_center:
                    self.divide_weight_headache(not self.need_fake_channelwise_coarseweight)
                else:
                    if self.need_fake_channelwise_coarseweight:
                        self.divide_weight_tensorwise()
                    else:
                        self.divide_weight_channelwise()
                coarse_w_quant_scale = (
                    self.weight_coarse.data.abs().amax([1, 2, 3]) / self.w_qmax
                ).detach()
                coarse_w_quant_param = QuantParam(
                    "int8", coarse_w_quant_scale.view(-1), granularity="dim0"
                )
                if self.need_fake_channelwise_coarseweight:
                    coarse_w_quant_param.scale[...] = coarse_w_quant_scale.max()  # fake channelwise, actually per-tensor.
                fine_w_quant_scale = (
                    self.weight_fine.data.abs().amax([1, 2, 3]) / self.w_qmax
                ).detach()
                fine_w_quant_param = QuantParam(
                    "int8", fine_w_quant_scale.view(-1), granularity="dim0"
                )
            elif x.quant_param.granularity == "dim1":
                x_scale_max = x.quant_param.scale.view(1, -1).max()
                self.x_scale_factor = torch.true_divide(x.quant_param.scale.view(1, -1, 1, 1), x_scale_max)
                self.weight = torch.nn.parameter.Parameter(torch.mul(self.weight.data, self.x_scale_factor), requires_grad=True)
                if self.extract_center:
                    self.divide_weight_headache(not self.need_fake_channelwise_coarseweight)
                else:
                    if self.need_fake_channelwise_coarseweight:
                        self.divide_weight_tensorwise()
                    else:
                        self.divide_weight_channelwise()
                x.quant_param.scale[...] = x_scale_max
                x.quant_param.scale = x.quant_param.scale.unique()
                x.quant_param.zero_point = x.quant_param.zero_point.unique()
                x.quant_param.granularity = "tensor"
                x.quant_param.granularity_dims = x.quant_param.granularity_dims[:0]
                coarse_w_quant_scale = (
                    self.weight_coarse.data.abs().amax([1, 2, 3]) / self.w_qmax
                ).detach()
                coarse_w_quant_param = QuantParam(
                    "int8", coarse_w_quant_scale.view(-1), granularity="dim0"
                )
                if self.need_fake_channelwise_coarseweight:
                    coarse_w_quant_param.scale[...] = coarse_w_quant_scale.max()  # fake channelwise, actually per-tensor.
                fine_w_quant_scale = (
                    self.weight_fine.data.abs().amax([1, 2, 3]) / self.w_qmax
                ).detach()
                fine_w_quant_param = QuantParam(
                    "int8", fine_w_quant_scale.view(-1), granularity="dim0"
                )
            else:
                raise NotImplementedError
        else:
            if x.quant_param.granularity == "tensor":
                coarse_w_quant_scale = (self.weight_coarse.data.abs().max() / self.w_qmax).detach()
                coarse_w_quant_param = QuantParam("int8", coarse_w_quant_scale, granularity="tensor")
                fine_w_quant_scale = (self.weight_fine.data.abs().max() / self.w_qmax).detach()
                fine_w_quant_param = QuantParam("int8", fine_w_quant_scale, granularity="tensor")
            else:
                raise NotImplementedError
        if self.bias is not None:
            # self.bias_interval = self.o_interval/8
            bias_scale = (self.bias.abs().max() / self.bias_qmax).detach()
            bias_quant_param = QuantParam(
                f"int{self.bias_bit}", bias_scale, granularity="tensor"
            )
        else:
            bias_quant_param = None
        return coarse_w_quant_param, fine_w_quant_param, bias_quant_param

    def search_o_quant_param(self, x):
        if self.klconv:
            return self.get_quant_param("out_0", bits=self.o_bit)
        else:
            from hmquant.ptq.utils import intf_get_minmax_quant_param
            return intf_get_minmax_quant_param(self, "out_0")

    def divide_weight_tensorwise(self):
        granularity = self.weight.abs().max() / self.w_qmax
        self.weight_coarse = torch.true_divide(self.weight, granularity).add(0.5).floor_() * granularity
        self.weight_fine = torch.sub(input=self.weight, alpha=1, other=self.weight_coarse)
        return granularity

    def divide_weight_channelwise(self):
        granularity_tensor = self.weight.abs().amax([1, 2, 3]) / self.w_qmax
        granularity_tensor = granularity_tensor.view(-1, 1, 1, 1)
        self.weight_coarse = torch.mul(torch.true_divide(self.weight, granularity_tensor).add(0.5).floor_(), granularity_tensor)
        self.weight_fine = torch.sub(input=self.weight, alpha=1, other=self.weight_coarse)
        return granularity_tensor
    
    def divide_weight_headache(self, channlewise: bool=False):
        """This is a divide method for weight which was came up with as the headache'd wrapped around me.
        We first dig the center parameters of the weight, then do divide it to the fine weight and the coarse weight.
        At last, put the fine weight into the kernel and regard it as the fine weight overall, keep the coarse weight.
        Args:
            channlewise (bool, optional): if we divide center parameters with channel-wise, set True, otherwise we use per-tensor quantizaiton. Defaults to False.
        Returns:
            tensor: granularity of central parameters of kernel
        """
        weight_center = self.weight.data[:, :, 1, 1].view(self.weight.data.shape[0], self.weight.data.shape[1], 1, 1)
        weight_center_padded = F.pad(weight_center, pad=(1, 1, 1, 1), mode="constant", value=0)
        weight_outside = torch.sub(self.weight.data, weight_center_padded)
        if channlewise:
            center_granularity_tensor = weight_center.abs().amax([1, 2, 3]) / self.w_qmax
            center_granularity_tensor = center_granularity_tensor.view(-1, 1, 1, 1)
        else:
            center_granularity_tensor = weight_center.abs().max() / self.w_qmax
        weight_center_coarse = torch.mul(torch.true_divide(weight_center, center_granularity_tensor).add(0.5).floor_(), center_granularity_tensor)
        weight_center_fine = torch.sub(input=weight_center, alpha=1, other=weight_center_coarse)
        weight_center_fine_padded = F.pad(weight_center_fine, pad=(1, 1, 1, 1), mode="constant", value=0)
        self.weight_fine = torch.add(input=weight_outside, other=weight_center_fine_padded, alpha=1)
        self.weight_coarse = F.pad(weight_center_coarse, pad=(1, 1, 1, 1), mode="constant", value=0)
        return center_granularity_tensor
    
    def calibration_step2(self, x):
        # step2: search for the best S^w and S^o of each layer
        if not hasattr(x, "quant_param"):
            quant_scale = (x.abs().max() / (self.o_qmax)).detach()
            x_quant_param = QuantParam("int8", quant_scale)
            x = x_quant_param.quant_tensor(x)
            print(f"Warning: x has not quantized, set quant param {x.quant_param}")
        self.coarse_w_quant_param, self.fine_w_quant_param, self.bias_quant_param = self.search_w_bias_quant_param(x)
        self.o_quant_param = self.search_o_quant_param(x)
        out = self.quant_forward(x)
        return out
