import torch
import torch.nn as nn

from transformers.activations import ACT2FN

def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

def linear_forward_int4(x, weight_int4pack, scales_and_zeros, out_features, groupsize):
    origin_x_size = x.size()
    x = x.reshape(-1, origin_x_size[-1])
    c = torch.ops.aten._weight_int4pack_mm(x, weight_int4pack, groupsize, scales_and_zeros)
    new_shape = origin_x_size[:-1] + (out_features,)
    c = c.reshape(new_shape)
    return c

def _check_linear_int4_k(k, groupsize = 1, inner_k_tiles = 1):
    return k % groupsize == 0 and k % (inner_k_tiles * 16) == 0

def get_group_qparams(w, n_bit=4, groupsize=128):
    # needed for GPTQ with padding
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    zeros = min_val + scales * (2 ** (n_bit - 1))
    return scales.to(torch.bfloat16).reshape(w.shape[0], -1), zeros.to(
        torch.bfloat16
    ).reshape(w.shape[0], -1)

def group_quantize_tensor_from_qparams(w, scales, zeros, n_bit=4, groupsize=128):
    assert groupsize > 1
    # needed for GPTQ single column quantize
    if groupsize > w.shape[-1] and scales.shape[-1] == 1:
        groupsize = w.shape[-1]

    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)
    min_val = zeros - scales * (2 ** (n_bit - 1))
    max_int = 2**n_bit - 1
    min_int = 0
    w_int32 = (
        to_quant.sub(min_val)
        .div(scales)
        .round()
        .clamp_(min_int, max_int)
        .to(torch.int32)
        .reshape_as(w)
    )

    return w_int32

def pack_scales_and_zeros(scales, zeros):
    assert scales.shape == zeros.shape
    assert scales.dtype == torch.bfloat16
    assert zeros.dtype == torch.bfloat16
    return (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )

def group_quantize_tensor(w, n_bit=4, groupsize=128):
    scales, zeros = get_group_qparams(w, n_bit, groupsize)
    w_int32 = group_quantize_tensor_from_qparams(w, scales, zeros, n_bit, groupsize)
    scales_and_zeros = pack_scales_and_zeros(scales, zeros)
    return w_int32, scales_and_zeros

def prepare_int4_weight_and_scales_and_zeros(weight_bf16, groupsize, inner_k_tiles):
    weight_int32, scales_and_zeros = group_quantize_tensor(
        weight_bf16, n_bit=4, groupsize=groupsize
    )
    weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(weight_int32, inner_k_tiles)
    return weight_int4pack, scales_and_zeros

def quant_weight_int4(weight):

    groupsize = 128
    inner_k_tiles = 8
    padding = True

    out_features, in_features = weight.shape

    if not _check_linear_int4_k(in_features, groupsize, inner_k_tiles):
        if padding:
            import torch.nn.functional as F
            print(f"warning: {fqn} is padded to satisfy in_features % 1024 == 0")
            padded_in_features = find_multiple(in_features, 1024)
            weight = F.pad(weight, pad=(0, padded_in_features - in_features))
        else:
            print(f"warning: {fqn} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, " +
                "and that groupsize and inner_k_tiles*16 evenly divide into it")
            return weight, None
        
    weight_int4pack, scales_and_zeros = prepare_int4_weight_and_scales_and_zeros(
        weight.to(torch.bfloat16).to(device="cuda"), groupsize, inner_k_tiles
    )
    
    return weight_int4pack, scales_and_zeros


class WeightOnlyInt4Linear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    dtype = torch.bfloat16

    def __init__(
            self, in_features: int, out_features: int,
            bias=True, device=None, dtype=None, groupsize: int = 128, inner_k_tiles: int = 8, padding: bool = True,
    ) -> None:
        super().__init__()
        self.padding = padding
        if padding:
            # from model import find_multiple
            self.origin_in_features = in_features
            in_features = find_multiple(in_features, 1024)

        self.in_features = in_features
        self.out_features = out_features
        assert not bias, "require bias=False"
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles

        assert out_features % 8 == 0, "require out_features % 8 == 0"
        assert in_features % (inner_k_tiles * 16) == 0, "require in_features % (innerKTiles * 16) == 0"
        self.register_buffer(
            "weight",
            torch.empty((out_features // 8, in_features // (inner_k_tiles * 16), 32, inner_k_tiles // 2), dtype=torch.int32)
        )
        self.register_buffer(
            "scales_and_zeros",
            torch.empty((in_features // groupsize, out_features, 2), dtype=torch.bfloat16)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(torch.bfloat16)
        if self.padding:
            import torch.nn.functional as F
            input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        return linear_forward_int4(
            input,
            self.weight, self.scales_and_zeros, self.out_features, self.groupsize
        )


class DeepseekV2MLP_int4(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        groupsize = 128
        inner_k_tiles = 8
        padding = True

        if _check_linear_int4_k(self.hidden_size, groupsize, inner_k_tiles):
            self.gate_proj = WeightOnlyInt4Linear(self.hidden_size, self.intermediate_size, bias=False, groupsize=groupsize, inner_k_tiles=inner_k_tiles, padding=False,)
            self.up_proj = WeightOnlyInt4Linear(self.hidden_size, self.intermediate_size, bias=False, groupsize=groupsize, inner_k_tiles=inner_k_tiles, padding=False,)
        elif padding:
            self.gate_proj = WeightOnlyInt4Linear(self.hidden_size, self.intermediate_size, bias=False, groupsize=groupsize, inner_k_tiles=inner_k_tiles, padding=True,)
            self.up_proj = WeightOnlyInt4Linear(self.hidden_size, self.intermediate_size, bias=False, groupsize=groupsize, inner_k_tiles=inner_k_tiles, padding=True,)

        if _check_linear_int4_k(self.intermediate_size, groupsize, inner_k_tiles):
            self.down_proj = WeightOnlyInt4Linear(self.intermediate_size, self.hidden_size, bias=False, groupsize=groupsize, inner_k_tiles=inner_k_tiles, padding=False,)
        elif padding:
            self.down_proj = WeightOnlyInt4Linear(self.intermediate_size, self.hidden_size, bias=False, groupsize=groupsize, inner_k_tiles=inner_k_tiles, padding=True,)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
