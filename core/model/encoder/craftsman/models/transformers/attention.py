#  Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from craftsman.utils.typing import *
from craftsman.utils.checkpoint import checkpoint

from .utils import init_linear, MLP
from timm.models.vision_transformer import Attention


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
        qkv_bias: bool,
        use_flash: bool = False
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(heads=heads, n_ctx=n_ctx, use_flash=use_flash)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        x = checkpoint(self.attention, (x,), (), True)
        x = self.c_proj(x)
        return x


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, heads: int, n_ctx: int, use_flash: bool = False):
        super().__init__()
        self.heads = heads
        self.n_ctx = n_ctx
        self.use_flash = use_flash

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        if self.use_flash:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            out = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(bs, n_ctx, -1)
        else:
            weight = torch.einsum(
                "bthc,bshc->bhts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            wdtype = weight.dtype
            weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
            out = torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

        return out

class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
        qkv_bias: bool = True,
        use_flash: bool = False,
        use_checkpoint: bool = False
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.attn = MultiheadAttention(
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash
        )
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = MLP(width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width)

    def _forward(self, x: torch.Tensor): # 103，256，768
        x = x + self.attn(self.ln_1(x)) # 103，256，768
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward(self, x: torch.Tensor):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)


class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        init_scale: float,
        qkv_bias: bool = True,
        use_flash: bool = False,
        n_data: Optional[int] = None,
        data_width: Optional[int] = None,
    ):
        super().__init__()
        self.n_data = n_data
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(self.data_width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadCrossAttention(
            heads=heads, n_data=n_data, use_flash=use_flash
        )
        init_linear(self.c_q, init_scale)
        init_linear(self.c_kv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x, data): # x: 103，256，768.  data:103,4096,768
        x = self.c_q(x) # 103，256，768
        data = self.c_kv(data) # 103,4096,1536
        x = checkpoint(self.attention, (x, data), (), True) # 103，256，768
        x = self.c_proj(x) # 103，256，768
        return x


class QKVMultiheadCrossAttention(nn.Module):
    def __init__(self, *, heads: int, use_flash: bool = False, n_data: Optional[int] = None):

        super().__init__()
        self.heads = heads
        self.n_data = n_data
        self.use_flash = use_flash

    def forward(self, q, kv):
        _, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        q = q.view(bs, n_ctx, self.heads, -1) # 103,256,12,64.  12是head的数量
        kv = kv.view(bs, n_data, self.heads, -1) # 103,4096,12,128
        k, v = torch.split(kv, attn_ch, dim=-1) # 103,4096,12,64.  103,4096,12,64

        if self.use_flash:
            
            q = q.permute(0, 2, 1, 3) # 103,12,256,64
            k = k.permute(0, 2, 1, 3) # 103,12,4096,64
            v = v.permute(0, 2, 1, 3) # 103，12，4096，64
            out = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1, 3).reshape(bs, n_ctx, -1) # 103，256，768
        else:
            weight = torch.einsum(
                "bthc,bshc->bhts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            wdtype = weight.dtype
            weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
            out = torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

        return out


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_data: Optional[int] = None,
        width: int,
        heads: int,
        data_width: Optional[int] = None,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        use_flash: bool = False,
        use_checkpoint: bool = False
    ):
        super().__init__()

        if data_width is None:
            data_width = width
        self.use_checkpoint = use_checkpoint
        self.attn = MultiheadCrossAttention(
            n_data=n_data,
            width=width,
            heads=heads,
            data_width=data_width,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
        )
        self.ln_1 = nn.LayerNorm(width)
        self.ln_2 = nn.LayerNorm(data_width)
        self.mlp = MLP(width=width, init_scale=init_scale)
        self.ln_3 = nn.LayerNorm(width)
    def _forward(self, x: torch.Tensor, data: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_2(data)) # x是query 103，256，768,   data是point clound  103,4096,768
        x = x + self.mlp(self.ln_3(x))
        return x
    def forward(self, x: torch.Tensor, data: torch.Tensor):
        return checkpoint(self._forward, (x, data), self.parameters(), self.use_checkpoint)