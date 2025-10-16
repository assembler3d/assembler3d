# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import xformers.ops

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0., proj_drop=0., **block_kwargs):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model*2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None):
        # query: img tokens; key/value: condition; mask: if padding tokens
        B, N, C = x.shape
        N_cond = cond.shape[1]

        q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_linear(cond).view(B, N_cond, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        k, v = k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3) # [B, H, N, D]

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).to(torch.bool)
        
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop.p)
        x = x.permute(0,2,1,3).view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with cross attention for conditioning. Adapted from PixArt implementation.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, block_self_attn=False, cond_cross_attn=True, **block_kwargs):
        super().__init__()
        if block_self_attn:
            self.norm_sa = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.block_attn = MultiHeadCrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        else:
            self.norm_sa = None
            self.block_attn = None
        
        if cond_cross_attn:
            self.norm_ca = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        else:
            self.norm_ca = None
            self.cross_attn = None
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, y, t, mask):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)

        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C)
        
        if self.block_attn is not None and self.cross_attn is not None:
            # block-wise self-attention with mask
            x = x + self.block_attn(self.norm_sa(x), self.norm_sa(x), mask)
            x = x + self.cross_attn(self.norm_ca(x), y)
        elif self.block_attn is None and self.cross_attn is not None:
            x = x + self.cross_attn(x, y)
        elif self.block_attn is not None and self.cross_attn is None:
            x = x + self.block_attn(self.norm_sa(x), self.norm_sa(x), mask)
        else:
            raise NotImplementedError
        
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size ** 0.5)
        self.out_channels = out_channels

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels=3,
        hidden_size=768,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        condition_channels=768,
        learn_sigma=True,
        additional_condition_channels=768,
        part_emb=False,
        part_emb_channels=128,
        block_self_attn=False,
        cond_cross_attn=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels + condition_channels + part_emb_channels if part_emb else in_channels + condition_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads
        self.part_emb = part_emb
        self.part_emb_channels = part_emb_channels
        self.cond_cross_attn = cond_cross_attn

        self.x_embedder = nn.Linear(self.in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        if self.cond_cross_attn:
            self.cond_embedder = Mlp(in_features=additional_condition_channels, 
                                hidden_features=hidden_size, out_features=hidden_size, 
                                act_layer=approx_gelu, drop=0)
        else:
            self.cond_embedder = None

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, block_self_attn=block_self_attn, cond_cross_attn=cond_cross_attn) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize label embedding table:
        if self.cond_embedder is not None:
            nn.init.normal_(self.cond_embedder.fc1.weight, std=0.02)
            nn.init.normal_(self.cond_embedder.fc2.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            if self.cond_cross_attn:
                nn.init.constant_(block.cross_attn.proj.weight, 0)
                nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def forward(self, x, t, y, cond, part_ids, mask):
        """
        Forward pass of DiT.
        x: (B, 3, T) tensor of points
        t: (B,) tensor of diffusion timesteps
        y: (B, 1, C) or (B, T, C) tensor of condition point features (point length == input token length)
        cond: (B, M, C) tensor of additional condition, such as text / image features
        part_ids: (B, T) tensor of part ids
        mask: (B, T, T) tensor of attention mask
        """
        # concat condition with input noise
        x = x.permute(0, 2, 1)
        x = torch.cat([x, y], dim=-1)
        # concat part id embedding with input noise
        if self.part_emb:
            part_emb = get_1d_sincos_pos_embed(self.part_emb_channels, part_ids)
            x = torch.cat([x, part_emb], dim=-1)

        x = self.x_embedder(x)                   # (B, T, D), where T is the input token number (params number)
        t = self.t_embedder(t)                   # (B, D)
        t0 = self.t_block(t)
        if self.cond_cross_attn:
            cond = self.cond_embedder(cond)   # (B, M, D)

        # dit blocks
        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, cond, t0, mask)       # (B, T, D)
        x = self.final_layer(x, t)                # (B, T, out_channels)
        return x.permute(0, 2, 1)
    
    def forward_with_cfg(self, x, t, y, cond, part_ids, mask, cfg_scale=1.0):
        # inference batch: [x, y, cond] & [x, y, null]
        single = x[:len(x) // 2]
        combined = torch.cat([single, single], dim=0)
        model_out = self.forward(combined, t, y, cond, part_ids, mask)

        # cfg
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        # modulate
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        # combine
        model_out = torch.cat([eps, rest], dim=1)
        return model_out


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_1d_sincos_pos_embed(embed_dim, pos, base=10000.0):
    """
    Batched 1D sine-cosine positional embeddings (PyTorch version)
    
    Args:
        embed_dim (int): Output embedding dimension (must be even)
        pos (Tensor): Positions tensor of shape (B, M)
        base (float): Frequency scaling base (default: 10000.0)
        interleaved (bool): Whether to interleave sin/cos (default: True)
    
    Returns:
        Tensor: Positional embeddings of shape (B, M, embed_dim)
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even"
    B, M = pos.shape
    
    # Calculate omega terms with matching dtype and device
    omega = torch.arange(embed_dim // 2, dtype=pos.dtype, device=pos.device)
    omega = omega / (embed_dim / 2)  # Normalize to [0, 1]
    omega = 1.0 / (base ** omega)    # Frequency scaling
    
    # Reshape for broadcasting: (B, M, 1) * (1, 1, D/2) -> (B, M, D/2)
    pos = pos.unsqueeze(-1)  # (B, M, 1)
    omega = omega.view(1, 1, -1)  # (1, 1, D/2)
    angles = pos * omega  # (B, M, D/2)
    
    # Generate sine and cosine components
    emb_sin = torch.sin(angles)
    emb_cos = torch.cos(angles)
    
    # Concatenate sin and cos blocks: [sin_block, cos_block]
    emb = torch.cat([emb_sin, emb_cos], dim=-1)  # (B, M, D)
    
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_M(**kwargs):
    # 195M
    return DiT(depth=16, hidden_size=768, num_heads=8, block_self_attn=True, part_emb=True, **kwargs)

def DiT_S(**kwargs):
    # 39M
    return DiT(depth=16, hidden_size=384, num_heads=8, block_self_attn=True, part_emb=True, **kwargs)


DiT_models = {
    'DiT_M': DiT_M,
    'DiT_S': DiT_S,
}
