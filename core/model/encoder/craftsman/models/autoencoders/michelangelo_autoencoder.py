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
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from einops import repeat, rearrange

import craftsman
from craftsman.models.transformers.perceiver_1d import Perceiver
from craftsman.models.transformers.attention import ResidualCrossAttentionBlock
from craftsman.utils.checkpoint import checkpoint
from craftsman.utils.base import BaseModule
from craftsman.utils.typing import *

from .utils import AutoEncoder, FourierEmbedder, get_embedder
import numpy as np
class PerceiverCrossAttentionEncoder(nn.Module):
    def __init__(self,
                 use_downsample: bool,
                 num_latents: int,
                 embedder: FourierEmbedder,
                 point_feats: int,
                 embed_point_feats: bool,
                 width: int,
                 heads: int,
                 layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 use_ln_post: bool = False,
                 use_flash: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents
        self.use_downsample = use_downsample
        self.embed_point_feats = embed_point_feats

        if not self.use_downsample:
            self.query = nn.Parameter(torch.randn((num_latents, width)) * 0.02)

        self.embedder = embedder
        if self.embed_point_feats:
            self.input_proj = nn.Linear(self.embedder.out_dim * 2, width)
        else:
            self.input_proj = nn.Linear(self.embedder.out_dim + point_feats, width)
            self.input_proj1 = nn.Linear(self.embedder.out_dim + point_feats, width)

        self.cross_attn = ResidualCrossAttentionBlock(
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_checkpoint=False
        )
        
        self.cross_attn1 = ResidualCrossAttentionBlock( 
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_checkpoint=False
        )

        self.self_attn = Perceiver(
            n_ctx=num_latents,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_checkpoint=use_checkpoint
        )

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width)
        else:
            self.ln_post = None

    def _forward(self, coarse_pc, sharp_pc, coarse_feats , sharp_feats, split):
        bs, N_coarse, D_coarse = coarse_pc.shape
        bs, N_sharp, D_sharp = sharp_pc.shape
        
        coarse_data = self.embedder(coarse_pc)  
        if coarse_feats is not None:
            if self.embed_point_feats:
                coarse_feats = self.embedder(coarse_feats)
            coarse_data = torch.cat([coarse_data, coarse_feats], dim=-1) 

        coarse_data = self.input_proj(coarse_data) 
        
        sharp_data = self.embedder(sharp_pc)
        if sharp_feats is not None:
            if self.embed_point_feats:
                sharp_feats = self.embedder(sharp_feats)
            sharp_data = torch.cat([sharp_data, sharp_feats], dim=-1) 
        sharp_data = self.input_proj1(sharp_data) 

        if self.use_downsample:
            ###### fps
            #tokens = np.array([128.0,256.0,384.0,512.0,640.0,1024.0,2048.0])
            tokens = np.array([4.0, 8.0, 16.0, 24.0, 36.0, 64.0, 128.0])
            
            coarse_ratios = tokens/ N_coarse
            sharp_ratios = tokens/ N_sharp
            if split =='val':
                probabilities = np.array([0,0,0,0,0,1,0]) 
            elif split =='train':
                probabilities = np.array([ 0.1,0.1,0.1,0.1,0.1,0.3,0.2])
            ratio_coarse = np.random.choice(coarse_ratios, size=1, p=probabilities)[0]
            index = np.where(coarse_ratios == ratio_coarse)[0]
            ratio_sharp = sharp_ratios[index].item()

            flattened = coarse_pc.view(bs*N_coarse, D_coarse) 
            batch = torch.arange(bs).to(coarse_pc.device) 
            batch = torch.repeat_interleave(batch, N_coarse) 
            pos = flattened
            idx = fps(pos, batch, ratio=ratio_coarse)  
            query_coarse = coarse_data.view(bs*N_coarse, -1)[idx].view(bs, -1, coarse_data.shape[-1]) 

            flattened = sharp_pc.view(bs*N_sharp, D_sharp) 
            batch = torch.arange(bs).to(sharp_pc.device) 
            batch = torch.repeat_interleave(batch, N_sharp) 
            pos = flattened
            idx = fps(pos, batch, ratio=ratio_sharp) 
            query_sharp = sharp_data.view(bs*N_sharp, -1)[idx].view(bs, -1, sharp_data.shape[-1]) 

            query = torch.cat([query_coarse, query_sharp], dim=1)
            print('query shape',f'{query.shape}')
        else:
            query = self.query 
            query = repeat(query, "m c -> b m c", b=bs) 


        latents_coarse = self.cross_attn(query, coarse_data) 
        latents_sharp=  self.cross_attn1(query, sharp_data)
        latents = latents_coarse + latents_sharp

        latents = self.self_attn(latents) 
        if self.ln_post is not None:
            latents = self.ln_post(latents) 

        return latents
    

    def encoder_with_query(self, coarse_pc, coarse_feats, query_ind):
        bs, N_coarse, D_coarse = coarse_pc.shape
        
        coarse_data = self.embedder(coarse_pc)  
        if coarse_feats is not None:
            if self.embed_point_feats:
                coarse_feats = self.embedder(coarse_feats)
            coarse_data = torch.cat([coarse_data, coarse_feats], dim=-1) 

        coarse_data = self.input_proj(coarse_data) 
        
        sharp_data = self.embedder(coarse_pc)
        if coarse_feats is not None:
            sharp_feats = coarse_feats
            if self.embed_point_feats:
                sharp_feats = self.embedder(coarse_feats)
            sharp_data = torch.cat([sharp_data, sharp_feats], dim=-1) 
        sharp_data = self.input_proj1(sharp_data) 

        query = coarse_data[:,query_ind,:] 

        latents_coarse = self.cross_attn(query, coarse_data) 
        latents_sharp=  self.cross_attn1(query, sharp_data)
        latents = latents_coarse + latents_sharp

        latents = self.self_attn(latents) 
        if self.ln_post is not None:
            latents = self.ln_post(latents) 

        return latents
        
    def forward(self, coarse_pc: torch.FloatTensor, sharp_pc: torch.FloatTensor, coarse_feats: Optional[torch.FloatTensor] = None, sharp_feats: Optional[torch.FloatTensor] = None, split: str = 'val'):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:
            dict
        """

        return self._forward(coarse_pc, sharp_pc, coarse_feats, sharp_feats,split)


class PerceiverCrossAttentionDecoder(nn.Module):

    def __init__(self,
                 num_latents: int,
                 out_dim: int,
                 embedder: FourierEmbedder,
                 width: int,
                 heads: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 use_flash: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.embedder = embedder

        self.query_proj = nn.Linear(self.embedder.out_dim, width)

        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            n_data=num_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_flash=use_flash,
            use_checkpoint=use_checkpoint
        )

        self.ln_post = nn.LayerNorm(width)
        self.output_proj = nn.Linear(width, out_dim)

    def _forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        queries = self.query_proj(self.embedder(queries))
        x = self.cross_attn_decoder(queries, latents)
        x = self.ln_post(x)
        x = self.output_proj(x)
        return x

    def forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        logits = checkpoint(self._forward, (queries, latents), self.parameters(), self.use_checkpoint)
        return logits


@craftsman.register("michelangelo-autoencoder")
class MichelangeloAutoencoder(AutoEncoder):
    r"""
    A VAE model for encoding shapes into latents and decoding latent representations into shapes.
    """

    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = ""
        use_downsample: bool = False
        num_latents: int = 256
        point_feats: int = 0
        embed_point_feats: bool = False
        out_dim: int = 1
        embed_dim: int = 64
        embed_type: str = "fourier"
        num_freqs: int = 8
        include_pi: bool = True
        width: int = 768
        heads: int = 12
        num_encoder_layers: int = 8
        num_decoder_layers: int = 16
        init_scale: float = 0.25
        qkv_bias: bool = True
        use_ln_post: bool = False
        use_flash: bool = False
        use_checkpoint: bool = True

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.embedder = get_embedder(embed_type=self.cfg.embed_type, num_freqs=self.cfg.num_freqs, include_pi=self.cfg.include_pi)
        self.cfg.init_scale = self.cfg.init_scale * math.sqrt(1.0 / self.cfg.width)
        self.encoder = PerceiverCrossAttentionEncoder(
            use_downsample=self.cfg.use_downsample,
            embedder=self.embedder,
            num_latents=self.cfg.num_latents,
            point_feats=self.cfg.point_feats,
            embed_point_feats=self.cfg.embed_point_feats,
            width=self.cfg.width,
            heads=self.cfg.heads,
            layers=self.cfg.num_encoder_layers,
            init_scale=self.cfg.init_scale,
            qkv_bias=self.cfg.qkv_bias,
            use_ln_post=self.cfg.use_ln_post,
            use_flash=self.cfg.use_flash,
            use_checkpoint=self.cfg.use_checkpoint
        )

        if self.cfg.embed_dim > 0:
            # VAE embed
            self.pre_kl = nn.Linear(self.cfg.width, self.cfg.embed_dim * 2)
            self.post_kl = nn.Linear(self.cfg.embed_dim, self.cfg.width)
            self.latent_shape = (self.cfg.num_latents, self.cfg.embed_dim)
        else:
            self.latent_shape = (self.cfg.num_latents, self.cfg.width)

        self.transformer = Perceiver(
            n_ctx=self.cfg.num_latents,
            width=self.cfg.width,
            layers=self.cfg.num_decoder_layers,
            heads=self.cfg.heads,
            init_scale=self.cfg.init_scale,
            qkv_bias=self.cfg.qkv_bias,
            use_flash=self.cfg.use_flash,
            use_checkpoint=self.cfg.use_checkpoint
        )

        # decoder
        self.decoder = PerceiverCrossAttentionDecoder(
            embedder=self.embedder,
            out_dim=self.cfg.out_dim,
            num_latents=self.cfg.num_latents,
            width=self.cfg.width,
            heads=self.cfg.heads,
            init_scale=self.cfg.init_scale,
            qkv_bias=self.cfg.qkv_bias,
            use_flash=self.cfg.use_flash,
            use_checkpoint=self.cfg.use_checkpoint
        )


        if self.cfg.pretrained_model_name_or_path != "":
            print(f"Loading pretrained shape model from {self.cfg.pretrained_model_name_or_path}")
            pretrained_ckpt = torch.load(self.cfg.pretrained_model_name_or_path, map_location="cpu", weights_only=False)
            if 'state_dict' in pretrained_ckpt:
                _pretrained_ckpt = {}
                for k, v in pretrained_ckpt['state_dict'].items():
                    if k.startswith('shape_model.'):
                        _pretrained_ckpt[k.replace('shape_model.', '')] = v
                pretrained_ckpt = _pretrained_ckpt
            else:
                _pretrained_ckpt = {}
                for k, v in pretrained_ckpt.items():
                    if k.startswith('shape_model.'):
                        _pretrained_ckpt[k.replace('shape_model.', '')] = v
                pretrained_ckpt = _pretrained_ckpt
                
            self.load_state_dict(pretrained_ckpt, strict=True)
            
    
    def encode(self,
               coarse_surface: torch.FloatTensor,
               sharp_surface: torch.FloatTensor,
               sample_posterior: bool = True,
               split='val'
               ):
        """
        Args:
            surface (torch.FloatTensor): [B, N, 3+C]
            sample_posterior (bool):

        Returns:
            shape_latents (torch.FloatTensor): [B, num_latents, width]
            kl_embed (torch.FloatTensor): [B, num_latents, embed_dim]
            posterior (DiagonalGaussianDistribution or None):
        """
        coarse_pc, coarse_feats = coarse_surface[..., :3], coarse_surface[..., 3:]
        sharp_pc, sharp_feats = sharp_surface[..., :3], sharp_surface[..., 3:] 
        shape_latents = self.encoder(coarse_pc, sharp_pc, coarse_feats ,sharp_feats,split=split)
        kl_embed, posterior = self.encode_kl_embed(shape_latents, sample_posterior)


        return shape_latents, kl_embed, posterior
    
    def encode_with_query(self,
               coarse_surface: torch.FloatTensor,
               query_ind: torch.IntTensor,
               sample_posterior: bool = True,
               split='val'
               ):
        """
        Args:
            surface (torch.FloatTensor): [B, N, 3+C]
            sample_posterior (bool):

        Returns:
            shape_latents (torch.FloatTensor): [B, num_latents, width]
            kl_embed (torch.FloatTensor): [B, num_latents, embed_dim]
            posterior (DiagonalGaussianDistribution or None):
        """
        coarse_pc, coarse_feats = coarse_surface[..., :3], coarse_surface[..., 3:]
        shape_latents = self.encoder.encoder_with_query(coarse_pc, coarse_feats, query_ind)
        kl_embed, posterior = self.encode_kl_embed(shape_latents, sample_posterior)


        return shape_latents, kl_embed, posterior


    def decode(self, 
               latents: torch.FloatTensor):
        """
        Args:
            latents (torch.FloatTensor): [B, embed_dim]

        Returns:
            latents (torch.FloatTensor): [B, embed_dim]
        """
        latents = self.post_kl(latents) # [B, num_latents, embed_dim] -> [B, num_latents, width]

        return self.transformer(latents)


    def query(self, 
              queries: torch.FloatTensor, 
              latents: torch.FloatTensor):
        """
        Args:
            queries (torch.FloatTensor): [B, N, 3]
            latents (torch.FloatTensor): [B, embed_dim]

        Returns:
            logits (torch.FloatTensor): [B, N], occupancy logits
        """
        logits = self.decoder(queries, latents).squeeze(-1)

        return logits
