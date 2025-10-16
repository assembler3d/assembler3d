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

import torch
import torch.nn as nn
from torch import distributed as tdist
from torch.nn import functional as F
import math
import mcubes
import numpy as np
from einops import repeat, rearrange
from skimage import measure

from craftsman.utils.base import BaseModule
from craftsman.utils.typing import *
from craftsman.utils.misc import get_world_size
from craftsman.utils.ops import generate_dense_grid_points
import time
import matplotlib.pyplot as plt
from math import ceil
from matplotlib.transforms import Affine2D
from joblib import Parallel, delayed
from functools import partial
import trimesh
from tqdm import trange
from functools import partial
from itertools import product

def unique_in_chunk(chunk):
    return np.unique(chunk,axis=0)

VALID_EMBED_TYPES = ["identity", "fourier", "hashgrid", "sphere_harmonic", "triplane_fourier"]
RES = 1 
LINE_COLOR = "red"

def scale_tensor(
    dat, inp_scale, tgt_scale
):
    if inp_scale is None:
        inp_scale = (0, 1)
    if tgt_scale is None:
        tgt_scale = (0, 1)
    if isinstance(tgt_scale, Tensor):
        assert dat.shape[-1] == tgt_scale.shape[-1]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_point(self):
        return self.x, self.y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __lt__(self, other):
        return self.x < other.x and self.y < other.y

    def __gt__(self, other):
        return self.x > other.x and self.y > other.y

    def __mul__(self, other):
        if type(other) == Point:
            return Point(self.x * other.x, self.y * other.y)
        else:
            return Point(self.x * other, self.y * other)

def getState(a, b, c, d):
    return a * 8 + b * 4 + c * 2 + d

def draw_line(point_1, point_2):
    plt.plot([point_1.x, point_2.x], [point_1.y, point_2.y], color=LINE_COLOR, linewidth=1)

def linear_interpolation(point_1, point_2, a_val, b_val):
    t = (0.5 - a_val) / (b_val - a_val)

    return point_1 + (point_2 - point_1) * t


def draw_seperator_line(a, b, c, d, grid):
    grid[grid<=-1] = -0.99
    grid[grid>=1] = 0.99
    a_val = grid[a.x // RES][a.y // RES]
    b_val = grid[b.x // RES][b.y // RES]
    c_val = grid[c.x // RES][c.y // RES]
    d_val = grid[d.x // RES][d.y // RES]

    state = getState(ceil(a_val), ceil(b_val), ceil(c_val), ceil(d_val))

    
    if state == 1 or state == 14:
        draw_line(a - Point(0, 0.5 * RES), d + Point(0.5 * RES, 0))
    elif state == 2 or state == 13:
        draw_line(b - Point(0, 0.5 * RES), c - Point(0.5 * RES, 0))
    elif state == 3 or state == 12:
        draw_line(a - Point(0, 0.5 * RES), b - Point(0, 0.5 * RES))
    elif state == 4 or state == 11:
        draw_line(a + Point(0.5 * RES, 0), b - Point(0, 0.5 * RES))
    elif state == 5:
        draw_line(a + Point(0.5 * RES, 0), a - Point(0, 0.5 * RES))
        draw_line(b - Point(0, 0.5 * RES), c - Point(0.5 * RES, 0))
    elif state == 6 or state == 9:
        draw_line(a + Point(0.5 * RES, 0), c - Point(0.5 * RES, 0))
    elif state == 7 or state == 8:
        draw_line(a + Point(0.5 * RES, 0), a - Point(0, 0.5 * RES))
    elif state == 10:
        draw_line(a + Point(0.5 * RES, 0), b - Point(0, 0.5 * RES))
        draw_line(a - Point(0, 0.5 * RES), d + Point(0.5 * RES, 0))
    elif state == 0 or state == 15:
        pass
    else:
        raise Exception(f"Invalid state {state}")
    
class FourierEmbedder(nn.Module):
    def __init__(self,
                 num_freqs: int = 6,
                 logspace: bool = True,
                 input_dim: int = 3,
                 include_input: bool = True,
                 include_pi: bool = True) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                num_freqs,
                dtype=torch.float32
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (num_freqs - 1),
                num_freqs,
                dtype=torch.float32
            )

        if include_pi:
            frequencies *= torch.pi

        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs

        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)

        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x


    
def get_embedder(embed_type="fourier", num_freqs=-1, input_dim=3, include_pi=True):
    if embed_type == "identity" or (embed_type == "fourier" and num_freqs == -1):
        return nn.Identity(), input_dim

    elif embed_type == "fourier":
        embedder_obj = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

    elif embed_type == "learned_fourier":
        embedder_obj = LearnedFourierEmbedder(in_channels=input_dim, dim=num_freqs)
    
    elif embed_type == "siren":
        embedder_obj = Siren(in_dim=input_dim, out_dim=num_freqs * input_dim * 2 + input_dim)
    
    elif embed_type == "hashgrid":
        embedder_obj = HashEmbedder(input_dims = 3, output_dims=48)

    elif embed_type == "sphere_harmonic":
        raise NotImplementedError

    else:
        raise ValueError(f"{embed_type} is not valid. Currently only supprts {VALID_EMBED_TYPES}")
    return embedder_obj
    

###################### AutoEncoder
class AutoEncoder(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = ""
        num_latents: int = 256
        embed_dim: int = 64
        width: int = 768
        
    cfg: Config

    def configure(self) -> None:
        super().configure()

    def encode(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        raise NotImplementedError

    def decode(self, z: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

    def encode_kl_embed(self, latents: torch.FloatTensor, sample_posterior: bool = True):
        posterior = None
        if self.cfg.embed_dim > 0: # 64
            moments = self.pre_kl(latents) # 103，256，768 -》 103，256，128
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
            if sample_posterior:
                kl_embed = posterior.sample() # 1，768，64
            else:
                kl_embed = posterior.mode()
        else:
            kl_embed = latents
        return kl_embed, posterior
    
    def forward(self,
                coarse_surface: torch.FloatTensor,
                sharp_surface: torch.FloatTensor,
                queries: torch.FloatTensor,
                sample_posterior: bool = True,
                split: str ="val"):
                
        self.split = split
        shape_latents, kl_embed, posterior = self.encode(coarse_surface, sharp_surface, sample_posterior=sample_posterior)
        latents = self.decode(kl_embed) 
        logits = self.query(queries, latents) 

        mean_value = torch.mean(kl_embed).item()
        variance_value = torch.var(kl_embed).item()


        return shape_latents, latents, posterior, logits, mean_value, variance_value
    
    def query(self, queries: torch.FloatTensor, latents: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError

    @torch.no_grad()
    def extract_geometry_by_diffdmc(self,
                         latents: torch.FloatTensor,
                         bounds: Union[Tuple[float], List[float], float] = (-1.05, -1.05, -1.05, 1.05, 1.05, 1.05),
                         octree_depth: int = 8,
                         num_chunks: int = 10000,
                         ):
        
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min

        xyz_samples, grid_size, length,xs,ys,zs = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_depth=octree_depth,
            indexing="ij"
        )
        xyz_samples = torch.FloatTensor(xyz_samples)
        batch_size = latents.shape[0]

        batch_logits = []
        
        for start in trange(0, xyz_samples.shape[0], num_chunks):
            queries = xyz_samples[start: start + num_chunks, :].to(latents.device)
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size)
            logits = self.query(batch_queries, latents)
            batch_logits.append(logits)
        grid_logits = torch.cat(batch_logits, dim=1).view((batch_size, grid_size[0], grid_size[1], grid_size[2])).float()

        mesh_v_f = []
        has_surface = np.zeros((batch_size,), dtype=np.bool_)
        diffdmc = DiffDMC(dtype=torch.float32).to(latents.device)
        for i in range(batch_size):
            try:
                vertices, faces = diffdmc(-grid_logits[i], isovalue=0, normalize=False)
                vertices = (vertices + 1) / grid_size[0] * bbox_size[0] + bbox_min[0]
                mesh_v_f.append((vertices, faces))
                has_surface[i] = True
            except:
                mesh_v_f.append((None, None))
                has_surface[i] = False

        return mesh_v_f, has_surface

    @torch.no_grad()
    def extract_geometry(self,
                         latents: torch.FloatTensor,
                         bounds: Union[Tuple[float], List[float], float] = (-1.05, -1.05, -1.05, 1.05, 1.05, 1.05),
                         octree_depth: int = 8,
                         num_chunks: int = 10000,
                         save_slice_dir: str = ''
                         ):
        
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min

        xyz_samples, grid_size, length,xs,ys,zs = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_depth=octree_depth,
            indexing="ij"
        ) 
        xyz_samples = torch.FloatTensor(xyz_samples)
        batch_size = latents.shape[0] 

        batch_logits = []
        for start in range(0, xyz_samples.shape[0], num_chunks):
            queries = xyz_samples[start: start + num_chunks, :].to(latents.device) 
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size) 

            logits = self.query(batch_queries, latents) 
            batch_logits.append(logits.cpu())

        grid_logits = torch.cat(batch_logits, dim=1).view((batch_size, grid_size[0], grid_size[1], grid_size[2])).float().numpy() * 100# 1，513，513，513

        mesh_v_f = []
        has_surface = np.zeros((batch_size,), dtype=np.bool_)
        for i in range(batch_size):
            try:
                vertices, faces, normals, _ = measure.marching_cubes(grid_logits[i], 0, method="lewiner")
                vertices = vertices / grid_size * bbox_size + bbox_min
                faces = faces[:, [2, 1, 0]]
                mesh_v_f.append((vertices.astype(np.float32), np.ascontiguousarray(faces)))
                has_surface[i] = True
            except:
                mesh_v_f.append((None, None))
                has_surface[i] = False

        return mesh_v_f, has_surface



class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: Union[torch.Tensor, List[torch.Tensor]], deterministic=False, feat_dim=1):
        self.feat_dim = feat_dim
        self.parameters = parameters

        if isinstance(parameters, list):
            self.mean = parameters[0]
            self.logvar = parameters[1]
        else:
            self.mean, self.logvar = torch.chunk(parameters, 2, dim=feat_dim)

        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl(self, other=None, dims=(1, 2)):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                        + self.var - 1.0 - self.logvar,
                                        dim=dims)
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=dims)

    def nll(self, sample, dims=(1, 2)):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean
