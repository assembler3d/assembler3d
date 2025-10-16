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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import craftsman
from craftsman.utils.typing import *


def dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)


class Mesh:
    def __init__(
        self, v_pos: Float[Tensor, "Nv 3"], t_pos_idx: Integer[Tensor, "Nf 3"], **kwargs
    ) -> None:
        self.v_pos: Float[Tensor, "Nv 3"] = v_pos
        self.t_pos_idx: Integer[Tensor, "Nf 3"] = t_pos_idx
        self._v_nrm: Optional[Float[Tensor, "Nv 3"]] = None
        self._v_tng: Optional[Float[Tensor, "Nv 3"]] = None
        self._v_tex: Optional[Float[Tensor, "Nt 3"]] = None
        self._t_tex_idx: Optional[Float[Tensor, "Nf 3"]] = None
        self._v_rgb: Optional[Float[Tensor, "Nv 3"]] = None
        self._edges: Optional[Integer[Tensor, "Ne 2"]] = None
        self.extras: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.add_extra(k, v)

    def add_extra(self, k, v):
        self.extras[k] = v

    def remove_outlier(self, outlier_n_faces_threshold: Union[int, float]):
        if self.requires_grad:
            craftsman.debug("Mesh is differentiable, not removing outliers")
            return self

        # use trimesh to first split the mesh into connected components
        # then remove the components with less than n_face_threshold faces
        import trimesh

        # construct a trimesh object
        mesh = trimesh.Trimesh(
            vertices=self.v_pos.detach().cpu().numpy(),
            faces=self.t_pos_idx.detach().cpu().numpy(),
        )

        # split the mesh into connected components
        components = mesh.split(only_watertight=False)
        # log the number of faces in each component
        craftsman.debug(
            "Mesh has {} components, with faces: {}".format(
                len(components), [c.faces.shape[0] for c in components]
            )
        )

        n_faces_threshold: int
        if isinstance(outlier_n_faces_threshold, float):
            # set the threshold to the number of faces in the largest component multiplied by outlier_n_faces_threshold
            n_faces_threshold = int(
                max([c.faces.shape[0] for c in components]) * outlier_n_faces_threshold
            )
        else:
            # set the threshold directly to outlier_n_faces_threshold
            n_faces_threshold = outlier_n_faces_threshold

        # log the threshold
        craftsman.debug(
            "Removing components with less than {} faces".format(n_faces_threshold)
        )

        # remove the components with less than n_face_threshold faces
        components = [c for c in components if c.faces.shape[0] >= n_faces_threshold]

        # log the number of faces in each component after removing outliers
        craftsman.debug(
            "Mesh has {} components after removing outliers, with faces: {}".format(
                len(components), [c.faces.shape[0] for c in components]
            )
        )
        # merge the components
        mesh = trimesh.util.concatenate(components)

        # convert back to our mesh format
        v_pos = torch.from_numpy(mesh.vertices).to(self.v_pos)
        t_pos_idx = torch.from_numpy(mesh.faces).to(self.t_pos_idx)

        clean_mesh = Mesh(v_pos, t_pos_idx)
        # keep the extras unchanged

        if len(self.extras) > 0:
            clean_mesh.extras = self.extras
            craftsman.debug(
                f"The following extra attributes are inherited from the original mesh unchanged: {list(self.extras.keys())}"
            )
        return clean_mesh

    @property
    def requires_grad(self):
        return self.v_pos.requires_grad

    @property
    def v_nrm(self):
        if self._v_nrm is None:
            self._v_nrm = self._compute_vertex_normal()
        return self._v_nrm

    @property
    def v_rgb(self):
        return self._v_rgb

    @property
    def edges(self):
        if self._edges is None:
            self._edges = self._compute_edges()
        return self._edges

    def _compute_vertex_normal(self):
        i0 = self.t_pos_idx[:, 0]
        i1 = self.t_pos_idx[:, 1]
        i2 = self.t_pos_idx[:, 2]

        v0 = self.v_pos[i0, :]
        v1 = self.v_pos[i1, :]
        v2 = self.v_pos[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)

        # Splat face normals to vertices
        v_nrm = torch.zeros_like(self.v_pos)
        v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        v_nrm = torch.where(
            dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.as_tensor([0.0, 0.0, 1.0]).to(v_nrm)
        )
        v_nrm = F.normalize(v_nrm, dim=1)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(v_nrm))

        return v_nrm

    def set_vertex_color(self, v_rgb):
        assert v_rgb.shape[0] == self.v_pos.shape[0]
        self._v_rgb = v_rgb

    def _compute_edges(self):
        # Compute edges
        edges = torch.cat(
            [
                self.t_pos_idx[:, [0, 1]],
                self.t_pos_idx[:, [1, 2]],
                self.t_pos_idx[:, [2, 0]],
            ],
            dim=0,
        )
        edges = edges.sort()[0]
        edges = torch.unique(edges, dim=0)
        return edges

    def normal_consistency(self) -> Float[Tensor, ""]:
        edge_nrm: Float[Tensor, "Ne 2 3"] = self.v_nrm[self.edges]
        nc = (
            1.0 - torch.cosine_similarity(edge_nrm[:, 0], edge_nrm[:, 1], dim=-1)
        ).mean()
        return nc

        
class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> Float[Tensor, "N 3"]:
        raise NotImplementedError


class MarchingCubeCPUHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        import mcubes

        self.mc_func: Callable = mcubes.marching_cubes
        self._grid_vertices: Optional[Float[Tensor, "N3 3"]] = None
        self._dummy: Float[Tensor, "..."]
        self.register_buffer(
            "_dummy", torch.zeros(0, dtype=torch.float32), persistent=False
        )

    @property
    def grid_vertices(self) -> Float[Tensor, "N3 3"]:
        if self._grid_vertices is None:
            # keep the vertices on CPU so that we can support very large resolution
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
        self,
        level: Float[Tensor, "N3 1"],
        deformation: Optional[Float[Tensor, "N3 3"]] = None,
    ) -> Mesh:
        if deformation is not None:
            craftsman.warn(
                f"{self.__class__.__name__} does not support deformation. Ignoring."
            )
        level = -level.view(self.resolution, self.resolution, self.resolution)
        v_pos, t_pos_idx = self.mc_func(
            level.detach().cpu().numpy(), 0.0
        )  # transform to numpy
        v_pos, t_pos_idx = (
            torch.from_numpy(v_pos).float().to(self._dummy.device),
            torch.from_numpy(t_pos_idx.astype(np.int64)).long().to(self._dummy.device),
        )  # transform back to torch tensor on CUDA
        v_pos = v_pos / (self.resolution - 1.0)
        return Mesh(v_pos=v_pos, t_pos_idx=t_pos_idx)


class MarchingTetrahedraHelper(IsosurfaceHelper):
    def __init__(self, resolution: int, tets_path: str):
        super().__init__()
        self.resolution = resolution
        self.tets_path = tets_path

        self.triangle_table: Float[Tensor, "..."]
        self.register_buffer(
            "triangle_table",
            torch.as_tensor(
                [
                    [-1, -1, -1, -1, -1, -1],
                    [1, 0, 2, -1, -1, -1],
                    [4, 0, 3, -1, -1, -1],
                    [1, 4, 2, 1, 3, 4],
                    [3, 1, 5, -1, -1, -1],
                    [2, 3, 0, 2, 5, 3],
                    [1, 4, 0, 1, 5, 4],
                    [4, 2, 5, -1, -1, -1],
                    [4, 5, 2, -1, -1, -1],
                    [4, 1, 0, 4, 5, 1],
                    [3, 2, 0, 3, 5, 2],
                    [1, 3, 5, -1, -1, -1],
                    [4, 1, 2, 4, 3, 1],
                    [3, 0, 4, -1, -1, -1],
                    [2, 0, 1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1],
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.num_triangles_table: Integer[Tensor, "..."]
        self.register_buffer(
            "num_triangles_table",
            torch.as_tensor(
                [0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long
            ),
            persistent=False,
        )
        self.base_tet_edges: Integer[Tensor, "..."]
        self.register_buffer(
            "base_tet_edges",
            torch.as_tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long),
            persistent=False,
        )

        tets = np.load(self.tets_path)
        self._grid_vertices: Float[Tensor, "..."]
        self.register_buffer(
            "_grid_vertices",
            torch.from_numpy(tets["vertices"]).float(),
            persistent=False,
        )
        self.indices: Integer[Tensor, "..."]
        self.register_buffer(
            "indices", torch.from_numpy(tets["indices"]).long(), persistent=False
        )

        self._all_edges: Optional[Integer[Tensor, "Ne 2"]] = None

    def normalize_grid_deformation(
        self, grid_vertex_offsets: Float[Tensor, "Nv 3"]
    ) -> Float[Tensor, "Nv 3"]:
        return (
            (self.points_range[1] - self.points_range[0])
            / (self.resolution)  # half tet size is approximately 1 / self.resolution
            * torch.tanh(grid_vertex_offsets)
        )  # FIXME: hard-coded activation

    @property
    def grid_vertices(self) -> Float[Tensor, "Nv 3"]:
        return self._grid_vertices

    @property
    def all_edges(self) -> Integer[Tensor, "Ne 2"]:
        if self._all_edges is None:
            # compute edges on GPU, or it would be VERY SLOW (basically due to the unique operation)
            edges = torch.tensor(
                [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3],
                dtype=torch.long,
                device=self.indices.device,
            )
            _all_edges = self.indices[:, edges].reshape(-1, 2)
            _all_edges_sorted = torch.sort(_all_edges, dim=1)[0]
            _all_edges = torch.unique(_all_edges_sorted, dim=0)
            self._all_edges = _all_edges
        return self._all_edges

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)
            b = torch.gather(input=edges_ex2, index=1 - order, dim=1)

        return torch.stack([a, b], -1)

    def _forward(self, pos_nx3, sdf_n, tet_fx4):
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum > 0) & (occ_sum < 4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:, self.base_tet_edges].reshape(-1, 2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
            mapping = (
                torch.ones(
                    (unique_edges.shape[0]), dtype=torch.long, device=pos_nx3.device
                )
                * -1
            )
            mapping[mask_edges] = torch.arange(
                mask_edges.sum(), dtype=torch.long, device=pos_nx3.device
            )
            idx_map = mapping[idx_map]  # map edges to verts

            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, 3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
        edges_to_interp_sdf[:, -1] *= -1

        denominator = edges_to_interp_sdf.sum(1, keepdim=True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1, 6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device=pos_nx3.device))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat(
            (
                torch.gather(
                    input=idx_map[num_triangles == 1],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 1]][:, :3],
                ).reshape(-1, 3),
                torch.gather(
                    input=idx_map[num_triangles == 2],
                    dim=1,
                    index=self.triangle_table[tetindex[num_triangles == 2]][:, :6],
                ).reshape(-1, 3),
            ),
            dim=0,
        )

        return verts, faces

    def forward(
        self,
        level: Float[Tensor, "N3 1"],
        deformation: Optional[Float[Tensor, "N3 3"]] = None,
    ) -> Mesh:
        if deformation is not None:
            grid_vertices = self.grid_vertices + self.normalize_grid_deformation(
                deformation
            )
        else:
            grid_vertices = self.grid_vertices

        v_pos, t_pos_idx = self._forward(grid_vertices, level, self.indices)

        mesh = Mesh(
            v_pos=v_pos,
            t_pos_idx=t_pos_idx,
            # extras
            grid_vertices=grid_vertices,
            tet_edges=self.all_edges,
            grid_level=level,
            grid_deformation=deformation,
        )

        return mesh
