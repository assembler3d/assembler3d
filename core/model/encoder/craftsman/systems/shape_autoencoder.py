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
from dataclasses import dataclass, field
import numpy as np
import torch
from skimage import measure
from einops import repeat, rearrange

import craftsman
from craftsman.systems.base import BaseSystem
from craftsman.utils.ops import generate_dense_grid_points
from craftsman.utils.typing import *
from craftsman.utils.misc import get_rank
import torch.distributed as dist
import os
from craftsman.models.geometry.utils import Mesh
import trimesh
from torchvision.utils import save_image,make_grid
from craftsman.systems.utils import *
import time


@craftsman.register("shape-autoencoder-system")
class ShapeAutoEncoderSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        shape_model_type: str = None
        shape_model: dict = field(default_factory=dict)
        sample_posterior: bool = True

    cfg: Config

    def configure(self):
        super().configure()

        self.shape_model = craftsman.find(self.cfg.shape_model_type)(self.cfg.shape_model)

    def forward(self, batch: Dict[str, Any],split: str) -> Dict[str, Any]:
        num = batch["number_sharp"]
        rand_points = batch["rand_points"] 
        if "sdf" in batch:
                target = batch["sdf"]
                coarse_target = target[:,num:]
                sharp_target = target[:,:num]
                criteria = torch.nn.MSELoss()
        elif "occupancies" in batch:
            target = batch["occupancies"] 
            coarse_target = target[:,num:]
            sharp_target = target[:,:num]
            criteria = torch.nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError
        _, latents, posterior, logits, mean_value, variance_value = self.shape_model(
            batch["coarse_surface"],  # xyz + normal
            batch["sharp_surface"],  # xyz + normal
            rand_points, 
            sample_posterior=self.cfg.sample_posterior,
            split= split
        )
        if self.cfg.sample_posterior:
            loss_kl = posterior.kl()
            loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            coarse_logits = logits[:,num:]
            sharp_logits = logits[:,:num]
            return {
                "loss_coarse_logits": criteria(coarse_logits, coarse_target).mean(),
                "loss_sharp_logits": criteria(sharp_logits, sharp_target).mean(),
                "loss_kl": loss_kl,
                "overall_logits": logits,
                "coarse_logits": coarse_logits,
                "sharp_logits": sharp_logits,
                "overall_target": target,
                "coarse_target": coarse_target,
                "sharp_target": sharp_target,
                "latents": latents,
                "mean_value": mean_value,
                "variance_value": variance_value
            }
        else:
            coarse_logits = logits[:,num:]
            sharp_logits = logits[:,:num]
            return {
                "loss_coarse_logits": criteria(coarse_logits, coarse_target).mean(),
                "loss_sharp_logits": criteria(sharp_logits, sharp_target).mean(),
                "overall_logits": logits,
                "coarse_logits": coarse_logits,
                "sharp_logits": sharp_logits,
                "overall_target": target,
                "coarse_target": coarse_target,
                "sharp_target": sharp_target,
                "latents": latents,
                "mean_value": mean_value,
                "variance_value": variance_value
            }

    def training_step(self, batch, batch_idx):
        """
        Description:

        Args:
            batch:
            batch_idx:
        Returns:
            loss:
        """
        out = self(batch,'train')

        loss = 0.
        for name, value in out.items():
            if name.startswith("loss_"):
                self.log(f"train/{name}", value)
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()
        device = batch['coarse_surface'].device
        out = self(batch,'val')
        try:
            mesh_v_f, has_surface = self.shape_model.extract_geometry_by_diffdmc(out["latents"],octree_depth=9)
            file_path = f"it{self.true_global_step}/{os.path.basename(batch['uid'][0])}".replace(".npz",".obj")
            if not os.path.exists(file_path):
                self.save_mesh(
                    file_path,
                    mesh_v_f[0][0], mesh_v_f[0][1]
            )
        except Exception as e:
            print(f"ERROR: in processing batch: {batch}. Error: {e}")
            return

        threshold = 0
        overall_outputs = out["overall_logits"]
        overall_labels = out["overall_target"]
        overall_labels = (overall_labels >= threshold).float()
        overall_pred = torch.zeros_like(overall_outputs)
        overall_pred[overall_outputs>=threshold] = 1
        overall_accuracy = (overall_pred==overall_labels).float().sum(dim=1) / overall_labels.shape[1]
        overall_accuracy = overall_accuracy.mean()
        overall_intersection = (overall_pred * overall_labels).sum(dim=1)
        overall_union = (overall_pred + overall_labels).gt(0).sum(dim=1)
        overall_iou = overall_intersection * 1.0 / overall_union + 1e-5
        overall_iou = overall_iou.mean()

        coarse_outputs = out["coarse_logits"]
        coarse_labels = out["coarse_target"]
        coarse_labels = (coarse_labels >= threshold).float()
        coarse_pred = torch.zeros_like(coarse_outputs)
        coarse_pred[coarse_outputs>=threshold] = 1
        coarse_accuracy = (coarse_pred==coarse_labels).float().sum(dim=1) / coarse_labels.shape[1]
        coarse_accuracy = coarse_accuracy.mean()
        coarse_intersection = (coarse_pred * coarse_labels).sum(dim=1)
        coarse_union = (coarse_pred + coarse_labels).gt(0).sum(dim=1)
        coarse_iou = coarse_intersection * 1.0 / coarse_union + 1e-5
        coarse_iou = coarse_iou.mean()

        sharp_outputs = out["sharp_logits"]
        sharp_labels = out["sharp_target"]
        sharp_labels = (sharp_labels >= threshold).float()
        sharp_pred = torch.zeros_like(sharp_outputs)
        sharp_pred[sharp_outputs>=threshold] = 1
        sharp_accuracy = (sharp_pred==sharp_labels).float().sum(dim=1) / sharp_labels.shape[1]
        sharp_accuracy = sharp_accuracy.mean()
        sharp_intersection = (sharp_pred * sharp_labels).sum(dim=1)
        sharp_union = (sharp_pred + sharp_labels).gt(0).sum(dim=1)
        sharp_iou = sharp_intersection * 1.0 / sharp_union + 1e-5
        sharp_iou = sharp_iou.mean()

        mean_value = out["mean_value"]
        variance_value = out["variance_value"]



        self.log(f"val/overall_accuracy", overall_accuracy,sync_dist=True, on_epoch=True)
        self.log(f"val/overall_iou", overall_iou,sync_dist=True, on_epoch=True)

        self.log(f"val/coarse_accuracy", coarse_accuracy,sync_dist=True, on_epoch=True)
        self.log(f"val/coarse_iou", coarse_iou,sync_dist=True, on_epoch=True)

        self.log(f"val/sharp_accuracy", sharp_accuracy,sync_dist=True, on_epoch=True)
        self.log(f"val/sharp_iou", sharp_iou,sync_dist=True, on_epoch=True)

        self.log(f"val/mean_value", mean_value,sync_dist=True, on_epoch=True)
        self.log(f"val/variance_value", variance_value,sync_dist=True, on_epoch=True)

       

        self.log(f"val_{batch['uid'][0]}/overall_accuracy", overall_accuracy, on_epoch=True)
        self.log(f"val_{batch['uid'][0]}/overall_iou",overall_iou, on_epoch=True)

        self.log(f"val_{batch['uid'][0]}/coarse_accuracy", coarse_accuracy, on_epoch=True)
        self.log(f"val_{batch['uid'][0]}/coarse_iou", coarse_iou, on_epoch=True)

        self.log(f"val_{batch['uid'][0]}/sharp_accuracy", sharp_accuracy, on_epoch=True)
        self.log(f"val_{batch['uid'][0]}/sharp_iou", sharp_iou, on_epoch=True)

        self.log(f"val_{batch['uid'][0]}/mean_value", mean_value, on_epoch=True)
        self.log(f"val_{batch['uid'][0]}/variance_value", variance_value, on_epoch=True)

        torch.cuda.empty_cache()
        return {"val/loss": out["loss_sharp_logits"]}
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.eval()
        device = batch['coarse_surface'].device
        out = self(batch,'val')
        try:
            mesh_v_f, has_surface = self.shape_model.extract_geometry_by_diffdmc(out["latents"],octree_depth=9)
            file_path = f"it{self.true_global_step}/{os.path.basename(batch['uid'][0])}".replace(".npz",".obj")
            if not os.path.exists(file_path):
                self.save_mesh(
                    file_path,
                    mesh_v_f[0][0], mesh_v_f[0][1]
            )
        except Exception as e:
            print(f"ERROR: in processing batch: {batch}. Error: {e}")
            return
    
        threshold = 0
        overall_outputs = out["overall_logits"]
        overall_labels = out["overall_target"]
        overall_labels = (overall_labels >= threshold).float()
        overall_pred = torch.zeros_like(overall_outputs)
        overall_pred[overall_outputs>=threshold] = 1
        overall_accuracy = (overall_pred==overall_labels).float().sum(dim=1) / overall_labels.shape[1]
        overall_accuracy = overall_accuracy.mean()
        overall_intersection = (overall_pred * overall_labels).sum(dim=1)
        overall_union = (overall_pred + overall_labels).gt(0).sum(dim=1)
        overall_iou = overall_intersection * 1.0 / overall_union + 1e-5
        overall_iou = overall_iou.mean()

        coarse_outputs = out["coarse_logits"]
        coarse_labels = out["coarse_target"]
        coarse_labels = (coarse_labels >= threshold).float()
        coarse_pred = torch.zeros_like(coarse_outputs)
        coarse_pred[coarse_outputs>=threshold] = 1
        coarse_accuracy = (coarse_pred==coarse_labels).float().sum(dim=1) / coarse_labels.shape[1]
        coarse_accuracy = coarse_accuracy.mean()
        coarse_intersection = (coarse_pred * coarse_labels).sum(dim=1)
        coarse_union = (coarse_pred + coarse_labels).gt(0).sum(dim=1)
        coarse_iou = coarse_intersection * 1.0 / coarse_union + 1e-5
        coarse_iou = coarse_iou.mean()

        sharp_outputs = out["sharp_logits"]
        sharp_labels = out["sharp_target"]
        sharp_labels = (sharp_labels >= threshold).float()
        sharp_pred = torch.zeros_like(sharp_outputs)
        sharp_pred[sharp_outputs>=threshold] = 1
        sharp_accuracy = (sharp_pred==sharp_labels).float().sum(dim=1) / sharp_labels.shape[1]
        sharp_accuracy = sharp_accuracy.mean()
        sharp_intersection = (sharp_pred * sharp_labels).sum(dim=1)
        sharp_union = (sharp_pred + sharp_labels).gt(0).sum(dim=1)
        sharp_iou = sharp_intersection * 1.0 / sharp_union + 1e-5
        sharp_iou = sharp_iou.mean()

        mean_value = out["mean_value"]
        variance_value = out["variance_value"]

        self.log(f"val/overall_accuracy", overall_accuracy,sync_dist=True, on_epoch=True)
        self.log(f"val/overall_iou", overall_iou,sync_dist=True, on_epoch=True)

        self.log(f"val/coarse_accuracy", coarse_accuracy,sync_dist=True, on_epoch=True)
        self.log(f"val/coarse_iou", coarse_iou,sync_dist=True, on_epoch=True)

        self.log(f"val/sharp_accuracy", sharp_accuracy,sync_dist=True, on_epoch=True)
        self.log(f"val/sharp_iou", sharp_iou,sync_dist=True, on_epoch=True)

        self.log(f"val/mean_value", mean_value,sync_dist=True, on_epoch=True)
        self.log(f"val/variance_value", variance_value,sync_dist=True, on_epoch=True)

        self.log(f"val_{batch['uid'][0]}/overall_accuracy", overall_accuracy, on_epoch=True)
        self.log(f"val_{batch['uid'][0]}/overall_iou",overall_iou, on_epoch=True)

        self.log(f"val_{batch['uid'][0]}/coarse_accuracy", coarse_accuracy, on_epoch=True)
        self.log(f"val_{batch['uid'][0]}/coarse_iou", coarse_iou, on_epoch=True)

        self.log(f"val_{batch['uid'][0]}/sharp_accuracy", sharp_accuracy, on_epoch=True)
        self.log(f"val_{batch['uid'][0]}/sharp_iou", sharp_iou, on_epoch=True)

        self.log(f"val_{batch['uid'][0]}/mean_value", mean_value, on_epoch=True)
        self.log(f"val_{batch['uid'][0]}/variance_value", variance_value, on_epoch=True)

        torch.cuda.empty_cache()
        return {"val/loss": out["loss_sharp_logits"]}
        


    def on_validation_epoch_end(self):
        pass