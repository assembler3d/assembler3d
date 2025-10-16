import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core/model/encoder"))

import argparse
import yaml
import numpy as np
import logging
import cv2
import glob
import fpsample
from collections import OrderedDict
from matplotlib import pyplot as plt
import rembg
from huggingface_hub import hf_hub_download

logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] [%(module)s] [%(levelname)s] | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

import trimesh
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from core.model.diffusion import create_diffusion
from core.model.denoiser.dit import DiT_models
from core.utils.train_utils import load_model
from core.data.data_utils import save_color_ply
from core.data.mesh_utils import transform_mesh, visualize_patches, hybrid_robust_transform, canonicalize_mesh, scale_part_mesh
from core.model.condition_encoder.dinov2_model import Dinov2Model
from craftsman.models.autoencoders.michelangelo_autoencoder import MichelangeloAutoencoder
from craftsman.utils.config import load_config


def load_data(cfg, input_dir):
    # load part meshes, and scale them into approximately [-1,1] cube
    part_meshes, part_names = [], []
    part_names = sorted(glob.glob(input_dir + "/*.glb"))
    bounds, areas = [], []
    for part_name in part_names:
        part_mesh = trimesh.load(part_name, force='mesh')
        # canonicalize the part mesh
        part_mesh = canonicalize_mesh(part_mesh)
        part_meshes.append(part_mesh)
        bounds.append((part_mesh.bounds[1] - part_mesh.bounds[0]).max() / 2)
        areas.append(part_mesh.area)
    
    bounds = np.array(bounds)
    areas = np.array(areas)
    idx = np.argmax(areas)
    ratio = areas.max() / areas.sum()
    scale = bounds[idx] / ratio
    # scale the part meshes
    part_meshes = [scale_part_mesh(part_mesh, scale, center=np.array([0,0,0])) for part_mesh in part_meshes]

    # sample points for each part
    object_sample_num = int(cfg['token_length'])
    part_sample_num = 4096
    # get each part's vertice number to determine the size
    part_verts_cnt = np.array([part_mesh.area for part_mesh in part_meshes])
    part_verts_ratio = part_verts_cnt / float(np.sum(part_verts_cnt))
    
    max_num = object_sample_num // float(len(part_meshes))
    minimum_num = np.minimum(10, max_num)
    object_part_sample_num = part_verts_ratio * (object_sample_num - minimum_num * len(part_meshes))
    # make it int and ensure the total number of points is object_sample_num
    object_part_sample_num = np.array([int(minimum_num + np.floor(part_sample_num)) for part_sample_num in object_part_sample_num])
    object_part_sample_num[0] = int(object_sample_num - np.sum(object_part_sample_num[1:]))

    all_parts_dense, all_parts_anchor, all_sparse_inds = [], [], []
    part_ids = []
    for ind, mesh in enumerate(part_meshes):
        # sample the points with surface normals
        coarse_surface_points, faces = mesh.sample(20000, return_index=True)
        normals = mesh.face_normals[faces]
        coarse_surface = np.concatenate([coarse_surface_points, normals], axis=1)

        kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(coarse_surface_points, part_sample_num, h=5)
        part_pc = coarse_surface[kdline_fps_samples_idx].reshape(-1,6)
        all_parts_dense.append(part_pc)

        # sub-sample the points to get the anchor points
        part_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(part_pc, object_part_sample_num[ind], h=5)
        part_anchor_pc = part_pc[part_fps_samples_idx].reshape(-1,6)
        all_parts_anchor.append(part_anchor_pc)
        all_sparse_inds.append(part_fps_samples_idx)
        part_ids.append(np.ones(part_anchor_pc.shape[0]) * ind)
    
    # prepare point-wise mask (NxN), used for attention
    acc_sum = np.cumsum(np.array(object_part_sample_num))
    acc_sum = np.insert(acc_sum, 0, 0)
    N_part = len(part_names)
    mask = (np.zeros((object_sample_num, object_sample_num)) > 0)
    for i in range(N_part):
        mask[acc_sum[i]:acc_sum[i+1], acc_sum[i]:acc_sum[i+1]] = True
    part_ids = np.concatenate(part_ids, 0)
        
    # load image
    img_name = glob.glob(input_dir + "/*.png")[0]
    image = cv2.imread(img_name, -1)
    if image.shape[-1] == 4:
        img_mask = (image[...,-1:] > 0)
    else:
        image = rembg.remove(np.array(image))
        img_mask = (image[...,-1:] > 0)

    image_rgb = cv2.cvtColor(image[...,:3], cv2.COLOR_BGR2RGB)
    # resize if not 256
    if image.shape[0] != 256 or image.shape[1] != 256:
        image_rgb = cv2.resize(image_rgb, (256, 256), interpolation=cv2.INTER_NEAREST)
        if image.shape[-1] == 4:
            img_mask = cv2.resize((255 * img_mask[:,:,0]).astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
            img_mask = (img_mask > 128)[:,:,None]
    # convert the transparent pixels to white background
    if image.shape[-1] == 4:
        image_whiteback = image_rgb * img_mask + 255 * (1 - img_mask)
        image_whiteback = np.array(image_whiteback).astype(np.uint8)
    else:
        image_whiteback = np.array(image_rgb).astype(np.uint8)

    part_names = [os.path.basename(x) + '.glb' for x in part_names]

    return all_parts_dense, all_parts_anchor, all_sparse_inds, part_meshes, part_ids, part_names, mask, acc_sum, image_whiteback


def main(args, cfg):
    # Setup PyTorch:
    torch.manual_seed(cfg["seed"])
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model:
    model = DiT_models[cfg["model"]](condition_channels=cfg["condition_channels"]).to(device)

    # download the checkpoint if not found:
    if not os.path.exists(cfg["ckpt_path"]):
        model_dir, model_name = os.path.dirname(cfg["ckpt_path"]), os.path.basename(cfg["ckpt_path"])
        os.makedirs(model_dir, exist_ok=True)
        checkpoint_path = hf_hub_download(repo_id="TencentARC/Assembler", 
                            local_dir=model_dir, filename=model_name)
        print("Downloading checkpoint {} from Hugging Face Hub...".format(model_name))
    print("Loading model from {}".format(cfg["ckpt_path"]))

    # load a custom DiT checkpoint from train.py:
    state_dict = load_model(cfg["ckpt_path"])
    # remove the 'module' prefix if it exists
    weight = OrderedDict()
    for key, value in state_dict.items():
        key = key[7:]  # module.xxx.xxx -> xxx.xxx
        weight[key] = value
    model.load_state_dict(weight, strict=True)
    model.eval()  # important!
    diffusion = create_diffusion(str(cfg["num_sampling_steps"]))

    # feature encoding model
    dino_model = Dinov2Model()
    # point cloud encoding model
    dora_cfg = load_config(cfg["dora_config_path"])
    dora_path = dora_cfg.system.shape_model.pretrained_model_name_or_path
    dora_dir, dora_model_name = os.path.dirname(dora_path), os.path.basename(dora_path)
    if not os.path.exists(dora_path):
        hf_hub_download(repo_id="Seed3D/Dora-VAE-1.1",local_dir=dora_dir, filename=dora_model_name)
    dora_model = MichelangeloAutoencoder(dora_cfg.system.shape_model).to(device).eval()
    
    # cfg scale
    cfg_scale = float(cfg["cfg_scale"])

    # Setup data:
    all_parts_dense, all_parts_anchor, all_sparse_inds, part_meshes, part_ids, part_names, mask, acc_sum, image = load_data(cfg, args.input_dir)
    # encode point cloud feature
    point_conditions = []
    for i in range(len(all_parts_dense)):
        points = all_parts_dense[i]
        query_ind = all_sparse_inds[i]
        points = torch.from_numpy(points).to(device).float().unsqueeze(0)
        # forward
        with torch.no_grad():
            shape_latents, kl_embed, posterior = dora_model.encode_with_query(points, query_ind, sample_posterior=True)
            latents = dora_model.decode(kl_embed)
        # concat latents and points
        latents = latents[0].cpu().numpy()
        sparse_points = points[0].cpu().numpy()[query_ind, :3]
        point_condition = np.concatenate([latents, sparse_points], axis=-1)
        point_conditions.append(point_condition)
    point_conditions = np.concatenate(point_conditions, axis=0)
    
    point_conditions = torch.from_numpy(point_conditions).unsqueeze(0).to(device).float()
    mask = torch.from_numpy(mask).unsqueeze(0).to(device).bool()
    part_ids = torch.from_numpy(part_ids).unsqueeze(0).to(device).float()
    token_length = point_conditions.shape[1]
    
    img_feats = dino_model.encode_batch_imgs([image], global_feat=False)
    for j in range(1):
        cond_feats = img_feats[j].unsqueeze(0).to(device).float()
        # Create sampling noise:
        z = torch.randn(1, 3, token_length, device=device)
        z = torch.cat([z, z], dim=0)

        # classifier-free guidance:
        uncond_feats = torch.zeros_like(cond_feats).to(device).float()
        cond_feats = torch.cat([cond_feats, uncond_feats], dim=0)
        point_conditions = torch.cat([point_conditions, point_conditions], dim=0)
        mask = torch.cat([mask, mask], dim=0)
        part_ids = torch.cat([part_ids, part_ids], dim=0)

        model_kwargs = dict(y=point_conditions, cond=cond_feats, part_ids=part_ids, mask=mask, cfg_scale=cfg_scale)

        with torch.no_grad():
            # Sample target params:
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
            samples, _ = samples.chunk(2, dim=0)
            samples = samples[0].permute(1,0).cpu().numpy()

        # save the generated anchor points
        save_dir = args.output_dir
        os.makedirs(save_dir, exist_ok=True)
        
        save_name = os.path.join(save_dir, 'demo')

        # compute the transformation and save the mesh
        all_meshes = []
        all_transforms = []
        for idx in range(len(all_parts_anchor)):
            raw_points = all_parts_anchor[idx][:,:3].copy()
            transform = hybrid_robust_transform(raw_points, samples[acc_sum[idx]:acc_sum[idx+1]])
            transformed_mesh = transform_mesh(part_meshes[idx], transform)
            all_meshes.append(transformed_mesh)
            all_transforms.append(transform)
        # visualize the mesh
        all_visual, part_colors = visualize_patches(all_meshes)
        all_visual.export(save_name + ".glb")

        # color each part by its index
        colors = []
        for idx in range(len(all_parts_anchor)):
            N = all_parts_anchor[idx].shape[0]
            part_color = np.tile(np.array(part_colors[idx, :3])[None], (N, 1))
            colors.append(part_color)
        colors = np.concatenate(colors, axis=0)
        save_color_ply(save_name + ".ply", samples, colors)
    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    # create visualize dir
    os.makedirs(args.output_dir, exist_ok=True)
    main(args, cfg)
