import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

class Dinov2Model(object):
    def __init__(self, device='cuda'):
        self.device = device
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        # To load from local
        # self.model = torch.hub.load('/path/to/your/local/dinov/repo', 'dinov2_vitb14', source='local', pretrained=False)
        # self.model.load_state_dict(torch.load('/path/to/your/local/weights'))
        self.model.to(device)
        self.model.eval()
        self.image_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
        )
        self.grid_size = 224 // self.model.patch_size
    
    def encode_img(self, img_path, background=0):
        image = Image.open(img_path).convert('RGB')
        if background == 0:
            mask = (np.array(image).sum(-1) <= 3)
            img_arr = np.array(image)
            img_arr[mask] = [255, 255, 255]
            image = Image.fromarray(img_arr)
        image = self.image_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_feat = self.model(image).float()
        return image_feat
    
    def encode_batch_imgs(self, batch_imgs, global_feat=True):
        with torch.no_grad():
            images = [self.image_transform(Image.fromarray(img)).to(self.device) for img in batch_imgs]
            images = torch.stack(images, 0)
            if global_feat:
                image_feat = self.model(images).float()
            else:
                image_feat = self.model.get_intermediate_layers(images)[0].float()
        return image_feat