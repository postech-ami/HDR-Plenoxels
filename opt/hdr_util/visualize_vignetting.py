import numpy as np
import os
import torch
import cv2

#FIXME:
H = 225
W = 337

#FIXME:
base_dir = "/local_data/kjs/hdr_plenoxel"
exp_name = "ribon_mix_change_optimizer"

directory = os.path.join(base_dir, exp_name)
txt_file = os.path.join(directory, "vig.txt")

vignetting = np.loadtxt(txt_file)

center = vignetting[:,:2]
coeff = vignetting[:,2:]

center = torch.tensor(center)
coeff = torch.tensor(coeff)

yy, xx = torch.meshgrid(
    torch.arange(H, dtype=torch.float32) + 0.5,
    torch.arange(W, dtype=torch.float32) + 0.5,
)
coords = torch.stack((xx-0.5, yy-0.5), dim=-1)
coords = coords.unsqueeze(0).repeat(vignetting.shape[0], 1, 1, 1)
coords = coords.reshape(vignetting.shape[0], -1, 2)

center[:, 0] = center[:, 0] / H
center[:, 1] = center[:, 1] / W
coords = coords.float()

coords[:, :, 0] = coords[:, :, 0] / H
coords[:, :, 1] = coords[:, :, 1] / W

r2 = torch.sum((coords - center.unsqueeze(1))**2, 2)
Iv = (1+coeff[:,0:1]*r2+coeff[:,1:2]*r2*r2)
Iv = Iv.reshape(-1, H, W, 1)

images = Iv.cpu().numpy()

#FIXME:
for i in range(images.shape[0]):
    name = "vis_vig" + str(i) + ".png"
    cv2.imwrite(name, images[i]*255)