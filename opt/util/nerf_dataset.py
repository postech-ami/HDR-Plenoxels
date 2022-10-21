# Standard NeRF Blender dataset loader
from .util import Rays, Intrin, select_or_shuffle_rays
from .dataset_base import DatasetBase
import torch
import torch.nn.functional as F
from typing import NamedTuple, Optional, Union
from os import path
import imageio
from tqdm import tqdm
import cv2
import json
import numpy as np


class NeRFDataset(DatasetBase):
    """
    NeRF dataset loader
    """

    focal: float
    c2w: torch.Tensor  # (n_images, 4, 4)
    gt: torch.Tensor  # (n_images, h, w, 3)
    h: int
    w: int
    n_images: int
    rays: Optional[Rays]
    split: str

    def __init__(
        self,
        root,
        split,
        epoch_size : Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        scene_scale: Optional[float] = None,
        factor: int = 1,
        scale : Optional[float] = None,
        permutation: bool = True,
        white_bkgd: bool = True,
        n_images = None,
        novel_view = False,
        rads_ratio : float = None,
        zrate_ratio : float = None,
        **kwargs
    ):
        super().__init__()
        assert path.isdir(root), f"'{root}' is not a directory"

        if scene_scale is None:
            scene_scale = 2/3
        if scale is None:
            scale = 1.0
        self.device = device
        self.permutation = permutation
        self.epoch_size = epoch_size
        all_c2w = []
        all_gt = []
        self.white_bkgd = white_bkgd

        split_name = split if split != "test_train" else "train" # train or test
        if split_name == "train":
            print("split_name is train")
            data_path = path.join(root, "test")
            data_json = path.join(root, "transforms_" + "test" + ".json")

            print("data_path", data_path, data_json)

            print("LOAD MASKED DATA", data_path)

            j = json.load(open(data_json, "r"))

            # OpenGL -> OpenCV
            cam_trans = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32))

            for frame in tqdm(j["frames"]):
                fpath = path.join(data_path, path.basename(frame["file_path"]) + ".png")
                c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
                c2w = c2w @ cam_trans  # To OpenCV

                im_gt = imageio.imread(fpath)
                if scale < 1.0:
                    full_size = list(im_gt.shape[:2])
                    rsz_h, rsz_w = [round(hw * scale) for hw in full_size]
                    im_gt = cv2.resize(im_gt, (rsz_w, rsz_h), interpolation=cv2.INTER_AREA)

                all_c2w.append(c2w)
                all_gt.append(torch.from_numpy(im_gt))
            self.num_masked = len(all_gt)
            print("self.num_masked = ", self.num_masked)
            # FIXME: assume same f
            # focal = float(
            #     0.5 * all_gt[0].shape[1] / np.tan(0.5 * j["camera_angle_x"])
            # )

        data_path = None
        data_json = None
        if novel_view:
            data_path = path.join(root, "train")
            data_json = path.join(root, "transforms_" + "train" + ".json")
        else:
            data_path = path.join(root, split_name)
            data_json = path.join(root, "transforms_" + split_name + ".json")

 
        print("data_path", data_path, data_json)
        print("LOAD DATA", data_path)

        j = json.load(open(data_json, "r"))

        # OpenGL -> OpenCV
        cam_trans = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32))

        for frame in tqdm(j["frames"]):
            fpath = path.join(data_path, path.basename(frame["file_path"]) + ".png")
            c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
            c2w = c2w @ cam_trans  # To OpenCV

            im_gt = imageio.imread(fpath)
            if scale < 1.0:
                full_size = list(im_gt.shape[:2])
                rsz_h, rsz_w = [round(hw * scale) for hw in full_size]
                im_gt = cv2.resize(im_gt, (rsz_w, rsz_h), interpolation=cv2.INTER_AREA)

            all_c2w.append(c2w)
            all_gt.append(torch.from_numpy(im_gt))
        focal = float(
            0.5 * all_gt[0].shape[1] / np.tan(0.5 * j["camera_angle_x"])
        )
        self.num_images = len(all_gt)       
        
        self.c2w = torch.stack(all_c2w)
        self.c2w[:, :3, :] *= scene_scale

        # novel view synthesis
        if novel_view:
            bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
            poses = self.c2w[:, :3, :]
            # poses = torch.tensor(poses)
            # poses = recenter_poses(poses.numpy())
            c2w_new = poses_avg(poses)
            up = normalize(poses[:, :3, 1].sum(0))
            # Get radii for spiral path
            tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
            # rads = np.percentile(np.abs(tt), 90, 0)
            rads = np.percentile(np.abs(tt), 90, 0)
            c2w_path = c2w_new
            N_views = 120
            N_rots = 2

            render_poses = render_path_spiral(
                c2w_path, up, rads, focal, zrate=0.5, rots=N_rots, N=N_views, rads_ratio=rads_ratio, zrate_ratio=zrate_ratio
            )

            print("\nrender poses")
            print(render_poses[:1])

            render_poses = np.array(render_poses).astype(np.float32)
            render_poses = buildNerfPoses(render_poses)

            render_c2w = []
            for idx in tqdm(range(len(render_poses))):
                R = render_poses[idx]["R"].astype(np.float64)
                t = render_poses[idx]["center"].astype(np.float64)
                c2w = np.concatenate([R, t], axis=1)
                cv2 = np.concatenate([c2w, bottom], axis=0)
                render_c2w.append(torch.from_numpy(c2w.astype(np.float32)))
            self.render_c2w = torch.stack(render_c2w)


        self.gt = torch.stack(all_gt).float() / 255.0
        if self.gt.size(-1) == 4:
            if white_bkgd:
                # Apply alpha channel
                self.alpha = self.gt[...,3:]
                self.gt = self.gt[..., :3] * self.gt[..., 3:] + (1.0 - self.gt[..., 3:])
            else:
                self.gt = self.gt[..., :3]

        self.n_images, self.h_full, self.w_full, _ = self.gt.shape
        
        #TODO: setting evaluation mask
        size = torch.Tensor([self.h_full, self.w_full])
        self.eval_mask = torch.ones((self.h_full, self.w_full,1), dtype=torch.bool)
        
        mask_ratio = torch.Tensor([1/2, 1/2]) #[1/3, 1/3] [1/2,1/2] [1, 1/2]
        small = torch.round(mask_ratio * size).int()
        big = torch.round(2*mask_ratio * size).int()
        small[0] = 0

        self.eval_mask[small[0]:big[0], small[1]:big[1],:] = False
        self.eval_mask_height = big[0]-small[0]
        self.eval_mask_width = big[1]-small[1]
               
        # Choose a subset of training images
        if n_images is not None: #False
            if n_images > self.n_images:
                print(f'using {self.n_images} available training views instead of the requested {n_images}.')
                n_images = self.n_images
            self.n_images = n_images
            self.gt = self.gt[0:n_images,...]
            self.c2w = self.c2w[0:n_images,...]

        self.intrins_full : Intrin = Intrin(focal, focal,
                                            self.w_full * 0.5,
                                            self.h_full * 0.5)

        self.split = split
        self.scene_scale = scene_scale
        
        if self.split == "train":
            index_masked = np.arange(self.num_masked)
            index_full = np.arange(self.num_masked, self.num_images)
            print("index_masked, index_full", index_masked, index_full)
            
            print(" Generating rays, scaling factor", factor)
            # Generate rays
            self.factor = factor
            self.h = self.h_full // factor
            self.w = self.w_full // factor
            true_factor = self.h_full / self.h
            self.intrins = self.intrins_full.scale(1.0 / true_factor)
            yy, xx = torch.meshgrid(
                torch.arange(self.h, dtype=torch.float32) + 0.5,
                torch.arange(self.w, dtype=torch.float32) + 0.5,
            )
            coords = torch.stack((xx-0.5, yy-0.5), dim=-1)
            coords = coords.unsqueeze(0).repeat(index_full.shape[0], 1, 1, 1)
            coords = coords.reshape(index_full.shape[0], -1, 2)

            xx = (xx - self.intrins.cx) / self.intrins.fx
            yy = (yy - self.intrins.cy) / self.intrins.fy
            zz = torch.ones_like(xx)
            dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
            dirs /= torch.norm(dirs, dim=-1, keepdim=True)
            dirs = dirs.reshape(1, -1, 3, 1)
            del xx, yy, zz
            dirs = (self.c2w[index_full, None, :3, :3] @ dirs)[..., 0]

            yy_masked, xx_masked = torch.meshgrid(
                torch.arange(self.h, dtype=torch.float32) + 0.5,
                torch.arange(self.w, dtype=torch.float32) + 0.5,
            )
            coords_masked = torch.stack((xx_masked-0.5, yy_masked-0.5), dim=-1)
            coords_masked = coords_masked.unsqueeze(0).repeat(index_masked.shape[0], 1, 1, 1)
            coords_masked = torch.masked_select(coords_masked, self.eval_mask).view(index_masked.shape[0], -1,2)
            
            xx_masked = (xx_masked - self.intrins.cx) / self.intrins.fx
            yy_masked = (yy_masked - self.intrins.cy) / self.intrins.fy
            zz_masked = torch.ones_like(xx_masked)
            dirs_masked = torch.stack((xx_masked, yy_masked, zz_masked), dim=-1)  # OpenCV convention
            dirs_masked /= torch.norm(dirs_masked, dim=-1, keepdim=True)
            dirs_masked = torch.masked_select(dirs_masked, self.eval_mask).view(-1,3)
            dirs_masked = dirs_masked.unsqueeze(0).unsqueeze(-1)

            del xx_masked, yy_masked, zz_masked
            dirs_masked = (self.c2w[index_masked, None, :3, :3] @ dirs_masked)[..., 0]


            if factor != 1:
                gt_masked = torch.masked_select(self.gt[index_masked], self.eval_mask).view(index_masked.shape[0], -1, 3)
                gt = self.gt[index_full, :,:,:]
                
                gt_masked = F.interpolate(
                    gt_masked.permute([0, 3, 1, 2]), size=(self.h, self.w), mode="area"
                ).permute([0, 2, 3, 1])
                gt = F.interpolate(
                    self.gt.permute([0, 3, 1, 2]), size=(self.h, self.w), mode="area"
                ).permute([0, 2, 3, 1])
                
                gt_masked = gt_masked.reshape(index_masked, -1, 3)
                gt = gt.reshape(self.n_images, -1, 3)
            else:
                gt_masked = torch.masked_select(self.gt[index_masked], self.eval_mask).view(index_masked.shape[0], -1, 3)
                gt_masked = gt_masked.reshape(index_masked.shape[0], -1, 3)
                gt = self.gt[index_full, :,:,:]
                gt = gt.reshape(index_full.shape[0], -1, 3)

            origins = self.c2w[index_full, None, :3, 3].expand(-1, self.h * self.w, -1).contiguous()
            origins_masked = self.c2w[index_masked, None, :3, 3].expand(-1, torch.sum(1*self.eval_mask), -1).contiguous()

            index_full = torch.from_numpy(index_full).reshape(-1,1).repeat(1, self.h*self.w).view(-1)
            index_masked = torch.from_numpy(index_masked).reshape(-1,1).repeat(1, torch.sum(1*self.eval_mask)).view(-1)
            
            if self.split == "train":
                origins = origins.view(-1, 3)
                origins_masked = origins_masked.view(-1,3)
                origins_merged = torch.cat((origins_masked, origins), 0)

                dirs = dirs.view(-1, 3)
                dirs_masked = dirs_masked.view(-1, 3)
                dirs_merged = torch.cat((dirs_masked, dirs), 0)
                
                gt = gt.reshape(-1, 3)
                gt_masked = gt_masked.reshape(-1, 3)
                gt_merged = torch.cat((gt_masked, gt), 0)
                
                coords = coords.view(-1, 2)
                coords_masked = coords_masked.view(-1, 2)
                coords_merged = torch.cat((coords_masked, coords), 0)
                
                index_merged = torch.cat((index_masked, index_full), 0)

            if white_bkgd:
                alpha_masked = self.alpha[np.arange(self.num_masked),:,:,:]
                alpha_masked = torch.masked_select(alpha_masked, self.eval_mask.unsqueeze(0).repeat(self.num_masked,1,1,1)).unsqueeze(-1)
                alpha_full = self.alpha[np.arange(self.num_masked, self.num_images),:,:,:]
                alpha_full = alpha_full.reshape(np.arange(self.num_masked, self.num_images).shape[0],-1,1).view(-1,1)
                
                alpha_merged = torch.cat((alpha_masked, alpha_full),0)

            self.rays_init = Rays(origins=origins_merged, dirs=dirs_merged, gt=gt_merged, coords=coords_merged, index=index_merged, alpha=alpha_merged)
            self.rays = self.rays_init
            
        else:
            # Rays are not needed for testing
            self.h, self.w = self.h_full, self.w_full
            self.intrins : Intrin = self.intrins_full
            index = np.arange(self.num_images)
            print("@" * 20)
            print("nerf dataset modify num_images:", self.num_images)
            print("nerf dataset modify index:", index)
            print("@" * 20)
            self.cam_index = torch.from_numpy(index)
            print(self.cam_index)

        self.should_use_background = False  # Give warning

# For novel view rendering

def poses_avg(poses):
    # poses [images, 3, 4] not [images, 3, 5]
    # hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center)], 1)

    return c2w

def render_path_spiral(c2w, up, rads, focal, zrate, rots, N, rads_ratio=0.08, zrate_ratio=0.8):
    render_poses = []

    print("rads ratio:", rads_ratio)
    print("zrate_Ratio:", zrate_ratio)
    # small rads -> small circle move
    
    rads = rads * rads_ratio
    rads = np.array(list(rads) + [1.0])
    # hwf = c2w[:,4:5]
    # focal = focal * 0.5
    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            # np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            np.array([np.cos(theta), -np.sin(theta), -np.sin(np.pi * zrate * zrate_ratio), 1.0])
            * rads,  # 0.95 -> make z smaller
        )

        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        # render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def nerf_pose_to_ours(cam):
    R = cam[:3, :3]
    center = cam[:3, 3].reshape([3, 1])
    center[1:] *= 1
    R[1:, 0] *= 1
    R[0, 1:] *= 1

    r = np.transpose(R)
    t = -r @ center
    return R, center, r, t

def buildNerfPoses(poses, images_path=None):
    output = {}
    for poses_id in range(poses.shape[0]):
        R, center, r, t = nerf_pose_to_ours(poses[poses_id].astype(np.float32))
        output[poses_id] = {"camera_id": 0, "r": r, "t": t, "R": R, "center": center}
        if images_path is not None:
            output[poses_id]["path"] = images_path[poses_id]

    return output

def recenter_poses(poses):
    # poses [images, 3, 4]
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)

    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses