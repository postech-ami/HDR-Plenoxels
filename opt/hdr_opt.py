# Copyright 2021 Alex Yu

# First, install svox2
# Then, python opt.py <path_to>/nerf_synthetic/<scene> -t ckpt/<some_name>
# or use launching script:   sh launch.sh <EXP_NAME> <GPU> <DATA_DIR>
import torch
import torch.cuda
import torch.optim
import torch.nn.functional as F
import svox2
import json
import imageio
import os
from os import path
import shutil
import gc
import numpy as np
import math
import argparse
import cv2
from util.dataset import datasets
from util.util import Timing, get_expon_lr_func, generate_dirs_equirect, viridis_cmap, saturation_mask, get_linear_lr_func
from util import config_util
from util import cam_param

from warnings import warn
from datetime import datetime

from util.util import compute_ssim
import lpips

from tqdm import tqdm
from typing import NamedTuple, Optional, Union

def main():
    # fix Ampere GPU problem - round off float
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_vgg = lpips.LPIPS(net="vgg").eval().to(device)

    parser = argparse.ArgumentParser()
    config_util.define_common_args(parser)


    group = parser.add_argument_group("general")
    group.add_argument('--train_dir', '-t', type=str, default='ckpt',
                        help='checkpoint and logging directory')

    group.add_argument('--reso',
                            type=str,
                            default=
                            "[[256, 256, 256], [512, 512, 512]]",
                        help='List of grid resolution (will be evaled as json);'
                                'resamples to the next one every upsamp_every iters, then ' +
                                'stays at the last one; ' +
                                'should be a list where each item is a list of 3 ints or an int')
    group.add_argument('--upsamp_every', type=int, default=
                        3 * 12800,
                        help='upsample the grid every x iters')
    group.add_argument('--init_iters', type=int, default=
                        0,
                        help='do not upsample for first x iters')
    group.add_argument('--upsample_density_add', type=float, default=
                        0.0,
                        help='add the remaining density by this amount when upsampling')

    group.add_argument('--basis_type',
                        choices=['sh', '3d_texture', 'mlp'],
                        default='sh',
                        help='Basis function type')

    group.add_argument('--basis_reso', type=int, default=32,
                    help='basis grid resolution (only for learned texture)')
    group.add_argument('--sh_dim', type=int, default=9, help='SH/learned basis dimensions (at most 10)')

    group.add_argument('--mlp_posenc_size', type=int, default=4, help='Positional encoding size if using MLP basis; 0 to disable')
    group.add_argument('--mlp_width', type=int, default=32, help='MLP width if using MLP basis')

    group.add_argument('--background_nlayers', type=int, default=0,#32,
                    help='Number of background layers (0=disable BG model)')
    group.add_argument('--background_reso', type=int, default=512, help='Background resolution')

    parser.add_argument('--use_tone_mapping', type=bool, default=False)
    parser.add_argument('--use_initialize', type=bool, default=False)
    parser.add_argument('--use_sat_mask', type=bool, default=False)
    parser.add_argument('--use_sh_mask', type=bool, default=False)
    parser.add_argument('--crop_margin', type=int, default=0)

    group = parser.add_argument_group("optimization")
    group.add_argument('--n_iters', type=int, default=10 * 12800, help='total number of iters to optimize for')
    group.add_argument('--batch_size', type=int, default=
                        #    100,
                        2500,
                        #  100000,
                        #  2000,
                    help='batch size')


    # TODO: make the lr higher near the end
    group.add_argument('--sigma_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Density optimizer")
    group.add_argument('--lr_sigma', type=float, default=3e1, help='SGD/rmsprop lr for sigma')
    group.add_argument('--lr_sigma_final', type=float, default=5e-2)
    group.add_argument('--lr_sigma_decay_steps', type=int, default=250000)
    group.add_argument('--lr_sigma_delay_steps', type=int, default=15000,
                    help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_sigma_delay_mult', type=float, default=1e-2)#1e-4)#1e-4)


    group.add_argument('--sh_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="SH optimizer")
    group.add_argument('--lr_sh', type=float, default=
                        1e-2,
                    help='SGD/rmsprop lr for SH')
    group.add_argument('--lr_sh_final', type=float,
                        default=
                        5e-6
                        )
    group.add_argument('--lr_sh_decay_steps', type=int, default=250000)
    group.add_argument('--lr_sh_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_sh_delay_mult', type=float, default=1e-2)

    group.add_argument('--cam_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Our camera tone-mapping module optimizer")
    group.add_argument('--lr_cam', type=float, default=
                        1e-2,
                    help='SGD/rmsprop lr for SH')
    group.add_argument('--lr_cam_final', type=float,
                        default=
                        5e-6
                        )
    group.add_argument('--lr_cam_decay_steps', type=int, default=250000)
    group.add_argument('--lr_cam_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_cam_delay_mult', type=float, default=1e-2)


    group.add_argument('--lr_fg_begin_step', type=int, default=0, help="Foreground begins training at given step number")

    # BG LRs
    group.add_argument('--bg_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Background optimizer")
    group.add_argument('--lr_sigma_bg', type=float, default=3e0,
                        help='SGD/rmsprop lr for background')
    group.add_argument('--lr_sigma_bg_final', type=float, default=3e-3,
                        help='SGD/rmsprop lr for background')
    group.add_argument('--lr_sigma_bg_decay_steps', type=int, default=250000)
    group.add_argument('--lr_sigma_bg_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_sigma_bg_delay_mult', type=float, default=1e-2)

    group.add_argument('--lr_color_bg', type=float, default=1e-1,
                        help='SGD/rmsprop lr for background')
    group.add_argument('--lr_color_bg_final', type=float, default=5e-6,#1e-4,
                        help='SGD/rmsprop lr for background')
    group.add_argument('--lr_color_bg_decay_steps', type=int, default=250000)
    group.add_argument('--lr_color_bg_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_color_bg_delay_mult', type=float, default=1e-2)
    # END BG LRs

    group.add_argument('--basis_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Learned basis optimizer")
    group.add_argument('--lr_basis', type=float, default=#2e6,
                        1e-6,
                    help='SGD/rmsprop lr for SH')
    group.add_argument('--lr_basis_final', type=float,
                        default=
                        1e-6
                        )
    group.add_argument('--lr_basis_decay_steps', type=int, default=250000)
    group.add_argument('--lr_basis_delay_steps', type=int, default=0,#15000,
                    help="Reverse cosine steps (0 means disable)")
    group.add_argument('--lr_basis_begin_step', type=int, default=0) # 4 * 12800
    group.add_argument('--lr_basis_delay_mult', type=float, default=1e-2)

    group.add_argument('--rms_beta', type=float, default=0.95, help="RMSProp exponential averaging factor")

    group.add_argument('--print_every', type=int, default=20, help='print every')
    group.add_argument('--save_every', type=int, default=5,
                    help='save every x epochs')
    group.add_argument('--eval_every', type=int, default=1,
                    help='evaluate every x epochs')

    group.add_argument('--init_sigma', type=float,
                    default=0.1,
                    help='initialization sigma')
    group.add_argument('--init_sigma_bg', type=float,
                    default=0.1,
                    help='initialization sigma (for BG)')

    # Extra logging
    group.add_argument('--log_mse_image', action='store_true', default=False)
    group.add_argument('--log_depth_map', action='store_true', default=False)
    group.add_argument('--log_depth_map_use_thresh', type=float, default=None,
            help="If specified, uses the Dex-neRF version of depth with given thresh; else returns expected term")


    group = parser.add_argument_group("misc experiments")
    group.add_argument('--thresh_type',
                        choices=["weight", "sigma"],
                        default="weight",
                    help='Upsample threshold type')
    group.add_argument('--weight_thresh', type=float,
                        default=0.0005 * 512,
                        #  default=0.025 * 512,
                    help='Upsample weight threshold; will be divided by resulting z-resolution')
    group.add_argument('--density_thresh', type=float,
                        default=5.0,
                    help='Upsample sigma threshold')
    group.add_argument('--background_density_thresh', type=float,
                        default=1.0+1e-9,
                    help='Background sigma threshold for sparsification')
    group.add_argument('--max_grid_elements', type=int,
                        default=44_000_000,
                    help='Max items to store after upsampling '
                            '(the number here is given for 22GB memory)')

    group.add_argument('--tune_mode', action='store_true', default=False,
                    help='hypertuning mode (do not save, for speed)')
    group.add_argument('--tune_nosave', action='store_true', default=False,
                    help='do not save any checkpoint even at the end')


    group = parser.add_argument_group("losses")
    # Foreground TV
    group.add_argument('--lambda_tv', type=float, default=1e-5)
    group.add_argument('--tv_sparsity', type=float, default=0.01)
    group.add_argument('--tv_logalpha', action='store_true', default=False,
                    help='Use log(1-exp(-delta * sigma)) as in neural volumes')

    group.add_argument('--lambda_tv_sh', type=float, default=1e-3)
    group.add_argument('--tv_sh_sparsity', type=float, default=0.01)

    group.add_argument('--lambda_tv_lumisphere', type=float, default=0.0)#1e-2)#1e-3)
    group.add_argument('--tv_lumisphere_sparsity', type=float, default=0.01)
    group.add_argument('--tv_lumisphere_dir_factor', type=float, default=0.0)

    group.add_argument('--tv_decay', type=float, default=1.0)

    group.add_argument('--lambda_l2_sh', type=float, default=0.0)#1e-4)
    group.add_argument('--tv_early_only', type=int, default=1, help="Turn off TV regularization after the first split/prune")

    group.add_argument('--tv_contiguous', type=int, default=1,
                            help="Apply TV only on contiguous link chunks, which is faster")
    # End Foreground TV

    group.add_argument('--lambda_sparsity', type=float, default=
                        0.0,
                        help="Weight for sparsity loss as in SNeRG/PlenOctrees " +
                            "(but applied on the ray)")
    group.add_argument('--lambda_beta', type=float, default=
                        0.0,
                        help="Weight for beta distribution sparsity loss as in neural volumes")


    # Background TV
    group.add_argument('--lambda_tv_background_sigma', type=float, default=1e-2)
    group.add_argument('--lambda_tv_background_color', type=float, default=1e-2)

    group.add_argument('--tv_background_sparsity', type=float, default=0.01)
    # End Background TV

    # Basis TV
    group.add_argument('--lambda_tv_basis', type=float, default=0.0,
                    help='Learned basis total variation loss')
    # End Basis TV

    group.add_argument('--weight_decay_sigma', type=float, default=1.0)
    group.add_argument('--weight_decay_sh', type=float, default=1.0)

    group.add_argument('--lr_decay', action='store_true', default=True)

    group.add_argument('--n_train', type=int, default=None, help='Number of training images. Defaults to use all avaiable.')

    group.add_argument('--nosphereinit', action='store_true', default=False,
                        help='do not start with sphere bounds (please do not use for 360)')

    #TODO:
    group = parser.add_argument_group("wandb")
    group.add_argument("--wandb", type=bool, default=False)
    group.add_argument('--project', type=str, default="hdr_plenoxels",
                    help='Wandb project name')
    group.add_argument('--entity', type=str, default="",
                    help='Wandb entity')
    group.add_argument('--exp_name', type=str, default="room_mid")
    group.add_argument("--tag_name", type=str, default="room")

    group.add_argument("--rads_ratio",
                        type=float,
                        default=1)

    group.add_argument("--zrate_ratio",
                        type=float,
                        default=0.8)


    args = parser.parse_args()
    config_util.maybe_merge_config_file(args)

    # wandb.config.update(args)

    assert args.lr_sigma_final <= args.lr_sigma, "lr_sigma must be >= lr_sigma_final"
    assert args.lr_sh_final <= args.lr_sh, "lr_sh must be >= lr_sh_final"
    assert args.lr_basis_final <= args.lr_basis, "lr_basis must be >= lr_basis_final"

    #TODO: wandb logging
    if args.wandb:

        import wandb
        wandb.login()

        wandb.init(
            project = args.project,
            entity = args.entity,
            name = args.exp_name,
            tags=[args.tag_name],
            config=args,
            save_code=True
        )

    os.makedirs(args.train_dir, exist_ok=True)

    print("[args.reso]:", args.reso)
    reso_list = json.loads(args.reso)
    reso_id = 0

    with open(path.join(args.train_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        # Changed name to prevent errors
        shutil.copyfile(__file__, path.join(args.train_dir, 'opt_frozen.py'))

    torch.manual_seed(20200823)
    np.random.seed(20200823)

    factor = 1
    dset = datasets[args.dataset_type](
                args.data_dir,
                split="train",
                device=device,
                factor=factor,
                n_images=args.n_train,
                **config_util.build_data_options(args))

    if args.background_nlayers > 0 and not dset.should_use_background:
        warn('Using a background model for dataset type ' + str(type(dset)) + ' which typically does not use background')

    dset_test = datasets[args.dataset_type](
            args.data_dir, split="test", **config_util.build_data_options(args))

    global_start_time = datetime.now()

    grid = svox2.SparseGrid(reso=reso_list[reso_id],
                            center=dset.scene_center, # [0,0,0,0]
                            radius=dset.scene_radius, # [radx, rady, radz] where, 1 + 2 * self.sfm.offset / self.gt.size(2), offset = 250, gt.size(i) = 3*h*w
                            use_sphere_bound=dset.use_sphere_bound and not args.nosphereinit, # false
                            basis_dim=args.sh_dim, # 9
                            use_z_order=True,
                            device=device,
                            basis_reso=args.basis_reso, # 32
                            basis_type=svox2.__dict__['BASIS_TYPE_' + args.basis_type.upper()], # default = sh, ['sh', '3d_texture', 'mlp']
                            mlp_posenc_size=args.mlp_posenc_size, # 4
                            mlp_width=args.mlp_width, # 32
                            background_nlayers=args.background_nlayers, # 0
                            background_reso=args.background_reso)  # 512

    # DC -> gray; mind the SH scaling!
    grid.sh_data.data[:] = 0.0
    grid.density_data.data[:] = 0.0 if args.lr_fg_begin_step > 0 else args.init_sigma # args.lr_fg_begin_step ==0, args.init_sigma : 0.1

    if grid.use_background: # false 
        grid.background_data.data[..., -1] = args.init_sigma_bg
        #  grid.background_data.data[..., :-1] = 0.5 / svox2.utils.SH_C0

    #  grid.sh_data.data[:, 0] = 4.0
    #  osh = grid.density_data.data.shape
    #  den = grid.density_data.data.view(grid.links.shape)
    #  #  den[:] = 0.00
    #  #  den[:, :256, :] = 1e9
    #  #  den[:, :, 0] = 1e9
    #  grid.density_data.data = den.view(osh)

    optim_basis_mlp = None

    if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
        grid.reinit_learned_bases(init_type='sh')
        #  grid.reinit_learned_bases(init_type='fourier')
        #  grid.reinit_learned_bases(init_type='sg', upper_hemi=True)
        #  grid.basis_data.data.normal_(mean=0.28209479177387814, std=0.001)

    elif grid.basis_type == svox2.BASIS_TYPE_MLP:
        # MLP!
        optim_basis_mlp = torch.optim.Adam(
                        grid.basis_mlp.parameters(),
                        lr=args.lr_basis
                    )

    if args.use_tone_mapping == True:
        #TODO:
        N, H, W, C = dset.gt.shape
        Cam_param = cam_param.CamParam(N, H, W, device=device, gts=dset.gt, initialize=args.use_initialize, tone_mapping=args.tone_mapping)
        optim_cam = Cam_param.optimizer(l_rate = args.lr_sh)

        grid.requires_grad_(True)
        config_util.setup_render_opts(grid.opt, args)

    gstep_id_base = 0

    resample_cameras = [
            svox2.Camera(c2w.to(device=device),
                        dset.intrins.get('fx', i),
                        dset.intrins.get('fy', i),
                        dset.intrins.get('cx', i),
                        dset.intrins.get('cy', i),
                        width=dset.get_image_size(i)[1],
                        height=dset.get_image_size(i)[0],
                        ndc_coeffs=dset.ndc_coeffs) for i, c2w in enumerate(dset.c2w)
        ]
    ckpt_path = path.join(args.train_dir, 'ckpt.npz')
    ckpt_cam_path = path.join(args.train_dir, 'ckpt_cam.npz')

    best_ckpt_path = path.join(args.train_dir, 'best_ckpt.npz')
    best_ckpt_cam_path = path.join(args.train_dir, 'best_ckpt_cam.npz')

    if args.use_tone_mapping == True:
        Cam_param.save_txt(args.train_dir)

    lr_sigma_func = get_expon_lr_func(args.lr_sigma, args.lr_sigma_final, args.lr_sigma_delay_steps,
                                    args.lr_sigma_delay_mult, args.lr_sigma_decay_steps)
    lr_sh_func = get_expon_lr_func(args.lr_sh, args.lr_sh_final, args.lr_sh_delay_steps,
                                args.lr_sh_delay_mult, args.lr_sh_decay_steps)
    lr_basis_func = get_expon_lr_func(args.lr_basis, args.lr_basis_final, args.lr_basis_delay_steps,
                                args.lr_basis_delay_mult, args.lr_basis_decay_steps)
    lr_sigma_bg_func = get_expon_lr_func(args.lr_sigma_bg, args.lr_sigma_bg_final, args.lr_sigma_bg_delay_steps,
                                args.lr_sigma_bg_delay_mult, args.lr_sigma_bg_decay_steps)
    lr_color_bg_func = get_expon_lr_func(args.lr_color_bg, args.lr_color_bg_final, args.lr_color_bg_delay_steps,
                                args.lr_color_bg_delay_mult, args.lr_color_bg_decay_steps)
    lr_sigma_factor = 1.0
    lr_sh_factor = 1.0
    lr_basis_factor = 1.0

    if args.use_tone_mapping == True:
        lr_cam_func = get_expon_lr_func(args.lr_cam, args.lr_cam_final, args.lr_cam_delay_steps,
                                    args.lr_cam_delay_mult, args.lr_cam_decay_steps)
        lr_cam_factor = 1.0

    last_upsamp_step = args.init_iters

    if args.enable_random:
        warn("Randomness is enabled for training (normal for LLFF & scenes with background)")

    epoch_id = -1
    best_psnr = 0.0
    while True:
        dset.shuffle_rays()
        epoch_id += 1
        epoch_size = dset.rays.origins.size(0) # N*W*H
        batches_per_epoch = (epoch_size-1)//args.batch_size+1 # batch_size = 5000

        # Test
        def eval_step():
            # Put in a function to avoid memory leak
            print('Eval step')
            with torch.no_grad():
                stats_test = {'psnr' : 0.0, 'mse' : 0.0, 'ssim' : 0.0, 'lpips' : 0.0 }

                # Determine 
                # Standard set
                N_IMGS_TO_EVAL = min(20 if epoch_id > 0 else 5, dset_test.n_images) 
                N_IMGS_TO_SAVE = N_IMGS_TO_EVAL # if not args.tune_mode else 1 
                img_eval_interval = dset_test.n_images // N_IMGS_TO_EVAL 
                img_save_interval = (N_IMGS_TO_EVAL // N_IMGS_TO_SAVE) 
                img_ids = range(0, dset_test.n_images, img_eval_interval)

                # Special 'very hard' specular + fuzz set
                #  img_ids = [2, 5, 7, 9, 21,
                #             44, 45, 47, 49, 56,
                #             80, 88, 99, 115, 120,
                #             154]
                #  img_save_interval = 1

                n_images_gen = 0
                
                for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
                    c2w = dset_test.c2w[img_id].to(device=device)
                    cam = svox2.Camera(c2w,
                                    dset_test.intrins.get('fx', img_id),
                                    dset_test.intrins.get('fy', img_id),
                                    dset_test.intrins.get('cx', img_id),
                                    dset_test.intrins.get('cy', img_id),
                                    width=dset_test.get_image_size(img_id)[1],
                                    height=dset_test.get_image_size(img_id)[0],
                                    ndc_coeffs=dset_test.ndc_coeffs)
                    rgb_pred_test = grid.volume_render_image(cam, use_kernel=True)
                    rgb_gt_test = dset_test.gt[img_id].to(device=device)

                    #TODO:
                    if args.use_tone_mapping == True :
                        index = dset_test.cam_index[i]
                        ldr_pred_test = Cam_param.RAD2LDR_img(rgb_pred_test, index)
                    else :
                        ldr_pred_test = rgb_pred_test
                    if dset.white_bkgd:
                        alpha = dset_test.alpha[img_id].to(device)

                    if dset.white_bkgd:
                        ldr_pred_test = ldr_pred_test * alpha + (1-alpha)

                    mask = dset_test.eval_mask
                    mask = torch.logical_not(mask).to(device)
                    
                    height, width = dset_test.get_image_size(img_id)
                    
                    # save full prediction image - HDR
                    hdr_pred_test_full = rgb_pred_test.clone().detach()
                    hdr_pred_test_full = hdr_pred_test_full.view(height, width, 3)  # synthetic - (800, 800, 3)

                    # save full prediction image - LDR
                    ldr_pred_test_full = ldr_pred_test.clone().detach()
                    ldr_pred_test_full = ldr_pred_test_full.view(height, width, 3)  # synthetic - (800, 800, 3)

                    # save right-half prediction image (real pred)
                    ldr_pred_test = torch.masked_select(ldr_pred_test, mask).view(height, width // 2, 3)
                    rgb_gt_test = torch.masked_select(rgb_gt_test, mask).view(height, width // 2, 3)
                    
                    # Crop the boundary of rendering outputs
                    if args.crop_margin > 0:
                        crop_margin = args.crop_margin
                        rgb_gt_test = rgb_gt_test[crop_margin:-crop_margin, :-crop_margin, :]
                        ldr_pred_test = ldr_pred_test[crop_margin:-crop_margin, :-crop_margin, :]
                        
                    all_mses = ((rgb_gt_test - ldr_pred_test) ** 2).cpu()
                    # print("[CROP image size]:", rgb_gt_test_crop.shape)

                    if args.wandb:
                        wandb.log({
                            "test/image_"+str(img_id)+"_crop_test":wandb.Image(ldr_pred_test.cpu().numpy()) 
                        })

                    if i % img_save_interval == 0:
                        ldr_pred_full = ldr_pred_test_full.cpu()
                        ldr_pred_full.clamp_max_(1.0)

                        hdr_pred_full = hdr_pred_test_full.cpu()
                        hdr_pred_full.clamp_max_(1.0)

                        img_pred = ldr_pred_test.cpu()
                        img_pred.clamp_max_(1.0)
                        if args.wandb:
                            wandb.log({
                                "test/image_"+str(img_id):wandb.Image(img_pred.numpy()) 
                            })
                            wandb.log({
                                "test/image_hdr_"+str(img_id):wandb.Image(hdr_pred_full.numpy()) 
                            })
                            wandb.log({
                                "test/image_ldr_"+str(img_id):wandb.Image(ldr_pred_full.numpy()) 
                            })
                            # if args.log_mse_image:
                            mse_img = all_mses / all_mses.max()
                            wandb.log({
                                "test/mse_map_"+str(img_id):wandb.Image(mse_img.numpy()) 
                            })
                            # if args.log_depth_map:
                            depth_img = grid.volume_render_depth_image(cam,
                                        args.log_depth_map_use_thresh if
                                        args.log_depth_map_use_thresh else None
                                    )
                            depth_img = torch.masked_select(depth_img, mask.squeeze(-1)).view(height, width // 2)
                            depth_img = viridis_cmap(depth_img.cpu())
                            wandb.log({
                                "test/depth_map_"+str(img_id):wandb.Image(depth_img)
                            })
                            
                            gt_img = rgb_gt_test.cpu().clamp_max_(1.0)
                            wandb.log({
                                "test/gt_img_"+str(img_id):wandb.Image(gt_img.numpy())
                            })
                    # ssim = compute_ssim(rgb_gt_test, ldr_pred_test).item()
                    # lpips = lpips_vgg(rgb_gt_test.permute([2, 0, 1]).cuda().contiguous(),
                    #                   ldr_pred_test.permute([2, 0, 1]).cuda().contiguous(),
                    #                   normalize=True).item()

                    ssim = compute_ssim(rgb_gt_test, ldr_pred_test).item()
                    lpips = lpips_vgg(rgb_gt_test.permute([2, 0, 1]).cuda().contiguous(),
                                    ldr_pred_test.permute([2, 0, 1]).cuda().contiguous(),
                                    normalize=True).item()
                    
                    rgb_pred_test = rgb_gt_test = None
                    mse_num : float = all_mses.mean().item()
                    psnr = -10.0 * math.log10(mse_num)
                    if math.isnan(psnr):
                        print('NAN PSNR', i, img_id, mse_num)
                        assert False
                    stats_test['mse'] += mse_num
                    stats_test['psnr'] += psnr
                    stats_test['ssim'] += ssim
                    stats_test['lpips'] += lpips
                    n_images_gen += 1

                # False
                if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE or \
                grid.basis_type == svox2.BASIS_TYPE_MLP:
                    # Add spherical map visualization
                    EQ_RESO = 256
                    eq_dirs = generate_dirs_equirect(EQ_RESO * 2, EQ_RESO)
                    eq_dirs = torch.from_numpy(eq_dirs).to(device=device).view(-1, 3)

                    if grid.basis_type == svox2.BASIS_TYPE_MLP:
                        sphfuncs = grid._eval_basis_mlp(eq_dirs)
                    else:
                        sphfuncs = grid._eval_learned_bases(eq_dirs)
                    sphfuncs = sphfuncs.view(EQ_RESO, EQ_RESO*2, -1).permute([2, 0, 1]).cpu().numpy()

                    stats = [(sphfunc.min(), sphfunc.mean(), sphfunc.max())
                            for sphfunc in sphfuncs]
                    sphfuncs_cmapped = [viridis_cmap(sphfunc) for sphfunc in sphfuncs]
                    for im, (minv, meanv, maxv) in zip(sphfuncs_cmapped, stats):
                        cv2.putText(im, f"{minv=:.4f} {meanv=:.4f} {maxv=:.4f}", (10, 20),
                                    0, 0.5, [255, 0, 0])
                    sphfuncs_cmapped = np.concatenate(sphfuncs_cmapped, axis=0)
                    if args.wandb:
                        wandb.log({
                                    "test/spheric":wandb.Image(sphfuncs_cmapped.numpy())
                        })
                    # END add spherical map visualization

                stats_test['mse'] /= n_images_gen
                stats_test['psnr'] /= n_images_gen
                stats_test['ssim'] /= n_images_gen
                stats_test['lpips'] /= n_images_gen
                
                if args.wandb:
                    for stat_name in stats_test:
                        wandb.log({"test"+stat_name: stats_test[stat_name]})
                    wandb.log({"epoch_id": float(epoch_id)})
                print('eval stats:', stats_test)

                return stats_test['psnr']
        # if epoch_id % max(factor, args.eval_every) == 0: #and (epoch_id > 0 or not args.tune_mode):
        #     # NOTE: we do an eval sanity check, if not in tune_mode
        cur_psnr = eval_step()
        gc.collect()
        print("[best_psnr, cur_psnr]:", best_psnr, cur_psnr)
        if epoch_id != 0:
            if best_psnr <= cur_psnr:
                best_psnr = cur_psnr
                print(f"Saving BEST checkpoint at epoch {epoch_id}, ckpt_path is {best_ckpt_path}")
                grid.save(best_ckpt_path)
                if args.use_tone_mapping == True:
                    Cam_param.save(best_ckpt_cam_path)
                    # Cam_param.save_txt(args.train_dir)

        def train_step():
            print('Train step')
            pbar = tqdm(enumerate(range(0, epoch_size, args.batch_size)), total=batches_per_epoch)
            stats = {"mse" : 0.0, "psnr" : 0.0, "invsqr_mse" : 0.0, 'c_loss' : 0.0}
            
            for iter_id, batch_begin in pbar:
                if args.use_sh_mask==True :
                    # sh_mask = max((epoch_id/5),1)
                    # sh_mask = (epoch_id/10) * torch.ones((1,27),device=device)
                    sh_mask = min((epoch_id/5), 1) * torch.ones((1, 27), device=device)
                    sh_mask[:,0] = 1
                    sh_mask[:,9] = 1
                    sh_mask[:,18] = 1
                    grid.sh_data.data *= sh_mask
            
                gstep_id = iter_id + gstep_id_base
                if args.lr_fg_begin_step > 0 and gstep_id == args.lr_fg_begin_step:
                    grid.density_data.data[:] = args.init_sigma
                lr_sigma = lr_sigma_func(gstep_id) * lr_sigma_factor
                lr_sh = lr_sh_func(gstep_id) * lr_sh_factor
                lr_basis = lr_basis_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
                lr_sigma_bg = lr_sigma_bg_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
                lr_color_bg = lr_color_bg_func(gstep_id - args.lr_basis_begin_step) * lr_basis_factor
                if not args.lr_decay:
                    lr_sigma = args.lr_sigma * lr_sigma_factor
                    lr_sh = args.lr_sh * lr_sh_factor
                    lr_basis = args.lr_basis * lr_basis_factor

                if args.use_tone_mapping==True:
                    for g in optim_cam.param_groups:
                        g['lr'] = lr_cam_func(gstep_id - args.lr_basis_begin_step) * lr_cam_factor

                batch_end = min(batch_begin + args.batch_size, epoch_size)
                batch_origins = dset.rays.origins[batch_begin: batch_end]
                batch_dirs = dset.rays.dirs[batch_begin: batch_end]
                rgb_gt = dset.rays.gt[batch_begin: batch_end]
                rays = svox2.Rays(batch_origins, batch_dirs)
                
                if dset.white_bkgd:
                    alpha = dset.rays.alpha[batch_begin:batch_end]
                coords = dset.rays.coords[batch_begin: batch_end]
                index = dset.rays.index[batch_begin: batch_end]

                rad_pred = grid.volume_render(rays, True, True)

                if args.use_tone_mapping == True:
                    ldr_pred = Cam_param.RAD2LDR(rad_pred, index, coords, True)
                else : 
                    ldr_pred = rad_pred

                if dset.white_bkgd:
                    ldr_pred = ldr_pred * alpha + (1-alpha) * rad_pred
                #  with Timing("loss_comp"):
                mse = F.mse_loss(rgb_gt, ldr_pred)
                loss = mse
                #TODO: use saturation mask
                if args.use_sat_mask :
                    mask = saturation_mask(rgb_gt, threshold_low=0.15, threshold_high=0.9)
                    mse = torch.mean(mask * ((ldr_pred - rgb_gt) ** 2))
                    if dset.white_bkgd :
                        mse = torch.mean(alpha * mask * ((ldr_pred - rgb_gt)**2))
                        loss = mse + torch.mean((1-alpha)* torch.sub(rad_pred,1)**2)

                # Stats
                mse_num : float = mse.detach().item()
                psnr = -10.0 * math.log10(mse_num)
                stats['mse'] += mse_num
                stats['psnr'] += psnr
                stats['invsqr_mse'] += 1.0 / mse_num ** 2
                # stats['c_loss'] += c_loss

                if (iter_id + 1) % args.print_every == 0:
                    # Print averaged stats
                    pbar.set_description(f'epoch {epoch_id} psnr={psnr:.2f}')
                    for stat_name in stats:
                        stat_val = stats[stat_name] / args.print_every
                        if args.wandb:
                            wandb.log({stat_name: stat_val})    
                        stats[stat_name] = 0.0
                    if args.lambda_tv > 0.0:
                        with torch.no_grad():
                            tv = grid.tv(logalpha=args.tv_logalpha, ndc_coeffs=dset.ndc_coeffs)
                            if args.wandb:
                                wandb.log({'loss_tv': tv.item()})
                    if args.lambda_tv_sh > 0.0:
                        with torch.no_grad():
                            tv_sh = grid.tv_color()
                            if args.wandb:
                                wandb.log({'loss_tv_sh': tv_sh.item()})
                    with torch.no_grad():
                        tv_basis = grid.tv_basis()
                        if args.wandb:
                            wandb.log({'loss_tv_basis': tv_basis.item()})
                    
                    if args.wandb:
                        wandb.log({'lr_sh': lr_sh})
                        wandb.log({'lr_sigma': lr_sigma})
                        
                        if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                            wandb.log({'lr_basis', lr_basis})
                        if grid.use_background:
                            wandb.log({'lr_sigma_bg', lr_sigma_bg})
                            wandb.log({'lr_color_bg', lr_color_bg})
                    
                    if args.weight_decay_sh < 1.0:
                        grid.sh_data.data *= args.weight_decay_sigma
                    if args.weight_decay_sigma < 1.0:
                        grid.density_data.data *= args.weight_decay_sh

                #  # For outputting the % sparsity of the gradient
                #  indexer = grid.sparse_sh_grad_indexer
                #  if indexer is not None:
                #      if indexer.dtype == torch.bool:
                #          nz = torch.count_nonzero(indexer)
                #      else:
                #          nz = indexer.size()
                #      with open(os.path.join(args.train_dir, 'grad_sparsity.txt'), 'a') as sparsity_file:
                #          sparsity_file.write(f"{gstep_id} {nz}\nbac")
                
                if args.use_tone_mapping == True:
                    crf_Loss = Cam_param.crf_smoothness_loss()
                    loss = loss + 1e-3 * crf_Loss
                loss.backward()
                
                # Apply TV/Sparsity regularizers
                if args.lambda_tv > 0.0: #1e-5
                    #  with Timing("tv_inpl"):
                    grid.inplace_tv_grad(grid.density_data.grad,
                            scaling=args.lambda_tv,
                            sparse_frac=args.tv_sparsity,
                            logalpha=args.tv_logalpha,
                            ndc_coeffs=dset.ndc_coeffs,
                            contiguous=args.tv_contiguous)
                if args.lambda_tv_sh > 0.0: # 1e-3
                    #  with Timing("tv_color_inpl"):
                    grid.inplace_tv_color_grad(grid.sh_data.grad,
                            scaling=args.lambda_tv_sh,
                            sparse_frac=args.tv_sh_sparsity,
                            ndc_coeffs=dset.ndc_coeffs,
                            contiguous=args.tv_contiguous)
                if args.lambda_tv_lumisphere > 0.0: # false
                    grid.inplace_tv_lumisphere_grad(grid.sh_data.grad,
                            scaling=args.lambda_tv_lumisphere,
                            dir_factor=args.tv_lumisphere_dir_factor,
                            sparse_frac=args.tv_lumisphere_sparsity,
                            ndc_coeffs=dset.ndc_coeffs)
                if args.lambda_l2_sh > 0.0: # false
                    grid.inplace_l2_color_grad(grid.sh_data.grad,
                            scaling=args.lambda_l2_sh)
                if grid.use_background and (args.lambda_tv_background_sigma > 0.0 or args.lambda_tv_background_color > 0.0):
                    grid.inplace_tv_background_grad(grid.background_data.grad,
                            scaling=args.lambda_tv_background_color,
                            scaling_density=args.lambda_tv_background_sigma,
                            sparse_frac=args.tv_background_sparsity,
                            contiguous=args.tv_contiguous)
                if args.lambda_tv_basis > 0.0: # false 
                    tv_basis = grid.tv_basis()
                    loss_tv_basis = tv_basis * args.lambda_tv_basis
                    loss_tv_basis.backward()
                #  print('nz density', torch.count_nonzero(grid.sparse_grad_indexer).item(),
                #        ' sh', torch.count_nonzero(grid.sparse_sh_grad_indexer).item())
                # Manual SGD/rmsprop step
                if gstep_id >= args.lr_fg_begin_step:
                    grid.optim_density_step(lr_sigma, beta=args.rms_beta, optim=args.sigma_optim)
                    grid.optim_sh_step(lr_sh, beta=args.rms_beta, optim=args.sh_optim)
                    if args.use_tone_mapping == True:
                        optim_cam.step()
                        optim_cam.zero_grad()
                if grid.use_background:
                    grid.optim_background_step(lr_sigma_bg, lr_color_bg, beta=args.rms_beta, optim=args.bg_optim)
                if gstep_id >= args.lr_basis_begin_step:
                    if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
                        grid.optim_basis_step(lr_basis, beta=args.rms_beta, optim=args.basis_optim)
                    elif grid.basis_type == svox2.BASIS_TYPE_MLP:
                        optim_basis_mlp.step()
                        optim_basis_mlp.zero_grad()

        train_step()
        gc.collect()
        gstep_id_base += batches_per_epoch

        #  ckpt_path = path.join(args.train_dir, f'ckpt_{epoch_id:05d}.npz')
        # Overwrite prev checkpoints since they are very huge
        if args.save_every > 0 and (epoch_id + 1) % max(
                factor, args.save_every) == 0 and not args.tune_mode:
            print('Saving', ckpt_path)
            grid.save(ckpt_path)
            if args.use_tone_mapping == True:
                Cam_param.save(ckpt_cam_path)
                Cam_param.save_txt(args.train_dir)

        if (gstep_id_base - last_upsamp_step) >= args.upsamp_every:
            last_upsamp_step = gstep_id_base
            if reso_id < len(reso_list) - 1:
                print('* Upsampling from', reso_list[reso_id], 'to', reso_list[reso_id + 1])
                if args.tv_early_only > 0:
                    print('turning off TV regularization')
                    args.lambda_tv = 0.0
                    args.lambda_tv_sh = 0.0
                elif args.tv_decay != 1.0:
                    args.lambda_tv *= args.tv_decay
                    args.lambda_tv_sh *= args.tv_decay

                reso_id += 1
                use_sparsify = True
                z_reso = reso_list[reso_id] if isinstance(reso_list[reso_id], int) else reso_list[reso_id][2]
                grid.resample(reso=reso_list[reso_id],
                        sigma_thresh=args.density_thresh,
                        weight_thresh=args.weight_thresh / z_reso if use_sparsify else 0.0,
                        dilate=2, #use_sparsify,
                        cameras=resample_cameras if args.thresh_type == 'weight' else None,
                        max_elements=args.max_grid_elements)

                if grid.use_background and reso_id <= 1:
                    grid.sparsify_background(args.background_density_thresh)

                if args.upsample_density_add:
                    grid.density_data.data[:] += args.upsample_density_add

            if factor > 1 and reso_id < len(reso_list) - 1:
                print('* Using higher resolution images due to large grid; new factor', factor)
                factor //= 2
                dset.gen_rays(factor=factor)
                dset.shuffle_rays()

        if gstep_id_base >= args.n_iters:
            print('* Final eval and save')
            eval_step()
            global_stop_time = datetime.now()
            secs = (global_stop_time - global_start_time).total_seconds()
            timings_file = open(os.path.join(args.train_dir, 'time_mins.txt'), 'a')
            timings_file.write(f"{secs / 60}\n")
            if not args.tune_nosave:
                grid.save(ckpt_path)
                if args.use_tone_mapping == True:
                    Cam_param.save(ckpt_cam_path)
                    Cam_param.save_txt(args.train_dir)
            break

if __name__ == "__main__":
    main()
