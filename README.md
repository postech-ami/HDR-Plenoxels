# HDR-Plenoxels (ECCV 2022)

### [Paper](https://arxiv.org/abs/2208.06787) | [Project Page](https://hdr-plenoxels.github.io/)

This repository is official implementation for the ECCV 2022 paper, [HDR-Plenoxels: Self-Calibrating High Dynamic Range Radiance Fields](https://arxiv.org/abs/2208.06787). 

HDR-Plenoxels is end-to-end HDR radiance fields learning method w/ only LDR images of the varying camera and w/o additional camera information (e.g., exposure value).
We deign the tone-mapping module based on a physical camera pipeline.
We also deploy a multi-view dataset containing varying camera conditions.

https://user-images.githubusercontent.com/38632805/197250380-5048a4a4-f462-46ee-acf6-f1d4cf848795.mov

## Setup

This code is based on [Plenoxels](https://github.com/sxyu/svox2) official implementation. 
You have to follow setup detail of Plenoxels repository (below).


First create the virtualenv; we recommend using conda:
```sh
conda env create -f environment.yml
conda activate plenoxel
```

Then clone the repo and install the library at the root (svox2), which includes a CUDA extension.

If your CUDA toolkit is older than 11, then you will need to install CUB as follows:
`conda install -c bottler nvidiacub`.
Since CUDA 11, CUB is shipped with the toolkit.

To install the main library, simply run
```
pip install .
```
In the repo root directory.

## Prepare Datasets

We deploy our HDR training dataset for LLFF format, and the dataset will be auto-detected.


Please get the synthetic and real LLFF datasets from [this link](<https://postechackr-my.sharepoint.com/personal/gucka28_postech_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fgucka28%5Fpostech%5Fac%5Fkr%2FDocuments%2FHDR%2DPlenoxels%5Fdataset&ga=1>).


## Voxel Optimization (Training)

For training a single scene, see `opt/hdr_opt.py`.

You can train both of our synthetic and real HDR datasets.
Inside `opt/`, run below shell scripts.

If you want to use synthetic datset, run below shell scripts.
```bash
# Plenoxels + static dataset
./shell/syn/train_mid.sh

# Plenoxels + varying datset
./shell/syn/train_mix.sh

# HDR-Plenoxes + varying datset
./shell/syn/train_tone.sh
```

If you want to use real datset, run below shell scripts.
```bash
# Plenoxels + static dataset
./shell/real/train_mid.sh

# Plenoxels + varying datset
./shell/real/train_mix.sh

# HDR-Plenoxes + varying datset
./shell/real/train_tone.sh
```

We do not provide pretrained checkpoints.

## Evaluation

Use `opt/shell/render/render_hdr.sh` for rendering HDR radiance fields.

Use `opt/shell/render/render_ldr.sh` for rendering LDR radiance fields which is final output.

If you don't want to save all frames, which is very slow, add `--no_imsave` to avoid this.

## Metric

Inside opt/, run
```bash
CUDA_VISIBLE_DEVICES=0 python hdr_calc_metrics.py
```

You can get PSNR, SSIM, and LPIPS scores for right-half novel views.

## Citation

```
@inproceedings{jun2022hdr,
    title     = {HDR-Plenoxels: Self-Calibrating High Dynamic Range Radiance Fields},
    author    = {Jun-Seong, Kim and Yu-Ji, Kim and Ye-Bin, Moon and Oh, Tae-Hyun},
    booktitle = {ECCV},
    year      = {2022},
}
```
