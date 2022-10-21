import os
import glob
from turtle import color
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

device = "cuda:0"

def load_crf(ckpt_path):
    N = 50
    cam_ckpt = os.path.join(ckpt_path, "ckpt_cam.npz")
    CRF_init = torch.arange(1/256, 1, 1/256).repeat(3).view(3,255).permute(1,0)
    CRF_init = torch.pow(CRF_init, 1/2.2)
    CRF_init = CRF_init*2 - 1
    CRF_init = CRF_init.unsqueeze(0).repeat(N,1,1)
    CRF_init = CRF_init.requires_grad_()
    CRF_init = CRF_init.to(device)

    class crf_module(torch.nn.Module):
        def __init__(self, CRF_init):
            super(crf_module, self).__init__()
            self.alpha = nn.Parameter(CRF_init)

    CRF = crf_module(CRF_init)

    CRF.load_state_dict(torch.load(cam_ckpt)['CRF'])
    CRF = list(CRF.parameters())[0]
    print("[CRF shape]:", CRF.shape)

    CRF = torch.cat((-1*torch.ones(CRF.shape[0], 1, 3, requires_grad=False).to(device), 
                    CRF.to(device), 
                    torch.ones(CRF.shape[0], 1, 3, requires_grad=False).to(device)), dim=1)
    CRF = (torch.add(CRF, 1)/2)
    cam_resp_func = CRF.detach().cpu().numpy()

    # x = np.arange(0,1+1/256, 1/256)
    # print("first:", len(x))

    # x = np.arange(1, 256)
    x = list(range(1, 256))
    x.append(255)
    x.insert(0, 0.9)
    x = np.array(x)
    # print(x)
    
    print("second:", len(x))    

    print("third")
    print(cam_resp_func.shape)
    # print(cam_resp_func)
    cam_resp_func *= 255
    # print("After:")
    # print(cam_resp_func)
    # print(cam_resp_func[0][:,0].shape)
    # return
    N, _, _ = cam_resp_func.shape

    return cam_resp_func

def main():
    filmic_idx = [i*5 + 1 for i in range(10)]
    print(filmic_idx)

    ckpt_path = "/local_data/ugkim/hdr_synthetic/llff/classroom_filmic+"
    # ckpt_path_2 = "/local_data/ugkim/hdr_synthetic/llff/classroom_filmic+_tone_no-smooth"
    # data_path = "/local_data/hdr_synthetic/llff/filmic+"

    # ckpt_path = "/local_data/ugkim/hdr_synthetic/llff/classroom_filmic"
    # ckpt_path_2 = "/local_data/ugkim/hdr_synthetic/llff/classroom_tone_sh-mask"
    
    cam_resp_func_filmic = load_crf(ckpt_path)
    # cam_resp_func_standard = load_crf(ckpt_path_2)


    x = list(range(1, 256))
    x.append(255)
    x.insert(0, 0.9)
    x = np.array(x)

    save_path = "crf_filmic+"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    N = 50
    for i in range(N):
        plt.clf()
        plt.plot(np.log(x), cam_resp_func_filmic[i])
        print(f"shape {i}:", cam_resp_func_filmic[i].shape)
        # plt.plot(np.log(x), cam_resp_func[30][:,2])
        plt.xlabel('log exposure')
        plt.ylabel('pixel value')
        plt.savefig(f"./{save_path}/crf_{i}.png")

    # i = 5
    # plt.plot(np.log(x), cam_resp_func_filmic[i][:, 0], label="R", color='indianred',linewidth=4)
    # plt.plot(np.log(x), cam_resp_func_filmic[i][:, 1], label="G", color='seagreen',linewidth=4)
    # plt.plot(np.log(x), cam_resp_func_filmic[i][:, 2], label="B", color='royalblue',linewidth=4)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.xlabel('log exposure', fontsize=20)
    # plt.ylabel('pixel value', fontsize=20)
    # plt.legend(fontsize=15)
    # plt.savefig(f"./{save_path}/crf_filmic_{i}.png", bbox_inches='tight', pad_inches=0.1)

    # plt.clf()
    # plt.plot(np.log(x), cam_resp_func_standard[i][:, 0], label="R", color='indianred',linewidth=4)
    # plt.plot(np.log(x), cam_resp_func_standard[i][:, 1], label="G", color='seagreen',linewidth=4)
    # plt.plot(np.log(x), cam_resp_func_standard[i][:, 2], label="B", color='royalblue',linewidth=4)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.xlabel('log exposure', fontsize=20)
    # plt.ylabel('pixel value', fontsize=20)
    # plt.legend(fontsize=15)
    # plt.savefig(f"./{save_path}/crf_standard_{i}.png", bbox_inches='tight', pad_inches=0.1)

    # plt.clf()
    # plt.plot(np.log(x), cam_resp_func_standard[i][:, 0], label="standard", color='lightpink', linewidth=4)
    # plt.plot(np.log(x), cam_resp_func_filmic[i][:, 0], label="filmic", color='slateblue',linewidth=4)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.xlabel('log exposure', fontsize=20)
    # plt.ylabel('pixel value', fontsize=20)
    # plt.legend(fontsize=15)
    # plt.savefig(f"./{save_path}/crf_r_{i}.png", bbox_inches='tight', pad_inches=0.1)

    # plt.clf()
    # plt.plot(np.log(x), cam_resp_func_standard[i][:, 1], label="standard", color='lightpink',linewidth=4)
    # plt.plot(np.log(x), cam_resp_func_filmic[i][:, 1], label="filmic", color='slateblue',linewidth=4)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.xlabel('log exposure', fontsize=20)
    # plt.ylabel('pixel value', fontsize=20)
    # plt.legend(fontsize=15)
    # plt.savefig(f"./{save_path}/crf_g_{i}.png", bbox_inches='tight', pad_inches=0.1)


    # plt.clf()
    # plt.plot(np.log(x), cam_resp_func_standard[i][:, 2], label="standard", color='lightpink',linewidth=4)
    # plt.plot(np.log(x), cam_resp_func_filmic[i][:, 2], label="filmic", color='slateblue',linewidth=4)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.xlabel('log exposure', fontsize=20)
    # plt.ylabel('pixel value', fontsize=20)
    # plt.legend(fontsize=15)
    # plt.savefig(f"./{save_path}/crf_b_{i}.png", bbox_inches='tight', pad_inches=0.1)



if __name__ == "__main__":
    main()