import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def set_initial_values(images):
    images = images.numpy()
    linear_imgs = np.power(images, 2.2)
    overall_mean = np.mean(linear_imgs, axis=(0,1,2,3))
    per_image_mean = np.mean(linear_imgs, axis=(1,2,3))
    ref_idx = np.argmin(np.abs(np.subtract(per_image_mean,overall_mean)))

    overall_channel_wise = np.mean(linear_imgs, axis=(0,1,2))
    per_image_channel_wise = np.mean(linear_imgs, axis=(1,2))

    ratio = per_image_channel_wise / overall_channel_wise

    wb = np.log(ratio)

    return torch.Tensor(wb), ref_idx

def leaky_thresholding(Iv, is_train, leaky_values=0.01):
    _Iv = torch.empty_like(Iv).copy_(Iv)

    if is_train:
        add_over = leaky_values * Iv[Iv>1]
        denom = (Iv.abs() + 1e-4).sqrt()
        add_under = (-leaky_values / denom + leaky_values)[Iv<0]
        _Iv[Iv>1] = Iv[Iv>1]+add_over
        _Iv[Iv<0] = Iv[Iv<0]+add_under    
    
    else:
        _Iv[Iv>1] = 1
        _Iv[Iv<0] = 0
    
    return _Iv

def response_function(Iv, CRF, is_train, device, idx, leaky_values=0.01):
    Iv = Iv*2-1
    n, c = Iv.shape

    if idx.shape[0] != n :
        idx = idx.repeat(n)

    Ildr = torch.zeros(n,c) # N_rand, 3
    
    leak_add = torch.zeros_like(Ildr).to(device) # N_rand,3
    
    if is_train:
        leak_add = leak_add + torch.where(Iv>1, leaky_values*Iv, leak_add)
        leak_add = leak_add + torch.where(Iv<-1, (-leaky_values / ((Ildr.abs() + 1e-4).sqrt()) + leaky_values).to(device), leak_add)
    
    for i in range(c):
        response_sl = CRF[:,:,i].view(1,1,CRF.shape[0],-1)
        _idx = idx.to(device)
        sl = torch.cat((Iv[:,i].unsqueeze(-1), _idx.unsqueeze(-1)),axis=1)
        sl = sl.view(1,1,n,2)
        Ildr[:,i] = torch.nn.functional.grid_sample(response_sl, sl, padding_mode='border', align_corners=True)[0,0,0,:]
        
    if is_train==True:
        LDR = Ildr.to(device) + leak_add
    else :
        LDR = Ildr.to(device)

    LDR = (LDR+1)/2
    
    return LDR

class CamParam:
    def __init__(self, N, H, W, device, gts, initialize=True):
        self.device = device

        # learnable white-balance Tensor : N*3
        # initialize with 0 => used as exp(white_balance) > 0
        white_balance_init = torch.zeros((N,3), dtype=torch.float32)
        self.wb = nn.Embedding.from_pretrained(white_balance_init, freeze=False).to(self.device)

        # learnable vignetting Tensor : N*4
        # initialize as H/2, W/2, 0, 0
        vignetting_init = torch.FloatTensor([[H/2, W/2, 0.0, 0.0]]).repeat(N,1)
        self.vig = nn.Embedding.from_pretrained(vignetting_init, freeze=False).to(self.device)

        # learnable CRF Tensor : N*256*3
        # initialize as x^(2.2) (x in [0,1])
        CRF_init = torch.arange(1/256, 1, 1/256).repeat(3).view(3,255).permute(1,0)
        CRF_init = torch.pow(CRF_init, 1/2.2)
        CRF_init = CRF_init*2 - 1
        CRF_init = CRF_init.unsqueeze(0).repeat(N,1,1)
        CRF_init = CRF_init.requires_grad_()
        CRF_init = CRF_init.to(self.device)

        class crf_module(torch.nn.Module):
            def __init__(self, CRF_init):
                super(crf_module, self).__init__()
                self.alpha = nn.Parameter(CRF_init)

        self.CRF = crf_module(CRF_init)

        # Set initial exp, wb values
        if initialize :
            #choose reference image from training close to mean value
            white_balance_init, self.ref_idx = set_initial_values(gts)
            self.wb = nn.Embedding.from_pretrained(white_balance_init, freeze=False).to(self.device)            
        else : 
            #if not initialize, then ref_idx = 0
            self.ref_idx = 1
        
        ref_wb = self.wb.weight[self.ref_idx].detach().cpu().numpy()
        self.ref_wb = torch.exp(torch.Tensor(ref_wb)).to(self.device)
        
        # find boundary from gt dimension
        N, W, H, C = gts.shape
        self.size = (W,H)
        self.num = N
        bounds = []
        for i in range(N+1):
            bounds.append(i*W*H)
        self.boundary = torch.tensor(bounds)
        
    def optimizer(self, l_rate):
        grad_vars = list(self.wb.parameters())
        grad_vars += list(self.vig.parameters())
        grad_vars += list(self.CRF.parameters())
        
        optimizer = torch.optim.RMSprop(params=grad_vars, lr = l_rate, alpha=0.95)
        return optimizer
            
    def rad2ldr(self, HDR, white_balance, CRF, is_train, device, idx):
        if is_train == True:
            ref_idx = (idx == self.ref_idx).nonzero(as_tuple=True)[0]
            non_ref_idx = (idx != self.ref_idx).nonzero(as_tuple=True)[0]

            Iw = torch.zeros_like(HDR)
            Iw[non_ref_idx] = HDR[non_ref_idx] * white_balance[non_ref_idx]
            Iw[ref_idx] = HDR[ref_idx] * self.ref_wb  
        else:
            Iw = HDR * white_balance
        LDR = response_function(Iw, CRF, is_train, device, idx)
        return LDR
    
    def RAD2LDR(self, rad_pred, idx, coords, is_train):
        N_wb = torch.exp(self.wb.weight[idx])

        N_CRF = list(self.CRF.parameters())[0][idx,:,:].to(self.device)
        N_CRF = torch.cat((-1*torch.ones(N_CRF.shape[0],1,3, requires_grad=False).to(self.device), N_CRF, torch.ones(N_CRF.shape[0], 1, 3, requires_grad=False).to(self.device)), dim=1)
        
        ldr_pred = self.rad2ldr(rad_pred, N_wb, coords, N_CRF, is_train, self.device, idx)

        return ldr_pred
    
    def RAD2LDR_img(self, rad_img, idx):
        is_train = False
        H, W, _ = rad_img.shape
        
        rad_img = rad_img.reshape(H*W, -1)
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)), -1)
        coords = coords.reshape(H*W, -1).to(self.device)
                
        N_wb = torch.exp(self.wb.weight[idx:idx+1])

        N_CRF = list(self.CRF.parameters())[0][idx,:,:].to(self.device)
        N_CRF = torch.cat((-1*torch.ones(1,3, requires_grad=False).to(self.device), N_CRF, torch.ones(1, 3, requires_grad=False).to(self.device)), dim=0)
        N_CRF = N_CRF.unsqueeze(0)

        ldr_img = self.rad2ldr(rad_img, N_wb, coords, N_CRF, is_train, self.device, idx.unsqueeze(0))
        ldr_img = ldr_img.reshape(H,W,3)

        return ldr_img

    def RAD2LDR_img_control(self, rad_img, idx):
        is_train = False
        H, W, _ = rad_img.shape
        
        rad_img = rad_img.reshape(H*W, -1)
        coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)), -1)
        coords = coords.reshape(H*W, -1).to(self.device)
                
        N_wb = torch.exp(self.wb.weight[idx:idx+1])

        N_CRF = list(self.CRF.parameters())[0][idx,:,:].to(self.device)
        N_CRF = torch.cat((-1*torch.ones(1,3, requires_grad=False).to(self.device), N_CRF, torch.ones(1, 3, requires_grad=False).to(self.device)), dim=0)
        N_CRF = N_CRF.unsqueeze(0)

        ldr_img = self.rad2ldr(rad_img, N_wb, coords, N_CRF, is_train, self.device, idx.unsqueeze(0))
        ldr_img = ldr_img.reshape(H,W,3)

        return ldr_img
    
    def crf_smoothness_loss(self):
        CRF = list(self.CRF.parameters())[0][:,:,:].to(self.device)
        CRF = torch.cat((-1*torch.ones((len(CRF),1,3), requires_grad=False).to(self.device), CRF, torch.ones((len(CRF),1, 3), requires_grad=False).to(self.device)), dim=1)
        g_pp = -2*CRF[:, 1:256, :] + CRF[:,:255,:] + CRF[:,2:257,:]
        crf_loss = torch.sum(torch.square(g_pp))
        return crf_loss
    
    def save(self, path):
        torch.save({
            'white_balance': self.wb.state_dict(),
            'vignetting': self.vig.state_dict(),
            'CRF': self.CRF.state_dict(),
        }, path)
    
    def save_txt(self, dir):
        
        white_balance = torch.exp(self.wb.weight).detach().cpu().numpy()
        vignetting = self.vig.weight.detach().cpu().numpy()
        
        np.savetxt(dir+"/wb.txt", white_balance)
        np.savetxt(dir+"/vig.txt", vignetting)

        CRF = list(self.CRF.parameters())[0]
        CRF = torch.cat((-1*torch.ones(CRF.shape[0], 1, 3, requires_grad=False).to(self.device), CRF, torch.ones(CRF.shape[0], 1, 3, requires_grad=False).to(self.device)), dim=1)
        CRF = (torch.add(CRF, 1)/2)
        cam_resp_func = CRF.detach().cpu().numpy()
        
        i = 0
        with open(dir+"/crf.txt", 'w') as outfile:
                # I'm writing a header here just for the sake of readability
                # Any line starting with "#" will be ignored by numpy.loadtxt
                outfile.write('number of camera: {0}\n'.format(i))

                # Iterating through a ndimensional array produces slices along
                # the last axis. This is equivalent to data[i,:,:] in this case
                for data_slice in cam_resp_func:
                    i = i+1
                    # The formatting string indicates that I'm writing out
                    # the values in left-justified columns 7 characters in width
                    # with 2 decimal places.  
                    np.savetxt(outfile, data_slice.T, fmt='%-7.2f')

                    # Writing out a break to indicate different slices...
                    outfile.write('number of camera: {0}\n'.format(i))
    
        # np.savetxt(dir+"/crf.txt", cam_resp_func)
        
        x = np.arange(0,1+1/256, 1/256)
        plt.plot(np.log(x), cam_resp_func[0])
        plt.xlabel('log Exposure')
        plt.savefig(dir+"/sample_crf.png")

    def load_ckpt(self, ckpt):
        print(ckpt)
        self.wb.load_state_dict(torch.load(ckpt)['white_balance'])
        self.vig.load_state_dict(torch.load(ckpt)['vignetting'])
        self.CRF.load_state_dict(torch.load(ckpt)['CRF'])
