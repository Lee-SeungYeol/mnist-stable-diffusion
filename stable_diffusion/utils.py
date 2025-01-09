import torch
import torch.nn as nn

import numpy as np

from config import CFG
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf

from einops import rearrange
from PIL import Image



def Euler_Maruyama_sampler(
        score_model, # 시간 의존 스코어 모델
        marginal_prob_std, # 표준편차 반환 함수
        diffusion_coeff, # 확산 계수 함수
        batch_size=64,
        x_shape=(3,160,160),
        num_steps=500,
        device=CFG['device'],
        eps=1e-3, # 수치 안정성을 위한 허용값
        y=None # 타겟 텐서
):
    t=torch.ones(batch_size,device=device)
    init_x=torch.randn(batch_size,*x_shape,device=device)*marginal_prob_std(t)[:,None,None,None] # [batch_size,channel, height, width]
    time_stpes=torch.linspace(1.,eps,num_steps,device=device) #[numsteps]

    step_size=time_stpes[0]-time_stpes[1] # Steip size 시리즈
    x=x=init_x # 시간 t에 대한 초기 샘플

    image_list=[]
    count=0
    with torch.no_grad():
        for time_step in tqdm(time_stpes):
            batch_time_step=torch.ones(batch_size,device=device)*time_step
            g=diffusion_coeff(batch_time_step)
            mean_x=x+(g**2)[:,None,None,None]*score_model(x,batch_time_step,y=y)*step_size

            if (count+1)%100==0:
                image_list.append(mean_x)
            x=mean_x+torch.sqrt(step_size)*g[:,None,None,None]*torch.randn_like(x)
            count+=1

    return mean_x,image_list

def marginal_prob_std(t,sigma):
    t=torch.tensor(t,device=CFG['device'])
    return torch.sqrt((sigma**(2*t)-1)/2/np.log(sigma))


def diffusion_coeff(t,sigma): # 확산 계산 함수
    return torch.tensor(sigma**t,device=CFG['device'])


