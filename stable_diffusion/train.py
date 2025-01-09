import torch
import torchvision
import torch.optim as optim

from torchvision import transforms
from torchvision.datasets import ImageFolder


import matplotlib.pyplot as plt
import numpy as np
import functools as functions
from tqdm import tqdm

from utils import marginal_prob_std,diffusion_coeff,Euler_Maruyama_sampler
from config import CFG
from loss_function import loss_fn,loss_fn_cond
from model_layer import Unet,Unet_Transformer

transform = transforms.Compose([
    transforms.Resize((CFG['image-shape'],CFG['image-shape'])),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1] 범위로 변환
    
])
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print(CFG)
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

train_dataset=ImageFolder(root='C:/Users/admin/Desktop/VSC/custom_AI_data',
                     transform=transform)

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=22,shuffle=False)

sigma=25.0
marginal_prob_std_fn=functions.partial(marginal_prob_std,sigma=sigma)
diffusion_coeff_fn=functions.partial(diffusion_coeff,sigma=sigma)

score_model=torch.nn.DataParallel(Unet_Transformer(marginal_prob_std=marginal_prob_std_fn,channels=CFG['channels'],embed_dim=CFG['embed-dim'],text_dim=CFG['text-dim']))
score_model=score_model.to(CFG['device'])
optimizer=optim.Adam(score_model.parameters(),lr=CFG['lr'])
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.2, 0.98 ** epoch))
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
count=5
for epoch in range(1,CFG['EPOCH']+1):
    score_model.train()
    avg_loss=0
    num_items=0
    for images,labels in tqdm(train_loader):
        images=images.to(CFG['device'])
        loss,score = loss_fn_cond(score_model,images,labels,marginal_prob_std_fn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss+=loss.item()*images.shape[0]
        num_items+=images.shape[0]
        break
        


    scheduler.step()
    lr_current = scheduler.get_last_lr()[0]

    print(f"EPOCH: {epoch}   Train Average Loss: {avg_loss/num_items}  LR_CURRENT: {lr_current}" )

    torch.save(score_model.state_dict(),'../../models/stable_diffusion.pth')


    if count%5==0:
        score_model.eval()
        with torch.no_grad():
            test_label=0 # bear
            sample_batch_size=1
            sampler=Euler_Maruyama_sampler
            sampler,image_list=sampler(score_model,
                            marginal_prob_std_fn,
                            diffusion_coeff_fn,
                            sample_batch_size,
                            x_shape=(3,128,128),
                            num_steps=1000,
                            device=CFG['device'],
                            y=test_label*torch.ones(sample_batch_size, dtype=torch.long))
            
            sampler.clamp(0,1)

            plt.imshow(image_list[0][0].clamp(0,1).permute(1,2,0).cpu().detach())
            plt.savefig('./save_image/generation_image_0'+'.png')
            plt.imshow(image_list[1][0].clamp(0,1).permute(1,2,0).cpu().detach())
            plt.savefig('./save_image/generation_image_1'+'.png')
            plt.imshow(image_list[2][0].clamp(0,1).permute(1,2,0).cpu().detach())
            plt.savefig('./save_image/generation_image_2'+'.png')
            plt.imshow(image_list[3][0].clamp(0,1).permute(1,2,0).cpu().detach())
            plt.savefig('./save_image/generation_image_3'+'.png')
            plt.imshow(image_list[4][0].clamp(0,1).permute(1,2,0).cpu().detach())
            plt.savefig('./save_image/generation_image_4'+'.png')
            plt.imshow(image_list[5][0].clamp(0,1).permute(1,2,0).cpu().detach())
            plt.savefig('./save_image/generation_image_5'+'.png')
            plt.imshow(image_list[6][0].clamp(0,1).permute(1,2,0).cpu().detach())
            plt.savefig('./save_image/generation_image_6'+'.png')
            plt.imshow(image_list[7][0].clamp(0,1).permute(1,2,0).cpu().detach())
            plt.savefig('./save_image/generation_image_7'+'.png')
            plt.imshow(image_list[8][0].clamp(0,1).permute(1,2,0).cpu().detach())
            plt.savefig('./save_image/generation_image_8'+'.png')
            plt.imshow(image_list[9][0].clamp(0,1).permute(1,2,0).cpu().detach())
            plt.savefig('./save_image/generation_image_9'+'.png')
            # plt.imshow(sampler[0].permute(1,2,0).cpu().detach())
            # plt.savefig('./save_image/generation_image.png')
