import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


from einops import rearrange

class Dense(nn.Module): #특징 계산 클래스
    def __init__(self, input_dim, output_dim):# 입력 차원, 출려 차원
        super().__init__()
        self.dense=nn.Linear(input_dim,output_dim)

    def forward(self,x):
        return self.dense(x)[...,None,None] # 마지막에 차원을 추가하여 4D 텐서로 변경
    

class GaussianFourierProJection(nn.Module):
    def __init__(self,embed_dim,scale=30):
        super().__init__()
        self.W=nn.Parameter(torch.randn(embed_dim//2)*scale,requires_grad=False)

    def forward(self,x):
        x_proj=x[:,None]*self.W[None,:]*np.pi
        return torch.cat([torch.sin(x_proj),torch.cos(x_proj)],dim=-1)


class Unet(nn.Module):
    def __init__(self,marginal_prob_std, channels=[32,64,128,256],embed_dim=256):
        """
        marginal_prob_std: 시간 t에 대한 표준편차 반환 함수
        channels: 각 해상도의 특징 맵의 채널 수
        embed_dim: 가우시안 랜덤 특징 임베딩 차원
        """
        super().__init__()

        self.time_embed=nn.Sequential( # 시간에 대한 가우시안 랜덤 특징 임베딩 계층
            GaussianFourierProJection(embed_dim=embed_dim),
            nn.Linear(embed_dim,embed_dim)
        )

        # 인코딩 레이어 구현
        self.conv1=nn.Conv2d(3,channels[0],kernel_size=3,stride=1,bias=False)# mnist여서 1채널
        self.dense1=Dense(embed_dim,channels[0])
        self.gnorm1=nn.GroupNorm(4,num_channels=channels[0]) # 그룹 정규화
        self.conv2=nn.Conv2d(channels[0],channels[1],kernel_size=3,stride=2,bias=False)
        self.dense2=Dense(embed_dim,channels[1])
        self.gnorm2=nn.GroupNorm(32,num_channels=channels[1])

        self.conv3=nn.Conv2d(channels[1],channels[2],kernel_size=3,stride=2,bias=False)
        self.dense3=Dense(embed_dim,channels[2])
        self.gnorm3=nn.GroupNorm(32,channels[2])
        self.conv4=nn.Conv2d(channels[2],channels[3],kernel_size=3,stride=2,bias=False)
        self.dense4=Dense(embed_dim,channels[3])
        self.gnorm4=nn.GroupNorm(32,num_channels=channels[3])

        # 디코딩 레이어 구현(해상도 증가)
        self.tconv4=nn.ConvTranspose2d(channels[3],channels[2],kernel_size=3,stride=2,bias=False,output_padding=1)
        self.dense5=Dense(embed_dim,channels[2])
        self.tgnorm4=nn.GroupNorm(32,channels[2])
        self.tconv3=nn.ConvTranspose2d(channels[2], channels[1],kernel_size=3,stride=2,bias=False,output_padding=1)
        self.dense6=Dense(embed_dim,channels[1])
        self.tgnorm3=nn.GroupNorm(32,channels[1])

        self.tconv2=nn.ConvTranspose2d(channels[1],channels[0],kernel_size=3,stride=2,bias=False,output_padding=1)
        self.dense7=Dense(embed_dim,channels[0])
        self.tgnorm2=nn.GroupNorm(32,channels[0])
        self.tconv1=nn.ConvTranspose2d(channels[0],3,kernel_size=3,stride=1)

        # 스위시시 활성화 함수
        self.act=lambda x:x*torch.sigmoid(x)
        self.marginal_prob_std=marginal_prob_std

    def forward(self,x,t,y=None):
        """
        x는 입력텐서
        t는 시간 텐서
        y는 타겟 텐서
        h는 U-net을 통과한 출력텐서서
        """
        embed=self.act(self.time_embed(t))
        
        

        #인코딩
        h1=self.conv1(x)+self.dense1(embed)
        h1=self.act(self.gnorm1(h1))
        #print("h1: ",h1.shape)
        

        h2=self.conv2(h1)+self.dense2(embed)
        h2=self.act(self.gnorm2(h2))
        #print("h2: ",h2.shape)

        h3=self.conv3(h2)+self.dense3(embed)
        h3=self.act(self.gnorm3(h3))
        #print("h3: ",h3.shape)

        h4=self.conv4(h3)+self.dense4(embed)
        h4=self.act(self.gnorm4(h4))
        #print("h4: ",h4.shape)

        # 디코딩딩
        h=self.tconv4(h4)
        h+=self.dense5(embed)
        h=self.act(self.tgnorm4(h))
        #print("h: ",h.shape)

        h=self.tconv3(h+h3)
        h+=self.dense6(embed)
        h=self.act(self.tgnorm3(h))

        h=self.tconv2(h+h2)
        h+=self.dense7(embed)
        h=self.act(self.tgnorm2(h))

        h=self.tconv1(h+h1)

        h=h/self.marginal_prob_std(t)[:,None,None,None]

        return h
    

class CrossAttention(nn.Module):
    def __init__(self,embed_dim,hidden_dim,context_dim=None, num_heads=1): #임베딩 차원, 은닉차원, 컨텍스트 차원(self attention이면 None), 어텐션 헤드 수
        super(CrossAttention,self).__init__()

        self.hidden_dim=hidden_dim
        self.context_dim=context_dim
        self.embed_dim=embed_dim

        self.query=nn.Linear(hidden_dim,embed_dim,bias=False)# 쿼리에 대한 학습
        if context_dim is None:
            self.self_attn=True
            self.key=nn.Linear(hidden_dim,embed_dim,bias=False)
            self.value=nn.Linear(hidden_dim,hidden_dim,bias=False)

        else:
            self.self_attn=False
            self.key=nn.Linear(context_dim,embed_dim,bias=False)
            self.value=nn.Linear(context_dim,hidden_dim,bias=False)

    def forward(self,tokens,context=None): # 토큰들[배치, 시퀀스크기, 은닉차원], 컨텍스트 정보[배치, 컨텍스트 시퀀스 크기, 컨텍스트 차원]
        if self.self_attn: # self-attention
            Q=self.query(tokens)
            K=self.key(tokens)
            V=self.value(tokens)
        else:
            Q=self.query(tokens)
            K=self.key(context)
            V=self.value(context)

        
        # Compute score matrices, attention matrices, context vectors
        scoremats=torch.einsum("BTH,BSH->BTS", Q, K)  # Q, K간 내적 계산. 스코어 행렬 획득
        """    
            Q: torch.Size([1024, 25, 128])
            K: torch.Size([1024, 25, 128])
            scoremats: torch.Size([1024, 25, 25])
        """

        attnmats = F.softmax(scoremats / math.sqrt(self.embed_dim), dim=-1) # 스코어 행렬의 softmax 계산
        ctx_vecs = torch.einsum("BTS,BSH->BTH", attnmats, V)  # 어텐션 행렬 적용된 V벡터 계산 
        return ctx_vecs    
    
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, context_dim):
        super(TransformerBlock, self).__init__()

        self.attn_self=CrossAttention(hidden_dim,hidden_dim)
        self.attn_cross=CrossAttention(hidden_dim,hidden_dim,context_dim)

        self.norm1=nn.LayerNorm(hidden_dim)
        self.norm2=nn.LayerNorm(hidden_dim)
        self.norm3=nn.LayerNorm(hidden_dim)

        self.ffn=nn.Sequential(
            nn.Linear(hidden_dim,3*hidden_dim),
            nn.GELU(),
            nn.Linear(3*hidden_dim,hidden_dim)
        ) # 피드 포워드 네트워크 2개의 은닉 레이어구성

    def forward(self,x,context=None):
        x=self.attn_self(self.norm1(x))+x
        x=self.attn_cross(self.norm2(x),context=context)+x
        x=self.ffn(self.norm3(x))+x

        return x
    
class SpatialTransformer(nn.Module):
    def __init__(self,hidden_dim,context_dim):
        super(SpatialTransformer,self).__init__()
        self.transformer=TransformerBlock(hidden_dim,context_dim)

    def forward(self,x,context=None): # x: 입력텐서 [배치, 채널, 높이, 너비], context: 컨택스트 텐서[배치, 컨텍스트 시퀀스 크기, 컨텍스트 차원]
        b,c,h,w=x.shape
        x_in=x

        x=rearrange(x, "b c h w -> b (h w) c") # 입력텐서 재배열
        x=self.transformer(x,context)
        x=rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        return x+x_in


class Unet_Transformer(nn.Module):
    def __init__(self,marginal_prob_std, channels=[32,64,128,256],embed_dim=256,text_dim=256, nClass=10):
        """
        marginal_prob_std: 시간 t에 대한 표준편차 반환 함수
        channels: 각 해상도의 특징 맵의 채널 수
        embed_dim: 가우시안 랜덤 특징 임베딩 차원
        """
        super().__init__()

        self.time_embed=nn.Sequential( # 시간에 대한 가우시안 랜덤 특징 임베딩 계층
            GaussianFourierProJection(embed_dim=embed_dim),
            nn.Linear(embed_dim,embed_dim)
        )

        # 인코딩 레이어 구현
        self.conv1=nn.Conv2d(3,channels[0],kernel_size=3,stride=1,bias=False)# mnist여서 1채널
        self.dense1=Dense(embed_dim,channels[0])
        self.gnorm1=nn.GroupNorm(4,num_channels=channels[0]) # 그룹 정규화
        self.conv2=nn.Conv2d(channels[0],channels[1],kernel_size=3,stride=2,bias=False)
        self.dense2=Dense(embed_dim,channels[1])
        self.gnorm2=nn.GroupNorm(32,num_channels=channels[1])

        self.conv3=nn.Conv2d(channels[1],channels[2],kernel_size=3,stride=2,bias=False)
        self.dense3=Dense(embed_dim,channels[2])
        self.gnorm3=nn.GroupNorm(32,channels[2])
        self.attn3=SpatialTransformer(channels[2],text_dim) # 컨텍스트 정보, 텍스트 임베딩 차원을 공간 트랜스포머에 설정정
        
        self.conv4=nn.Conv2d(channels[2],channels[3],kernel_size=3,stride=2,bias=False)
        self.dense4=Dense(embed_dim,channels[3])
        self.gnorm4=nn.GroupNorm(32,num_channels=channels[3])
        self.attn4=SpatialTransformer(channels[3],text_dim)

        # 디코딩 레이어 구현(해상도 증가)
        self.tconv4=nn.ConvTranspose2d(channels[3],channels[2],kernel_size=3,stride=2,bias=False,output_padding=1)
        self.dense5=Dense(embed_dim,channels[2])
        self.tgnorm4=nn.GroupNorm(32,channels[2])
        self.tconv3=nn.ConvTranspose2d(channels[2], channels[1],kernel_size=3,stride=2,bias=False,output_padding=1)
        self.dense6=Dense(embed_dim,channels[1])
        self.tgnorm3=nn.GroupNorm(32,channels[1])

        self.tconv2=nn.ConvTranspose2d(channels[1],channels[0],kernel_size=3,stride=2,bias=False,output_padding=1)
        self.dense7=Dense(embed_dim,channels[0])
        self.tgnorm2=nn.GroupNorm(32,channels[0])
        self.tconv1=nn.ConvTranspose2d(channels[0],3,kernel_size=3,stride=1)

        # 스위시시 활성화 함수
        self.act=lambda x:x*torch.sigmoid(x)
        self.marginal_prob_std=marginal_prob_std
        self.cond_embed=nn.Embedding(nClass,text_dim)

    def forward(self,x,t,y=None):
        """
        x는 입력텐서
        t는 시간 텐서
        y는 타겟 텐서
        h는 U-net을 통과한 출력텐서서
        """
        embed=self.act(self.time_embed(t))
        y_embed=self.cond_embed(y).unsqueeze(1)
        
        

        #인코딩
        h1=self.conv1(x)+self.dense1(embed)
        h1=self.act(self.gnorm1(h1))
        #print("h1: ",h1.shape)
        

        h2=self.conv2(h1)+self.dense2(embed)
        h2=self.act(self.gnorm2(h2))
        #print("h2: ",h2.shape)

        h3=self.conv3(h2)+self.dense3(embed)
        h3=self.act(self.gnorm3(h3))
        h3=self.attn3(h3,y_embed)
        #print("h3: ",h3.shape)

        h4=self.conv4(h3)+self.dense4(embed)
        h4=self.act(self.gnorm4(h4))
        h4=self.attn4(h4,y_embed)
        #print("h4: ",h4.shape)

        # 디코딩딩
        h=self.tconv4(h4)
        h+=self.dense5(embed)
        h=self.act(self.tgnorm4(h))
        #print("h: ",h.shape)

        h=self.tconv3(h+h3)
        h+=self.dense6(embed)
        h=self.act(self.tgnorm3(h))

        h=self.tconv2(h+h2)
        h+=self.dense7(embed)
        h=self.act(self.tgnorm2(h))

        h=self.tconv1(h+h1)

        h=h/self.marginal_prob_std(t)[:,None,None,None]

        return h