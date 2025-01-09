import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim=dim

    def forward(self,time):
        device=time.device
        half_dim=self.dim//2
       
        embeddings=math.log(10000)/(half_dim-1)
        embeddings=torch.exp(torch.arange(half_dim,device=device)* -embeddings)
        embeddings=time[:,None]*embeddings[None,:]
        embeddings=torch.cat((embeddings.sin(),embeddings.cos()),dim=-1)
        return embeddings

class DiffusionModel:
    def __init__(self,start_schedule=0.0001, end_scedule=0.02,timesteps=1000):
        self.start_schedule=start_schedule
        self.end_scedule=end_scedule
        self.timesteps=timesteps

        self.betas=torch.linspace(start_schedule,end_scedule,timesteps)
        self.alphas=1-self.betas
        self.alphas_cumprod=torch.cumprod(self.alphas,axis=0) #누적곱

    def forward(self,x_0,t,device):

        noise=torch.randn_like(x_0)
        sqrt_alphas_cumprod_t=self.get_index_from_list(self.alphas_cumprod.sqrt(),t,x_0.shape)
        sqrt_one_minus_alphas_cumprod_t=self.get_index_from_list(torch.sqrt(1.-self.alphas_cumprod),t,x_0.shape)

        mean=sqrt_alphas_cumprod_t.to(device)*x_0.to(device)
        variance=sqrt_one_minus_alphas_cumprod_t.to(device)*noise.to(device)

        return mean+variance,noise.to(device)
    
    def backward(self,x,t,model,labels):
        betas_t=self.get_index_from_list(self.betas,t,x.shape)
        sqrt_one_minus_alphas_comprod_t=self.get_index_from_list(torch.sqrt(1.-self.alphas_cumprod),t,x.shape)
        sqrt_recip_alhas_t=self.get_index_from_list(torch.sqrt(1.0/self.alphas),t,x.shape)

        mean=sqrt_recip_alhas_t*(x-betas_t*model(x,t,labels)/sqrt_one_minus_alphas_comprod_t)
        posterior_variance_t=betas_t

        if t==0:
            return mean
        else:
            noise=torch.randn_like(x)
            variance=torch.sqrt(posterior_variance_t)*noise
            return mean+variance
    @staticmethod
    def get_index_from_list(values,t,x_shape):
        batch_size=t.shape[0]
        

        result=values.gather(-1,t.cpu())
       
        """
        batch_size=5
        result: 5
        len(x_shape)=4
        
        """

        return result.reshape(batch_size,*((1,)*(len(x_shape)-1))).to(t.device)
    
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

        self.hidden_dim=hidden_dim
        self.context_dim=context_dim

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
    

class Block(nn.Module):
    def __init__(self, channels_in, channels_out, time_embedding_dims, labels, num_filters = 3, downsample=True,text_dim=128):
        super().__init__()

        self.time_embedding_dims = time_embedding_dims
        self.time_embedding = SinusoidalPositionEmbeddings(time_embedding_dims)
        self.labels = labels
        

        self.downsample = downsample
        self.channels_out=channels_out

        if downsample:
            self.conv1 = nn.Conv2d(channels_in, channels_out, num_filters, padding=1)
            self.final = nn.Conv2d(channels_out, channels_out, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(2 * channels_in, channels_out, num_filters, padding=1)
            self.final = nn.ConvTranspose2d(channels_out, channels_out, 4, 2, 1)

        self.bnorm1 = nn.BatchNorm2d(channels_out)
        self.bnorm2 = nn.BatchNorm2d(channels_out)

        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1)
        self.time_mlp = nn.Linear(time_embedding_dims, channels_out)
        self.relu = nn.ReLU()
        self.attn=SpatialTransformer(channels_out,text_dim)

    def forward(self, x, t, y_embed):
        o = self.bnorm1(self.relu(self.conv1(x)))
        o_time = self.relu(self.time_mlp(self.time_embedding(t)))
        o = o + o_time[(..., ) + (None, ) * 2]

        o = self.bnorm2(self.relu(self.conv2(o)))
        
        if self.channels_out==1024 or self.channels_out==512:
            o=self.attn(o,y_embed)

        return self.final(o)
    

    class UNet_Transformer(nn.Module):
        def __init__(self, 
                    img_channels=3,
                    time_embedding_dims=128,
                    labels=False,
                    sequence_channels=(64, 128, 256, 512, 1024),
                    ):
            super().__init__()
            self.time_embedding_dims = time_embedding_dims
            sequence_channels_rev = reversed(sequence_channels)

            # Downsampling path
            self.downsampling = nn.ModuleList([
                Block(channels_in, channels_out, time_embedding_dims, labels) 
                for channels_in, channels_out in zip(sequence_channels, sequence_channels[1:])
            ])

            # Upsampling path 
            self.upsampling = nn.ModuleList([
                Block(channels_in, channels_out, time_embedding_dims, labels, downsample=False)
                for channels_in, channels_out in zip(sequence_channels[::-1], sequence_channels[::-1][1:])
            ])

            # Input and output convolutions
            self.conv1 = nn.Conv2d(img_channels, sequence_channels[0], kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(sequence_channels[0], img_channels, kernel_size=1)
            self.cond_embed=nn.Embedding(10,128)
        def forward(self, x, t, y):
            # Store residual connections
            residuals = []
            
            y_embed=self.cond_embed(y).unsqueeze(1)
            #print(y_embed.shape)#torch.Size([8, 128])
            # Initial convolution
            o = self.conv1(x)

            # Downsampling
            for down_block in self.downsampling:
                o = down_block(o, t, y_embed)
                residuals.append(o)

            # Upsampling with skip connections
            for up_block, residual in zip(self.upsampling, reversed(residuals)):
                o = up_block(torch.cat((o, residual), dim=1), t, y_embed)

            return self.conv2(o)