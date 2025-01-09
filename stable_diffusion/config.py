import torch
CFG={
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'lr':5e-3,
    'EPOCH':3000,
    'embed-dim':256,
    'text-dim':256,
    'channels':[128,128,256,256,256,512,512],
    'image-shape':416,

    'time-steps':1000

}