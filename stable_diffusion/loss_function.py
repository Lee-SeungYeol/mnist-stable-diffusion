import torch
import math

from torch.optim import lr_scheduler

def loss_fn(model,x,marginal_prob_std,eps=1e-5):# 시간 의존 스코어 모델 x: 훈련데이터 미니배치
    random_t=torch.rand(x.shape[0],device=x.device)*(1.-2*eps)+eps # 미니 배치 크기 만큼 랜덤시간 샘플링
    std=marginal_prob_std(random_t) #랜덤 시간에 대한 표준편차 계산
    z=torch.randn_like(x) # 미니배치 크기만큼 정규 분포 랜덤 노이즈 생성
    perturbed_x=x+z*std[:,None, None,None] # 노이즈로 입력데이터 왜곡

 

    score=model(perturbed_x,random_t)
    loss=torch.mean(torch.sum((score*std[:,None,None,None]+z)**2,dim=(1,2,3)))

    return loss

def loss_fn_cond(model,x,y,marginal_prob_std,eps=1e-5): #model : 시간 의존된 스코어 기반 모델, x 입력데이터 미니배치, y: 조건 정보(ex.:) 입력 텍스트 숫자), marginal_prob_std: 표준 푠차 반환 함수, eps: 수치 안정성을 위한 허용값
    random_t=torch.rand(x.shape[0],device=x.device)*(1.-eps)+eps
    z=torch.randn_like(x) # 노이즈 생성
    std=marginal_prob_std(random_t) # 랜덤 시간에 따른 표준편차 계산
    perturbed_x=x+z*std[:,None,None,None]

    score=model(perturbed_x,random_t,y) # 모델을 사용해 왜곡된 데이터와 시간에 대한 스코어획득, 트랜스포머 모델에 쿼리 입력을 위한 Y 추가

    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1, 2, 3)))

    return loss,score


class CosineAnnealingWarmUpRestarts(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (
                        1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr