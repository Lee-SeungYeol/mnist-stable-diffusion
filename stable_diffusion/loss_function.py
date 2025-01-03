import torch

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