


## 이게원본
import torch
import torch.nn as nn
from dotmap import DotMap
import sys
import os

# 현재 파일의 상위 디렉토리를 PYTHONPATH에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.resnet import ResNet


class SimCLR(nn.Module):
    def __init__(self, encoder_params: DotMap, temperature: float = 0.1):
        super(SimCLR, self).__init__()
        self.encoder = ResNet(embedding_dim=encoder_params.embedding_dim,
                              pretrained=encoder_params.pretrained,
                              use_norm=encoder_params.use_norm)
        self.temperature = temperature

    def forward(self, img_A, img_B):
        # img_A와 img_B의 차원이 [batch_size, num_crops, C, H, W]일 것으로 가정
        if img_A.dim() != 5 or img_B.dim() != 5:
            raise ValueError("Input images must have dimensions [batch_size, num_crops, 3, H, W].")

        batch_size, num_crops, C, H, W = img_A.size()  # [batch_size, num_crops, 3, 224, 224]

        # 디버깅: 변환 전 차원 확인
        print(f"Before view - img_A shape: {img_A.shape}, img_B shape: {img_B.shape}")

        # Encoder를 통해 feature 추출
        proj_A = self.encoder(img_A.view(-1, C, H, W))  # [batch_size * num_crops, embedding_dim]
        proj_B = self.encoder(img_B.view(-1, C, H, W))  # [batch_size * num_crops, embedding_dim]

        # proj_A와 proj_B가 tuple인 경우 첫 번째 요소 선택
        if isinstance(proj_A, tuple):
            proj_A = proj_A[0]  # 필요한 경우 첫 번째 요소를 선택
        if isinstance(proj_B, tuple):
            proj_B = proj_B[0]  # 필요한 경우 첫 번째 요소를 선택

        # 디버깅: 변환 후 차원 확인
        print(f"After view - proj_A shape: {proj_A.shape}, proj_B shape: {proj_B.shape}")

        return proj_A, proj_B

    def compute_loss(self, proj_q, proj_p):
        # NT-Xent 손실 계산
        return self.nt_xent_loss(proj_q, proj_p)

    def nt_xent_loss(self, a: torch.Tensor, b: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
        a_norm = torch.norm(a, dim=1).reshape(-1, 1)
        a_cap = torch.div(a, a_norm)
        b_norm = torch.norm(b, dim=1).reshape(-1, 1)
        b_cap = torch.div(b, b_norm)
        a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
        a_cap_b_cap_transpose = torch.t(a_cap_b_cap)
        b_cap_a_cap = torch.cat([b_cap, a_cap], dim=0)
        sim = torch.mm(a_cap_b_cap, a_cap_b_cap_transpose)
        sim_by_tau = torch.div(sim, tau)
        exp_sim_by_tau = torch.exp(sim_by_tau)
        sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
        exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
        numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(a_cap_b_cap, b_cap_a_cap), tau))
        denominators = sum_of_rows - exp_sim_by_tau_diag
        num_by_den = torch.div(numerators, denominators)
        neglog_num_by_den = -torch.log(num_by_den)
        return torch.mean(neglog_num_by_den)

if __name__ == "__main__":
    # 간단한 테스트 케이스
    import torch
    from dotmap import DotMap

    # 임시 encoder 파라미터 설정
    encoder_params = DotMap({
        'embedding_dim': 128,
        'pretrained': False,
        'use_norm': True
    })

    # SimCLR 모델 초기화
    model = SimCLR(encoder_params)

    # 임시 데이터 생성: [batch_size, num_crops, C, H, W] 크기의 텐서
    img_A = torch.randn(2, 5, 3, 224, 224)
    img_B = torch.randn(2, 5, 3, 224, 224)

    # 모델의 forward 함수 테스트
    proj_A, proj_B = model(img_A, img_B)

    # 결과 출력
    print(f"proj_A shape: {proj_A.shape}, proj_B shape: {proj_B.shape}")




## 시각화 위한 코드
""" 
import torch
import torch.nn as nn
from dotmap import DotMap
import sys
import os

# 현재 파일의 상위 디렉토리를 PYTHONPATH에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.resnet import ResNet

class SimCLR(nn.Module):
    def __init__(self, encoder_params: DotMap, temperature: float = 0.1):
        super(SimCLR, self).__init__()
        self.encoder = ResNet(embedding_dim=encoder_params.embedding_dim,
                              pretrained=encoder_params.pretrained,
                              use_norm=encoder_params.use_norm)
        self.temperature = temperature

        # 체크포인트 구조에 맞는 projector 레이어 확장
        self.encoder.projector = nn.Sequential(
            nn.Linear(self.encoder.feat_dim, 4096),  # 체크포인트의 크기에 맞춤
            nn.ReLU(),
            nn.Linear(4096, encoder_params.embedding_dim)
        )
    def forward(self, img_A, img_B):
        # img_A와 img_B의 차원이 [batch_size, num_crops, C, H, W]일 것으로 가정
        if img_A.dim() != 5 or img_B.dim() != 5:
            raise ValueError("Input images must have dimensions [batch_size, num_crops, 3, H, W].")

        batch_size, num_crops, C, H, W = img_A.size()  # [batch_size, num_crops, 3, 224, 224]

        # 디버깅: 변환 전 차원 확인
        print(f"Before view - img_A shape: {img_A.shape}, img_B shape: {img_B.shape}")

        # Encoder를 통해 feature 추출
        proj_A = self.encoder(img_A.view(-1, C, H, W))  # [batch_size * num_crops, embedding_dim]
        proj_B = self.encoder(img_B.view(-1, C, H, W))  # [batch_size * num_crops, embedding_dim]

        # proj_A와 proj_B가 tuple인 경우 첫 번째 요소 선택
        if isinstance(proj_A, tuple):
            proj_A = proj_A[0]  # 필요한 경우 첫 번째 요소를 선택
        if isinstance(proj_B, tuple):
            proj_B = proj_B[0]  # 필요한 경우 첫 번째 요소를 선택

        # 디버깅: 변환 후 차원 확인
        print(f"After view - proj_A shape: {proj_A.shape}, proj_B shape: {proj_B.shape}")

        return proj_A, proj_B

    def compute_loss(self, proj_q, proj_p):
        # NT-Xent 손실 계산
        return self.nt_xent_loss(proj_q, proj_p)

    def nt_xent_loss(self, a: torch.Tensor, b: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
        a_norm = torch.norm(a, dim=1).reshape(-1, 1)
        a_cap = torch.div(a, a_norm)
        b_norm = torch.norm(b, dim=1).reshape(-1, 1)
        b_cap = torch.div(b, b_norm)
        a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
        a_cap_b_cap_transpose = torch.t(a_cap_b_cap)
        b_cap_a_cap = torch.cat([b_cap, a_cap], dim=0)
        sim = torch.mm(a_cap_b_cap, a_cap_b_cap_transpose)
        sim_by_tau = torch.div(sim, tau)
        exp_sim_by_tau = torch.exp(sim_by_tau)
        sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
        exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
        numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(a_cap_b_cap, b_cap_a_cap), tau))
        denominators = sum_of_rows - exp_sim_by_tau_diag
        num_by_den = torch.div(numerators, denominators)
        neglog_num_by_den = -torch.log(num_by_den)
        return torch.mean(neglog_num_by_den)

if __name__ == "__main__":
    # 간단한 테스트 케이스
    import torch
    from dotmap import DotMap

    # 임시 encoder 파라미터 설정
    encoder_params = DotMap({
        'embedding_dim': 128,
        'pretrained': False,
        'use_norm': True
    })

    # SimCLR 모델 초기화
    model = SimCLR(encoder_params)

    # 임시 데이터 생성: [batch_size, num_crops, C, H, W] 크기의 텐서
    img_A = torch.randn(2, 5, 3, 224, 224)
    img_B = torch.randn(2, 5, 3, 224, 224)

    # 모델의 forward 함수 테스트
    proj_A, proj_B = model(img_A, img_B)

    # 결과 출력
    print(f"proj_A shape: {proj_A.shape}, proj_B shape: {proj_B.shape}")


 """

""" 
import torch
import torch.nn as nn
from dotmap import DotMap
import sys
import os

# 현재 파일의 상위 디렉토리를 PYTHONPATH에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.resnet import ResNet


class SimCLR(nn.Module):
    def __init__(self, encoder_params: DotMap, temperature: float = 0.1):
        super(SimCLR, self).__init__()
        self.encoder = ResNet(embedding_dim=encoder_params.embedding_dim,
                              pretrained=encoder_params.pretrained,
                              use_norm=encoder_params.use_norm)
        self.temperature = temperature

    def forward(self, img_A, img_B):
        # img_A와 img_B의 차원이 [batch_size, num_crops, C, H, W]일 것으로 가정
        if img_A.dim() != 5 or img_B.dim() != 5:
            raise ValueError("Input images must have dimensions [batch_size, num_crops, 3, H, W].")

        batch_size, num_crops, C, H, W = img_A.size()  # [batch_size, num_crops, 3, 224, 224]

        # 디버깅: 변환 전 차원 확인
        print(f"Before view - img_A shape: {img_A.shape}, img_B shape: {img_B.shape}")

        # Encoder를 통해 feature 추출
        proj_A = self.encoder(img_A.reshape(-1, C, H, W))  # [batch_size * num_crops, embedding_dim]
        proj_B = self.encoder(img_B.reshape(-1, C, H, W))  # [batch_size * num_crops, embedding_dim]

        # proj_A와 proj_B가 tuple인 경우 첫 번째 요소 선택
        if isinstance(proj_A, tuple):
            proj_A = proj_A[0]  # 필요한 경우 첫 번째 요소를 선택
        if isinstance(proj_B, tuple):
            proj_B = proj_B[0]  # 필요한 경우 첫 번째 요소를 선택

        # 디버깅: 변환 후 차원 확인
        print(f"After view - proj_A shape: {proj_A.shape}, proj_B shape: {proj_B.shape}")

        return proj_A, proj_B
    


    def compute_loss(self, proj_q, proj_p):
        # NT-Xent 손실 계산
        return self.nt_xent_loss(proj_q, proj_p)

    #def nt_xent_loss(self, a: torch.Tensor, b: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    #    a_norm = torch.norm(a, dim=1).reshape(-1, 1)
    #    a_cap = torch.div(a, a_norm)
    #    b_norm = torch.norm(b, dim=1).reshape(-1, 1)
    #    b_cap = torch.div(b, b_norm)
    #    a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
    #    a_cap_b_cap_transpose = torch.t(a_cap_b_cap)
    #    b_cap_a_cap = torch.cat([b_cap, a_cap], dim=0)
    #    sim = torch.mm(a_cap_b_cap, a_cap_b_cap_transpose)
    #    sim_by_tau = torch.div(sim, tau)
    #    exp_sim_by_tau = torch.exp(sim_by_tau)
    #    sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
    #    exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
    #    numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(a_cap_b_cap, b_cap_a_cap), tau))
    #    denominators = sum_of_rows - exp_sim_by_tau_diag
    #    num_by_den = torch.div(numerators, denominators)
    #    neglog_num_by_den = -torch.log(num_by_den)
    #    return torch.mean(neglog_num_by_den)
    


    def nt_xent_loss(self, a: torch.Tensor, b: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
        eps = 1e-10  # 최소값을 설정하여 NaN 및 음수 손실 방지
        a_norm = torch.norm(a, dim=1).clamp(min=eps).reshape(-1, 1)
        b_norm = torch.norm(b, dim=1).clamp(min=eps).reshape(-1, 1)
        a_cap = a / a_norm
        b_cap = b / b_norm

        pos_sim = torch.exp((a_cap * b_cap).sum(dim=1) / tau).clamp(min=eps)  # [batch_size]

        a_b_combined = torch.cat([a_cap, b_cap], dim=0)
        sim_matrix = torch.mm(a_b_combined, a_b_combined.transpose(0, 1)) / tau
        exp_sim_matrix = torch.exp(sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0])

        sum_of_rows = exp_sim_matrix.sum(dim=1).clamp(min=eps)  # 분모가 0이 되지 않도록 최소값 설정

        loss_a = -torch.log(pos_sim / sum_of_rows[:a.size(0)]).clamp(min=eps)
        loss_b = -torch.log(pos_sim / sum_of_rows[a.size(0):]).clamp(min=eps)

        loss = torch.cat([loss_a, loss_b]).mean()
        return loss



if __name__ == "__main__":
    # 간단한 테스트 케이스
    import torch
    from dotmap import DotMap

    # 임시 encoder 파라미터 설정
    encoder_params = DotMap({
        'embedding_dim': 128,
        'pretrained': False,
        'use_norm': True
    })

    # SimCLR 모델 초기화
    model = SimCLR(encoder_params)

    # 임시 데이터 생성: [batch_size, num_crops, C, H, W] 크기의 텐서
    img_A = torch.randn(2, 5, 3, 224, 224)
    img_B = torch.randn(2, 5, 3, 224, 224)

    # 모델의 forward 함수 테스트
    proj_A, proj_B = model(img_A, img_B)

    # 결과 출력
    print(f"proj_A shape: {proj_A.shape}, proj_B shape: {proj_B.shape}") """


#import torch
#import torch.nn as nn
#from dotmap import DotMap
#import sys
#import os
#
## 현재 파일의 상위 디렉토리를 PYTHONPATH에 추가
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#
#from models.resnet import ResNet
#
#class SimCLR(nn.Module):
#    """
#    SimCLR model class used for pre-training the encoder for IQA.
#
#    Args:
#        encoder_params (dict): encoder parameters with keys
#            - embedding_dim (int): embedding dimension of the encoder projection head
#            - pretrained (bool): whether to use pretrained weights for the encoder
#            - use_norm (bool): whether to normalize the embeddings
#        temperature (float): temperature for the loss function. Default: 0.1
#
#    Returns:
#        if training:
#            loss (torch.Tensor): loss value
#        if not training:
#            q (torch.Tensor): image embeddings before the projection head (NxC)
#            proj_q (torch.Tensor): image embeddings after the projection head (NxC)
#    """
#    def __init__(self, encoder_params: DotMap, temperature: float = 0.1):
#        super(SimCLR, self).__init__()
#        self.encoder = ResNet(embedding_dim=encoder_params.embedding_dim,
#                              pretrained=encoder_params.pretrained,
#                              use_norm=encoder_params.use_norm)
#        self.temperature = temperature
#        self.criterion = self.nt_xent_loss
#
#    def forward(self, img_A, img_B):
#        # img_A와 img_B가 [batch_size, num_crops, num_distortions, C, H, W] 형식이라고 가정
#        batch_size, num_crops, num_distortions, C, H, W = img_A.size()
#        
#        # ResNet 모델에 입력할 수 있도록 [batch_size * num_crops * num_distortions, C, H, W] 형식으로 변환
#        img_A = img_A.view(-1, C, H, W)
#        img_B = img_B.view(-1, C, H, W)
#
#        # ResNet을 통해 feature 추출, tuple로 반환되면 첫 번째 요소만 선택
#        proj_A = self.encoder(img_A)
#        proj_B = self.encoder(img_B)
#
#        if isinstance(proj_A, tuple):
#            proj_A = proj_A[0]
#        if isinstance(proj_B, tuple):
#            proj_B = proj_B[0]
#
#        # 다시 [batch_size, num_crops, num_distortions, embedding_dim] 형식으로 복원
#        proj_A = proj_A.view(batch_size, num_crops, num_distortions, -1)
#        proj_B = proj_B.view(batch_size, num_crops, num_distortions, -1)
#
#        return proj_A, proj_B
#    
#    def compute_loss(self, proj_A, proj_B):
#        """
#        NT-Xent 손실 계산 메서드입니다. proj_A와 proj_B의 특징 벡터 간의 유사도를 기반으로 손실을 계산합니다.
#        """
#        return self.nt_xent_loss(proj_A, proj_B, self.temperature)
#
#
#
#    def nt_xent_loss(self, a: torch.Tensor, b: torch.Tensor, tau: float = 0.1):
#        """
#        NT-Xent (Normalized Temperature-scaled Cross Entropy) 손실을 계산하는 메서드입니다.
#        """
#        eps = 1e-10  # 계산의 안정성을 위한 최소값
#
#        # 특징 벡터를 정규화하여 유사도 계산의 안정성을 높임
#        a_cap = a / (torch.norm(a, dim=-1, keepdim=True) + eps)
#        b_cap = b / (torch.norm(b, dim=-1, keepdim=True) + eps)
#
#        # 양성 샘플 간 유사도 계산
#        pos_sim = torch.exp((a_cap * b_cap).sum(dim=-1) / tau)
#        
#        # 모든 샘플 간 유사도 행렬 계산
#        all_cap = torch.cat([a_cap.view(-1, a_cap.size(-1)), b_cap.view(-1, b_cap.size(-1))], dim=0)
#        sim_matrix = torch.mm(all_cap, all_cap.t()) / tau
#
#        # 오버플로우 방지 조치 (큰 값에서 최대값을 뺀 후 exp 계산)
#        exp_sim_matrix = torch.exp(sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0])
#        sum_exp_sim = exp_sim_matrix.sum(dim=1, keepdim=True) + eps
#
#        # 분자와 분모의 크기 조정
#        pos_exp_sim = pos_sim.view(-1, 1)
#        sum_exp_sim = sum_exp_sim[:pos_exp_sim.size(0)]
#
#        # 손실 계산
#        num_by_den = pos_exp_sim / sum_exp_sim
#        loss = -torch.log(num_by_den + eps).mean()
#
#        return loss  # torch.relu 제거
#
#
#
#
#if __name__ == "__main__":
#    # 간단한 테스트 케이스
#    encoder_params = DotMap({
#        'embedding_dim': 128,
#        'pretrained': False,
#        'use_norm': True
#    })
#
#    # SimCLR 모델 초기화
#    model = SimCLR(encoder_params)
#
#    # 임시 데이터 생성: [batch_size, num_crops, C, H, W] 크기의 텐서
#    img_A = torch.randn(2, 5, 3, 224, 224)
#    img_B = torch.randn(2, 5, 3, 224, 224)
#
#    # 모델의 forward 함수 테스트
#    proj_A, proj_B = model(img_A, img_B)
#
#    # 결과 출력
#    print(f"proj_A shape: {proj_A.shape}, proj_B shape: {proj_B.shape}")
#