import torch
import torch.nn as nn
from torchvision.models import resnet50

class ResNet(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=True):
        super(ResNet, self).__init__()
        # Pretrained ResNet50 모델 로드
        self.base_model = resnet50(pretrained=pretrained)
        
        # Feature 추출을 위해 FC 계층을 제거
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        
        # ResNet의 출력 feature 크기 확인
        in_features = 2048  # ResNet50의 마지막 레이어 출력 크기
        
        # Projector 계층 정의
        self.projector = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        # Base 모델을 통과한 features
        features = self.base_model(x).flatten(1)  # Flattening to (batch_size, in_features)
        # Projector 계층 통과
        projections = self.projector(features)
        return features, projections
