import torch
import torch.nn as nn
from models.resnet import ResNet

class SimCLR(nn.Module):
    def __init__(self, encoder_params, temperature=0.1):
        super(SimCLR, self).__init__()
        self.encoder = ResNet(embedding_dim=encoder_params.embedding_dim, pretrained=encoder_params.pretrained)
        self.temperature = temperature
        self.linear_regressor = nn.Linear(encoder_params.embedding_dim, 1)  # 선형 회귀기 추가

    def forward(self, img_A, img_B):
        _, proj_A = self.encoder(img_A)
        _, proj_B = self.encoder(img_B)
        return proj_A, proj_B

    def compute_loss(self, proj_A, proj_B):
        return self.nt_xent_loss(proj_A, proj_B)

    def nt_xent_loss(self, proj_A, proj_B):
        proj = torch.cat([proj_A, proj_B], dim=0)
        logits = torch.mm(proj, proj.T) / self.temperature
        labels = torch.arange(proj_A.size(0)).repeat(2).to(proj.device)
        return nn.CrossEntropyLoss()(logits, labels)

    def predict_quality(self, img):
        _, proj = self.encoder(img)
        return self.linear_regressor(proj)
