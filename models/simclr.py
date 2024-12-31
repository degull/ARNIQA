
import torch
import torch.nn as nn
from models.resnet import ResNet

class SimCLR(nn.Module):
    def __init__(self, encoder_params, temperature=0.1):
        super(SimCLR, self).__init__()
        self.encoder = ResNet(
            embedding_dim=encoder_params.embedding_dim,
            pretrained=encoder_params.pretrained
        )
        self.temperature = temperature
        self.criterion = nt_xent_loss  # nt_xent_loss를 초기화

    def forward(self, im_q, im_k=None):
        if im_q.dim() == 5:
            im_q = im_q.view(-1, *im_q.shape[2:])
        if im_k is not None and im_k.dim() == 5:
            im_k = im_k.view(-1, *im_k.shape[2:])

        q, proj_q = self.encoder(im_q)

        if im_k is not None:
            k, proj_k = self.encoder(im_k)
        else:
            proj_k = None  # Key가 없는 경우 None으로 설정

        if not self.training:
            return proj_q, proj_k  # 검증 단계에서 두 값 반환

        loss = self.criterion(proj_q, proj_k, self.temperature)
        return loss

def nt_xent_loss(proj_q, proj_k, tau=0.1):
    q_norm = nn.functional.normalize(proj_q, dim=1)
    k_norm = nn.functional.normalize(proj_k, dim=1)
    batch_size = proj_q.size(0)

    positives = torch.sum(q_norm * k_norm, dim=1) / tau
    negatives = torch.mm(q_norm, q_norm.t()) / tau

    labels = torch.arange(batch_size).to(proj_q.device)
    mask = torch.eye(batch_size).bool().to(proj_q.device)

    negatives = negatives.masked_fill(mask, -float('inf'))
    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss
