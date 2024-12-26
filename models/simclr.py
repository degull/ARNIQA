import torch
import torch.nn as nn
import torch.nn.functional as F
from dotmap import DotMap
from models.resnet import ResNet


def nt_xent_loss(a: torch.Tensor, b: torch.Tensor, tau: float = 0.1, mos: torch.Tensor = None):
    """
    Compute the NT-Xent loss with optional MOS scaling.

    Args:
        a (torch.Tensor): First set of features
        b (torch.Tensor): Second set of features
        tau (float): Temperature parameter
        mos (torch.Tensor, optional): MOS values to weight the loss
    """
    a_norm = torch.norm(a, dim=1, keepdim=True)
    b_norm = torch.norm(b, dim=1, keepdim=True)
    a_cap = a / a_norm
    b_cap = b / b_norm

    sim = torch.mm(a_cap, b_cap.T) / tau
    labels = torch.arange(a.size(0), device=a.device)
    loss = F.cross_entropy(sim, labels)

    if mos is not None:
        loss = loss * mos.mean()  # Scale loss by MOS values
    return loss


class SimCLR(nn.Module):
    """
    SimCLR model class used for pre-training the encoder for IQA.

    Args:
        encoder_params (DotMap): encoder parameters with keys
            - embedding_dim (int): embedding dimension of the encoder projection head
            - pretrained (bool): whether to use pretrained weights for the encoder
            - use_norm (bool): whether to normalize the embeddings
        temperature (float): temperature for the loss function. Default: 0.1
    """
    def __init__(self, encoder_params: DotMap, temperature: float = 0.1):
        super(SimCLR, self).__init__()

        self.encoder = ResNet(embedding_dim=encoder_params.embedding_dim,
                              pretrained=encoder_params.pretrained,
                              use_norm=encoder_params.use_norm)

        self.temperature = temperature

    def compute_loss(self, proj_A, proj_B, mos=None):
        """
        Compute the NT-Xent loss with the option to use MOS values.
        """
        return nt_xent_loss(proj_A, proj_B, tau=self.temperature, mos=mos)

    def forward(self, im_q, im_k=None):
        print(f"[DEBUG] im_q shape before encoder: {im_q.shape}")
        if im_k is not None:
            print(f"[DEBUG] im_k shape before encoder: {im_k.shape}")
            q, proj_q = self.encoder(im_q)
            print(f"[DEBUG] q shape: {q.shape}, proj_q shape: {proj_q.shape}")
            k, proj_k = self.encoder(im_k)
            print(f"[DEBUG] k shape: {k.shape}, proj_k shape: {proj_k.shape}")
            return proj_q, proj_k
        else:
            _, proj_q = self.encoder(im_q)
            print(f"[DEBUG] proj_q shape: {proj_q.shape}")
            return proj_q

