import torch
import torch.nn as nn
from models.resnet import ResNet


class SimCLR(nn.Module):

    def __init__(self, encoder_params: dict, temperature: float = 0.1):
        super(SimCLR, self).__init__()

        # Extract encoder parameters from the dictionary
        embedding_dim = encoder_params['embedding_dim']
        pretrained = encoder_params['pretrained']
        use_norm = encoder_params['use_norm']

        # Initialize the ResNet encoder
        self.encoder = ResNet(
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            use_norm=use_norm
        )

        self.temperature = temperature
        self.criterion = nt_xent_loss

    def forward(self, im_q, im_k=None):
        #print(f"im_q shape: {im_q.shape}")  # Debugging message
        assert im_q.ndim == 4, f"Expected 4D input for im_q, but got {im_q.shape}"
        q, proj_q = self.encoder(im_q)

        if not self.training:
            return q, proj_q

        assert im_k is not None, "im_k must be provided during training"
        assert im_k.ndim == 4, f"Expected 4D input for im_k, but got {im_k.shape}"
        k, proj_k = self.encoder(im_k)

        loss = self.criterion(proj_q, proj_k, self.temperature)
        return loss




def nt_xent_loss(a: torch.Tensor, b: torch.Tensor, tau: float = 0.1):

    a_norm = torch.norm(a, dim=1).reshape(-1, 1)
    a_cap = torch.div(a, a_norm)
    b_norm = torch.norm(b, dim=1).reshape(-1, 1)
    b_cap = torch.div(b, b_norm)

    a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
    sim = torch.mm(a_cap_b_cap, a_cap_b_cap.t())
    sim /= tau

    exp_sim = torch.exp(sim)
    mask = torch.eye(exp_sim.size(0), device=exp_sim.device).bool()
    exp_sim = exp_sim.masked_fill(mask, 0)

    row_sum = exp_sim.sum(dim=1, keepdim=True)
    log_prob = sim - torch.log(row_sum + 1e-10)

    pos_mask = torch.cat(
        [torch.arange(exp_sim.size(0) // 2, device=exp_sim.device) for _ in range(2)],
        dim=0
    ).reshape(exp_sim.size(0), -1)
    pos_log_prob = log_prob.gather(dim=1, index=pos_mask).diag()
    loss = -pos_log_prob.mean()
    return loss


## 원본과 동일하게
""" 
import torch
import torch.nn
from dotmap import DotMap

from models.resnet import ResNet


class SimCLR(torch.nn.Module):

    def __init__(self, encoder_params: DotMap, temperature: float = 0.1):
        super(SimCLR, self).__init__()

        # Initialize the ResNet encoder with provided parameters
        self.encoder = ResNet(
            embedding_dim=encoder_params.embedding_dim,
            pretrained=encoder_params.pretrained,
            use_norm=encoder_params.use_norm
        )

        self.temperature = temperature
        self.criterion = nt_xent_loss

    def forward(self, im_q, im_k=None):
        # Extract embeddings for the query image
        q, proj_q = self.encoder(im_q)

        if not self.training:
            # During evaluation, return query embeddings
            return q, proj_q

        # Extract embeddings for the key image during training
        k, proj_k = self.encoder(im_k)

        # Compute NT-Xent loss
        loss = self.criterion(proj_q, proj_k, self.temperature)
        return loss


def nt_xent_loss(a: torch.Tensor, b: torch.Tensor, tau: float = 0.1):

    # Normalize the feature vectors
    a_norm = torch.norm(a, dim=1, keepdim=True)
    a_cap = a / (a_norm + 1e-10)
    b_norm = torch.norm(b, dim=1, keepdim=True)
    b_cap = b / (b_norm + 1e-10)

    # Concatenate normalized features
    a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)

    # Compute similarity matrix and scale by temperature
    sim = torch.mm(a_cap_b_cap, a_cap_b_cap.t()) / tau

    # Compute exponential of similarity and mask diagonal
    exp_sim = torch.exp(sim)
    mask = torch.eye(exp_sim.size(0), device=exp_sim.device).bool()
    exp_sim = exp_sim.masked_fill(mask, 0)

    # Compute row-wise normalization
    row_sum = exp_sim.sum(dim=1, keepdim=True)
    log_prob = sim - torch.log(row_sum + 1e-10)

    # Construct positive pair mask
    pos_mask = torch.arange(exp_sim.size(0) // 2, device=exp_sim.device).repeat(2)
    pos_mask = pos_mask.reshape(exp_sim.size(0), -1)

    # Extract log probabilities for positive pairs
    pos_log_prob = log_prob.gather(dim=1, index=pos_mask).diag()

    # Compute loss as negative mean log probability
    loss = -pos_log_prob.mean()
    return loss
 """