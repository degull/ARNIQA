
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50


class ResNet(nn.Module):

    def __init__(self, embedding_dim: int, pretrained: bool = True, use_norm: bool = True):
        super(ResNet, self).__init__()

        self.pretrained = pretrained
        self.use_norm = use_norm
        self.embedding_dim = embedding_dim

        if self.pretrained:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.model = resnet50(weights=weights)

        self.feat_dim = self.model.fc.in_features
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        # Projector uses embedding_dim for output
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

    def forward(self, x):
        f = self.model(x)
        f = f.view(-1, self.feat_dim)

        if self.use_norm:
            f = F.normalize(f, dim=1)

        g = self.projector(f)
        if self.use_norm:
            return f, F.normalize(g, dim=1)
        else:
            return f, g
 