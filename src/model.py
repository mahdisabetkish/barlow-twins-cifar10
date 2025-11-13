import torch.nn as nn
from torchvision.models import resnet18


class BarlowTwins(nn.Module):
    def __init__(self, projector_dims: list = None):
        super().__init__()
        
        if projector_dims is None:
            projector_dims = [512, 512, 512]

        self.backbone = resnet18()
        self.backbone.fc = nn.Identity()

        in_features = 512  # Output size of ResNet-18

        layers = []
        for i in range(len(projector_dims)):
            in_dim = in_features if i == 0 else projector_dims[i - 1]
            out_dim = projector_dims[i]

            layers.append(nn.Linear(in_dim, out_dim, bias=False))

            # No BatchNorm or ReLU on the last layer
            if i < len(projector_dims) - 1:
                layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU())

        self.projector = nn.Sequential(*layers)

    def forward(self, y1, y2):
        feature1 = self.backbone(y1)
        feature2 = self.backbone(y2)

        z1 = self.projector(feature1)
        z2 = self.projector(feature2)

        return z1, z2


class LinearClassifier(nn.Module):
    def __init__(self, num_features=512, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.linear(x)
