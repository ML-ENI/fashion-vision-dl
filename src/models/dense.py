"""Baseline visual: píxeles RGB → dos capas Dense → logits de clase."""
from torch import nn


class DenseClassifier(nn.Module):
    def __init__(self, image_size=64, hidden_dims=(80, 32), classes=12, dropout=0.0):
        super().__init__()
        inputs = 3 * image_size * image_size
        layers = [nn.Flatten()]
        for width in hidden_dims:
            layers.extend([nn.Linear(inputs, width), nn.ReLU(), nn.Dropout(dropout)])
            inputs = width
        layers.append(nn.Linear(inputs, classes))
        self.network = nn.Sequential(*layers)

    def forward(self, images):
        # CrossEntropyLoss recibe logits: no aplicar softmax aquí.
        return self.network(images)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
