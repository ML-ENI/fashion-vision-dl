"""CNN pequeña desde cero: bloques convolución → (BatchNorm) → ReLU → max-pooling."""
import torch
from torch import nn


class CNNClassifier(nn.Module):
    """RGB64 → cuatro bloques convolucionales → Dense → logits de clase.

    Cada bloque mira vecindarios de 3×3 con los mismos pesos en toda la imagen
    y reduce el tamaño a la mitad (64 → 32 → 16 → 8 → 4).
    """

    def __init__(self, image_size=64, conv_channels=(16, 32, 64, 128), hidden_dims=(400,),
                 classes=25, kernel_size=3, padding=1, batchnorm=False):
        super().__init__()
        blocks = []
        channels, size = 3, image_size
        for width in conv_channels:
            layers = [nn.Conv2d(channels, width, kernel_size, padding=padding)]
            if batchnorm:
                layers.append(nn.BatchNorm2d(width))
            layers += [nn.ReLU(), nn.MaxPool2d(2)]
            blocks.append(nn.Sequential(*layers))
            channels, size = width, size // 2
        self.features = nn.Sequential(*blocks)
        head = [nn.Flatten()]
        inputs = channels * size * size
        for width in hidden_dims:
            head += [nn.Linear(inputs, width), nn.ReLU()]
            inputs = width
        head.append(nn.Linear(inputs, classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, images):
        # CrossEntropyLoss recibe logits: no aplicar softmax aquí.
        return self.classifier(self.features(images))

    @torch.inference_mode()
    def feature_maps(self, images):
        """Salida de cada bloque, para visualizar qué responde a cada imagen."""
        maps = []
        x = images
        for block in self.features:
            x = block(x)
            maps.append(x)
        return maps


def multiply_adds(model, image_size=64):
    """Multiplicaciones-sumas por imagen en convoluciones y capas lineales.

    Excluye sesgos, activaciones, pooling, BatchNorm y softmax.
    """
    total = 0
    hooks = []

    def conv_hook(module, _inputs, output):
        nonlocal total
        total += module.weight.numel() * output.shape[2] * output.shape[3]

    def linear_hook(module, _inputs, _output):
        nonlocal total
        total += module.weight.numel()

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        model(torch.zeros(1, 3, image_size, image_size))
    model.train(was_training)
    for hook in hooks:
        hook.remove()
    return total
