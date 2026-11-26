"""Imagen como secuencia: RGB64 → 64 pasos de 192 valores → RNN, LSTM o GRU → clase."""
import torch
from torch import nn

CELLS = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
GATES = {'rnn': 1, 'lstm': 4, 'gru': 3}


def image_to_sequence(images, orientation='columns'):
    """(B, 3, 64, 64) → (B, 64, 192).

    columns: un paso por columna, de izquierda a derecha; cada paso son las 64
    filas × 3 canales de esa columna. rows: un paso por fila, de arriba abajo.
    """
    if orientation == 'columns':
        steps = images.permute(0, 3, 1, 2)   # (B, ancho, canales, alto)
    elif orientation == 'rows':
        steps = images.permute(0, 2, 1, 3)   # (B, alto, canales, ancho)
    else:
        raise ValueError(f'Orientación desconocida: {orientation}')
    return steps.reshape(images.shape[0], steps.shape[1], -1)


class SequenceClassifier(nn.Module):
    """Una capa recurrente; la clase sale del último estado o de una atención sobre todos."""

    def __init__(self, cell='lstm', input_size=192, hidden_size=128, classes=25,
                 orientation='columns', readout='last'):
        super().__init__()
        self.cell, self.orientation, self.readout = cell, orientation, readout
        self.recurrent = CELLS[cell](input_size, hidden_size, batch_first=True)
        if readout == 'attention':
            # Atención aditiva: una puntuación por paso a partir de su estado.
            self.attention = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))
        elif readout != 'last':
            raise ValueError(f'Lectura desconocida: {readout}')
        self.output = nn.Linear(hidden_size, classes)

    def states(self, images):
        states, _ = self.recurrent(image_to_sequence(images, self.orientation))
        return states  # (B, pasos, oculto)

    def attention_weights(self, states):
        return torch.softmax(self.attention(states).squeeze(-1), dim=1)  # (B, pasos), suman 1

    def forward(self, images):
        states = self.states(images)
        if self.readout == 'last':
            summary = states[:, -1]
        else:
            summary = (self.attention_weights(states).unsqueeze(-1) * states).sum(1)
        # CrossEntropyLoss recibe logits: no aplicar softmax aquí.
        return self.output(summary)


def multiply_adds(model, steps=64):
    """Multiplicaciones-sumas por imagen: celda recurrente en cada paso, atención y capas lineales.

    Excluye sesgos, activaciones y softmax.
    """
    r = model.recurrent
    per_step = GATES[model.cell] * (r.input_size * r.hidden_size + r.hidden_size * r.hidden_size)
    total = per_step * steps + model.output.in_features * model.output.out_features
    if model.readout == 'attention':
        total += steps * sum(layer.in_features * layer.out_features
                             for layer in model.attention if isinstance(layer, nn.Linear))
    return total
