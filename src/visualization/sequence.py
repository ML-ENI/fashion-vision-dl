"""Figuras propias de 06 (textos en inglés, estilos ML y ETSIT).

Las figuras comunes (curvas, pares, confusión, F1 por clase, errores) vienen de
``src.visualization.cnn`` con ``folder=FOLDER``.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.visualization.cnn import both as _both

FOLDER = '06_sequence'
MARKERS = {42: 'o', 123: 's', 2026: '^'}


def both(context, draw, name, note):
    return _both(context, draw, name, note, folder=FOLDER)


def sequence_figure(context, raw, tensors):
    """Una foto, sus columnas marcadas y la secuencia 64 × 192 que recibe la red."""
    from src.models.sequence import image_to_sequence
    row = context['rows']['train'].iloc[0]
    image = raw['train'][0].transpose(1, 2, 0)
    sequence = image_to_sequence(tensors['train'][:1], 'columns')[0].numpy().T  # (192, 64)
    bound = float(np.abs(sequence).max()) or 1.0

    def draw(T):
        fig, axes = plt.subplots(1, 2, figsize=T.pick((12, 5), (16, 6.5)), gridspec_kw={'width_ratios': [1, 1.4]})
        axes[0].imshow(image)
        for x in range(0, 65, 8):
            axes[0].axvline(x - .5, color=T.primary, linewidth=.8, alpha=.7)
        axes[0].set_title('Read left to right: 64 columns')
        axes[0].set_xticks([]); axes[0].set_yticks([])
        axes[1].imshow(sequence, aspect='auto', cmap=T.diverging, vmin=-bound, vmax=bound)
        axes[1].set(xlabel='Step (column)', ylabel='Value in the step (64 rows × 3 channels)',
                    title='The sequence: 64 steps × 192 values')
        for ax in axes:
            ax.grid(False)
        fig.suptitle('From a photo to a sequence')
        return fig
    return both(context, draw, '01_image_as_sequence',
                f'Source: first train article {row.article_id}, {row.label}; normalized RGB64. Grid lines every '
                '8 columns for reading only.')


def models_figure(context, table, name, title, note):
    """Macro-F1 de validation por modelo; un marcador por semilla y su media."""
    models = list(table.columns)

    def draw(T):
        fig, ax = plt.subplots(figsize=T.pick((10, 5.6), (13, 7)))
        for seed, values in table.iterrows():
            ax.plot(range(len(models)), values.to_numpy(), color=T.primary, alpha=.5, linewidth=1,
                    marker=MARKERS.get(seed, 'o'), label=f'Seed {seed}')
        ax.scatter(range(len(models)), table.mean().to_numpy(), color=T.reference, s=120, marker='_',
                   linewidths=3, zorder=3, label='Mean')
        ax.set_xticks(range(len(models)), models)
        ax.set(ylabel='Validation macro-F1', ylim=(0, 1))
        ax.legend(loc='best')
        fig.suptitle(title)
        return fig
    return both(context, draw, name, note)


@torch.inference_mode()
def attention_figure(context, model, tensors, count=4):
    """Fotos de validation y el peso de atención que recibe cada una de sus 64 columnas."""
    rows = context['rows']['validation']
    images = tensors['validation'][:count]
    weights = model.attention_weights(model.states(images)).numpy()
    pred = model(images).argmax(1).numpy()
    from PIL import Image
    photos = []
    for row in rows.iloc[:count].itertuples():
        with Image.open(context['processed'] / row.image_path) as image:
            photos.append(np.asarray(image.convert('RGB').resize((64, 64), Image.Resampling.LANCZOS)))

    def draw(T):
        fig, axes = plt.subplots(2, count, figsize=T.pick((12, 6), (16, 8)),
                                 gridspec_kw={'height_ratios': [3, 1]}, sharex='col')
        for k in range(count):
            axes[0, k].imshow(photos[k], extent=(-.5, 63.5, 63.5, -.5))
            axes[0, k].set_title(f'True: {rows.iloc[k].label}\nPred.: {context["names"][pred[k]]}',
                                 fontsize=T.pick(8, 11), fontweight='normal')
            axes[0, k].set_xticks([]); axes[0, k].set_yticks([]); axes[0, k].grid(False)
            axes[1, k].bar(range(64), weights[k], width=1, color=T.primary)
            axes[1, k].axhline(1 / 64, color=T.reference, linestyle='--', linewidth=1)
            axes[1, k].set_xlim(-.5, 63.5)
            axes[1, k].set_xlabel('Column')
        axes[1, 0].set_ylabel('Attention')
        fig.suptitle('Where the attention readout looks')
        return fig
    return both(context, draw, '07_attention_map',
                'Source: first four validation photos (fixed order, not chosen), LSTM with attention, seed 42. '
                'Dashed line = uniform weight 1/64. Weights show where the summary looks, not why the class is predicted.')
