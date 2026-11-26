"""Figuras de 05 en dos estilos (ML y ETSIT); textos de las figuras en inglés.

Cada figura prepara sus datos una vez y define ``draw(T)``; ``both`` la guarda
en ``results/figures/<estilo>/05_cnn`` y devuelve la versión ML para el notebook.
"""
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch

from src.visualization.dense import both as _both
from src.visualization.style import format_percent

FOLDER = '05_cnn'
LINE_STYLES = {42: '-', 123: '--', 2026: ':'}


def both(context, draw, name, note, folder=FOLDER):
    return _both(context, draw, name, note, folder=folder)


def convolution_figure(context, raw):
    """Un filtro fijo 3×3 (bordes verticales) sobre una foto real de train."""
    row = context['rows']['train'].iloc[0]
    image = raw['train'][0].transpose(1, 2, 0).astype(np.float32) / 255
    grey = image.mean(axis=2)
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    padded = np.pad(grey, 1, mode='edge')
    response = np.zeros_like(grey)
    for i in range(3):
        for j in range(3):
            response += kernel[i, j] * padded[i:i + grey.shape[0], j:j + grey.shape[1]]
    bound = float(np.abs(response).max()) or 1.0

    def draw(T):
        fig, axes = plt.subplots(1, 3, figsize=T.pick((12, 4.4), (16, 6)),
                                 gridspec_kw={'width_ratios': [1, .55, 1]})
        axes[0].imshow(image)
        axes[0].set_title('Input: RGB64')
        axes[1].imshow(kernel, cmap=T.diverging, vmin=-2, vmax=2)
        for (i, j), value in np.ndenumerate(kernel):
            axes[1].text(j, i, f'{value:g}', ha='center', va='center', color=T.ink)
        axes[1].set_title('3×3 filter')
        axes[2].imshow(response, cmap=T.diverging, vmin=-bound, vmax=bound)
        axes[2].set_title('Feature map')
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        fig.suptitle('One filter, the same weights at every position')
        return fig
    return both(context, draw, '01_convolution',
                f'Source: first train article {row.article_id}, {row.label}, in grey levels. Fixed vertical-edge '
                'filter for illustration; the CNN learns its own filters.')


def curves_figure(context, directory, variant='cnn', title='How the CNN learns', folder=FOLDER):
    seeds = context['config']['training']['seeds']
    histories = {seed: pd.read_csv(directory / f'{variant}_seed{seed}_history.csv') for seed in seeds}

    def draw(T):
        fig, axes = plt.subplots(1, 2, figsize=T.pick((13, 5.5), (19, 7)))
        for seed, history in histories.items():
            for split, label in [('train', 'Train'), ('validation', 'Validation')]:
                for ax, metric in zip(axes, ['loss', 'macro_f1']):
                    ax.plot(history.epoch, history[split + '_' + metric], color=T.split[split], alpha=.75,
                            linestyle=LINE_STYLES.get(seed, '-'), label=label if seed == seeds[0] else None)
        axes[0].set(ylabel='Cross-entropy', xlabel='Epoch', title='Loss: lower is better')
        axes[1].set(ylabel='Macro-F1', xlabel='Epoch', title='Macro-F1: higher is better', ylim=(0, 1))
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend(loc='best')
        fig.suptitle(title)
        return fig
    return both(context, draw, f'02_learning_curves_{variant}',
                'Source: local training. One line per seed: 42 solid, 123 dashed, 2026 dotted. '
                'Curves measured in evaluation mode.', folder)


def paired_figure(context, left, right, labels, name, title, note, folder=FOLDER):
    """Una línea por semilla entre dos modelos: macro-F1 de validation emparejado."""
    seeds = context['config']['training']['seeds']
    pairs = {seed: [left[seed], right[seed]] for seed in seeds}

    def draw(T):
        fig, ax = plt.subplots(figsize=T.pick((9, 5.6), (11, 7)))
        for seed, values in pairs.items():
            ax.plot([0, 1], values, color=T.primary, marker='o', alpha=.8,
                    linestyle=LINE_STYLES.get(seed, '-'), label=f'Seed {seed}')
        ax.set_xticks([0, 1], labels)
        ax.set(ylabel='Validation macro-F1', ylim=(0, 1))
        ax.legend(loc='best')
        fig.suptitle(title)
        return fig
    return both(context, draw, name, note, folder)


def confusion_figure(context, cm, model_name='CNN', prefix='04_confusion', folder=FOLDER):
    names = context['names']
    totals = cm.sum(1, keepdims=True)
    proportions = np.divide(cm, totals, out=np.zeros_like(cm, dtype=float), where=totals > 0)
    figures = []
    for values, title, vmax, suffix in zip([cm, proportions], ['Counts', 'Proportion per true class'],
                                           [None, 1], ['counts', 'proportions']):
        def draw(T, values=values, title=title, vmax=vmax):
            fig, ax = plt.subplots(figsize=T.pick((10, 9), (12, 11)))
            im = ax.imshow(values, cmap=T.sequential, vmin=0, vmax=vmax)
            ax.set_xticks(range(len(names)), names, rotation=90, ha='center', fontsize=T.pick(8, 11))
            ax.set_yticks(range(len(names)), names, fontsize=T.pick(8, 11))
            ax.set_xlabel('Predicted class', **T.pick({}, {'fontsize': 14}))
            ax.set_ylabel('True class', **T.pick({}, {'fontsize': 14}))
            ax.grid(False)
            bar = fig.colorbar(im, ax=ax, shrink=.75)
            bar.ax.tick_params(labelsize=11)
            fig.suptitle(f'{model_name} confusion · {title}', color=T.title_accent)
            return fig
        figures.append(both(context, draw, f'{prefix}_{suffix}',
                            f'Source: validation, {model_name}, seed 42. Diagonal = correct predictions. Test held out.',
                            folder))
    return figures


def per_class_figure(context, dense_report, cnn_report, labels=('Dense (04, seed 42)', 'CNN (seed 42)'),
                     title='Where the CNN gains or loses against the Dense network', folder=FOLDER):
    names = context['names']
    dense = np.array([dense_report[name]['f1-score'] for name in names])
    cnn = np.array([cnn_report[name]['f1-score'] for name in names])

    def draw(T):
        fig, ax = plt.subplots(figsize=T.pick((13, max(7, .3 * len(names) + 1.5)), (19, max(10, .43 * len(names) + 2))))
        y = np.arange(len(names))
        ax.barh(y - .18, dense, height=.34, color=T.reference, label=labels[0])
        ax.barh(y + .18, cnn, height=.34, color=T.primary, label=labels[1])
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set(xlabel='Per-class validation F1', xlim=(0, 1))
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
        fig.suptitle(title)
        return fig
    return both(context, draw, '05_per_class',
                'Source: same validation images and labels; seed 42 of each model. Classes without validation photos '
                'show F1 = 0.', folder)


def errors_figure(context, predictions, count=8, model_name='CNN', folder=FOLDER):
    rows = context['rows']['validation'].copy()
    rows['prediction'] = predictions.predicted_label_id.to_numpy()
    rows['confidence'] = predictions.confidence.to_numpy()
    rows['label_id'] = rows.label_id.astype(int)
    errors = rows.loc[rows.prediction.ne(rows.label_id)].sort_values(
        ['confidence', 'article_id'], ascending=[False, True]).head(count)
    photos = []
    for row in errors.itertuples():
        with Image.open(context['processed'] / row.image_path) as image:
            photos.append(image.copy())

    def draw(T):
        fig, axes = plt.subplots(2, 4, figsize=T.pick((10, 6), (12, 7)))
        for ax, row, photo in zip(axes.flat, errors.itertuples(), photos):
            ax.imshow(photo)
            label = context['names'][row.prediction]
            ax.set_title(f'True: {row.label}\nPred.: {label}\nSoftmax: {format_percent(row.confidence)}',
                         fontsize=T.pick(8, 10), fontweight='normal')
            ax.axis('off')
        for ax in axes.flat[len(errors):]:
            ax.axis('off')
        fig.suptitle(f'Most confident {model_name} errors')
        return fig
    return both(context, draw, '06_confident_errors',
                f'Source: validation, {model_name} seed 42. Automatic selection: the eight errors with the highest '
                'softmax. Not a representative sample nor a confirmed label review.', folder)


def filters_figure(context, model):
    """Los 16 filtros 3×3 de la primera capa, cada uno escalado a su propio rango."""
    weights = model.features[0][0].weight.detach().numpy().transpose(0, 2, 3, 1)
    low = weights.min(axis=(1, 2, 3), keepdims=True)
    high = weights.max(axis=(1, 2, 3), keepdims=True)
    scaled = (weights - low) / np.where(high > low, high - low, 1)

    def draw(T):
        fig, axes = plt.subplots(2, 8, figsize=T.pick((12, 3.8), (16, 5)))
        for index, (ax, values) in enumerate(zip(axes.flat, scaled)):
            ax.imshow(values, interpolation='nearest')
            ax.set_title(f'Filter {index}', fontsize=T.pick(8, 12))
            ax.axis('off')
        fig.suptitle('First-layer filters learned by the CNN')
        return fig
    return both(context, draw, '07_first_layer_filters',
                'Source: CNN seed 42. Each 3×3 RGB filter scaled to its own range. These are weights, not '
                'explanations of a prediction.')


def feature_maps_figure(context, model, tensors, raw, maps_per_block=6):
    row = context['rows']['train'].iloc[0]
    image = raw['train'][0].transpose(1, 2, 0)
    maps = [block[0, :maps_per_block].numpy() for block in model.feature_maps(tensors['train'][:1])]

    def draw(T):
        fig, axes = plt.subplots(len(maps), maps_per_block + 1, figsize=T.pick((12, 8.5), (16, 11)))
        for b, block in enumerate(maps):
            if b == 0:
                axes[b, 0].imshow(image)
                axes[b, 0].set_title('Input', fontsize=T.pick(9, 12))
            for c in range(maps_per_block):
                axes[b, c + 1].imshow(block[c], cmap=T.sequential)
                axes[b, c + 1].set_title(f'Block {b + 1} · map {c}\n{block.shape[1]}×{block.shape[2]}',
                                         fontsize=T.pick(8, 11))
        for ax in axes.flat:
            ax.axis('off')
        fig.suptitle('What each block responds to in one train photo')
        return fig
    return both(context, draw, '08_feature_maps',
                f'Source: first train article {row.article_id}, {row.label}; CNN seed 42 after ReLU and pooling. '
                'Bright = strong response. Not a causal explanation.')


def cost_frame(records):
    """Tabla común de coste: parámetros, operaciones, tiempo y memoria."""
    return records[['variant', 'seed', 'parameters', 'train_seconds', 'inference_ms_per_image',
                    'peak_process_rss_bytes']].assign(
        peak_process_rss_bytes=lambda f: f.peak_process_rss_bytes / 1024 ** 2).rename(
        columns={'peak_process_rss_bytes': 'RSS proceso (MiB)'})


def checkpoint_model(context, record, model_from_config, variant):
    """Reabrir un checkpoint si existe en este equipo; None si los pesos no están."""
    from pathlib import PureWindowsPath
    from src.training.dense_experiment import sha256
    path = context['root'] / PureWindowsPath(record['checkpoint']).as_posix()
    if not path.is_file():
        return None, path
    if sha256(path) != record['checkpoint_sha256']:
        raise ValueError('Checkpoint modificado')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = model_from_config(context['config'], variant)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model, path
