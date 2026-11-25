"""Figuras de 04 en dos estilos (ML y ETSIT); sin decisiones de modelado.

Cada figura prepara sus datos una vez y define ``draw(T)``. ``render`` la dibuja
con cada estilo, guarda ambas versiones y devuelve la de ML para el notebook.
"""
from pathlib import Path
from io import BytesIO
import textwrap

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.visualization.style import GRID, INK, MUTED, PAPER, format_number, format_percent
from src.visualization.themes import render


def show_compact(figures, width=850):
    """Acotar el tamaño en el notebook sin reducir los archivos de exportación."""
    from IPython.display import Image as DisplayImage, display
    if not isinstance(figures, (list, tuple)):
        figures = [figures]
    for fig in figures:
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=110, bbox_inches='tight', facecolor=PAPER)
        display(DisplayImage(data=buffer.getvalue(), width=width))
        plt.close(fig)  # Evita otra salida automática de tamaño completo.


def show_code(source, title):
    """El código completo sigue disponible sin ocupar toda la lectura principal."""
    from html import escape
    from IPython.display import HTML, display
    display(HTML(
        f'<details style="max-width:850px;margin:12px 0;border-top:1px solid {GRID};padding:10px">'
        f'<summary style="color:{INK};cursor:pointer">{escape(title)}</summary>'
        '<pre style="max-height:320px;overflow:auto;font-size:13px;line-height:1.5;white-space:pre">'
        + escape(source) + '</pre></details>'
    ))


def run_with_progress(action, log_path):
    """Mantener visible el progreso sin cientos de líneas en el cuaderno."""
    import contextlib
    from html import escape
    from IPython.display import HTML, display
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = display(HTML(f'<p style="color:{MUTED}">Entrenamiento en curso…</p>'), display_id=True)
    with log_path.open('w', encoding='utf-8') as log:
        class Stream:
            buffer = ''
            lines = []
            def write(self, text):
                log.write(text)
                log.flush()
                self.buffer += text
                if '\n' in self.buffer:
                    self.lines = (self.lines + self.buffer.splitlines())[-4:]
                    self.buffer = ''
                    handle.update(HTML('<pre style="font-size:13px;line-height:1.5;max-width:850px;white-space:pre-wrap">'
                                       + escape('\n'.join(self.lines)) + '</pre>'))
                return len(text)
            def flush(self):
                log.flush()
        with contextlib.redirect_stdout(Stream()):
            result = action()
    return result


def save(fig, root, name, note, T):
    folder = Path(root) / 'results/figures' / T.name / '04_dense'
    folder.mkdir(parents=True, exist_ok=True)
    # Como add_title en ML: el ajuste automático reserva sitio para la nota.
    if fig.get_layout_engine() is None:
        fig.set_layout_engine('constrained')
    if fig._suptitle is not None:
        fig._suptitle.set_text(textwrap.fill(fig._suptitle.get_text(), max(45, int(fig.get_figwidth()*T.title_chars))))
    T.style.add_source(fig, textwrap.fill(note, int(fig.get_figwidth()*T.source_chars)))
    fig.savefig(folder / (name+'.png'), dpi=300, bbox_inches='tight')
    fig.savefig(folder / (name+'.svg'), bbox_inches='tight')
    return fig


def both(context, draw, name, note):
    """Guardar la figura en los dos estilos y devolver la versión de ML."""
    return render(draw, lambda fig, T: save(fig, context['root'], name, note, T))


def input_figure(context, raw):
    row = context['rows']['train'].iloc[0]
    with Image.open(context['processed']/row.image_path) as image:
        stored = image.copy()
    small = raw['train'][0].transpose(1, 2, 0)

    def draw(T):
        fig, axes = plt.subplots(1, 2, figsize=T.pick((10, 5), (12, 6)))
        axes[0].imshow(stored)
        axes[1].imshow(small)
        for ax, title in zip(axes, ['Stored: RGB224', 'Dense input: RGB64']):
            ax.axis('off')
            ax.set_title(title, **T.pick({}, {'fontsize': 21}))
        fig.suptitle('The same garment with a smaller input')
        return fig
    return both(context, draw, '01_input',
                f'Source: first train article {row.article_id}, {row.label}. The prepared framing is kept; test is not opened.')


def curves_figure(context, directory):
    seeds = context['config']['training']['seeds']
    histories = {seed: pd.read_csv(directory/f'baseline_seed{seed}_history.csv') for seed in seeds}

    def draw(T):
        fig, axes = plt.subplots(1, 2, figsize=T.pick((13, 5.5), (19, 7)))
        for seed, history in histories.items():
            for split, label in [('train', 'Train'), ('validation', 'Validation')]:
                for ax, metric in zip(axes, ['loss', 'macro_f1']):
                    ax.plot(history.epoch, history[split+'_'+metric], color=T.split[split],
                            alpha=.75, linestyle={42: '-', 123: '--', 2026: ':'}[seed],
                            label=label if seed == 42 else None)
        axes[0].set(ylabel='Cross-entropy', xlabel='Epoch', title='Loss: lower is better')
        axes[1].set(ylabel='Macro-F1', xlabel='Epoch', title='Macro-F1: higher is better', ylim=(0, 1))
        for ax in axes:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend(loc='best')
        fig.suptitle('How the Dense network learns without dropout')
        return fig
    return both(context, draw, '02_learning_curves',
                'Source: local training. One line per seed: 42 solid, 123 dashed, 2026 dotted. All curves are measured in evaluation mode, without dropout.')


def dropout_figure(context, records):
    pairs = {}
    for seed in context['config']['training']['seeds']:
        row = records.loc[records.seed.eq(seed)].set_index('variant')
        pairs[seed] = row.loc[['baseline', 'dropout_030'], 'validation_macro_f1'].to_numpy()

    def draw(T):
        fig, ax = plt.subplots(figsize=T.pick((9, 5.6), (11, 7)))
        for seed, values in pairs.items():
            ax.plot([0, 1], values, color=T.primary, marker='o', alpha=.8,
                    linestyle={42:'-',123:'--',2026:':'}[seed], label=f'Seed {seed}')
        ax.set_xticks([0, 1], ['No dropout', 'Dropout 0,30'])
        ax.set(ylabel='Validation macro-F1', ylim=(0, 1))
        ax.legend(loc='best')
        fig.suptitle('A single change: switching off neurons during training')
        return fig
    return both(context, draw, '03_dropout_pairs',
                'Source: three pairs with the same data, per-seed initialisation and minibatch order. Each point uses the checkpoint selected on validation; test held out.')


def confusion_figure(context, cm):
    names = context['names']
    totals = cm.sum(1, keepdims=True)
    proportions = np.divide(cm, totals, out=np.zeros_like(cm, dtype=float), where=totals > 0)
    figures = []
    for values, title, vmax, suffix in zip([cm, proportions], ['Counts', 'Proportion per true class'], [None, 1], ['counts', 'proportions']):
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
            fig.suptitle(f'Dense network confusion · {title}', color=T.title_accent)
            return fig
        figures.append(both(context, draw, f'04_confusion_{suffix}',
                            'Source: validation, baseline Dense, seed 42. Diagonal = correct predictions. Test held out.'))
    return figures


def per_class_figure(context, dense_report, colour_report):
    names = context['names']
    dense = np.array([dense_report[name]['f1-score'] for name in names])
    colour = colour_report.loc[names, 'f1-score'].to_numpy()

    def draw(T):
        fig, ax = plt.subplots(figsize=T.pick((13, max(7, .3*len(names)+1.5)), (19, max(10, .43*len(names)+2))))
        y = np.arange(len(names))
        ax.barh(y-.18, colour, height=.34, color=T.reference, label='Colour logistic regression (03)')
        ax.barh(y+.18, dense, height=.34, color=T.primary, label='Dense (seed 42)')
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set(xlabel='Per-class validation F1', xlim=(0, 1))
        # Fuera de los datos: no suponemos que una clase deje una esquina libre.
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
        fig.suptitle('The overall improvement can vary across classes')
        return fig
    return both(context, draw, '05_per_class',
                'Source: same validation images and labels. One pre-set Dense seed; training variability is reported separately. It does not isolate the effect of the architecture alone.')


def errors_figure(context, predictions, count=8):
    rows = context['rows']['validation'].copy()
    rows['prediction'] = predictions.predicted_label_id.to_numpy()
    rows['confidence'] = predictions.confidence.to_numpy()
    rows['label_id'] = rows.label_id.astype(int)
    errors = rows.loc[rows.prediction.ne(rows.label_id)].sort_values(['confidence', 'article_id'], ascending=[False, True]).head(count)
    photos = []
    for row in errors.itertuples():
        with Image.open(context['processed']/row.image_path) as image:
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
        fig.suptitle('Most confident errors: they also need a look')
        return fig
    return both(context, draw, '06_confident_errors',
                'Source: validation, Dense baseline seed 42. Automatic selection: the eight errors with the highest softmax. Not a representative sample nor a confirmed label review.')


def weight_figure(context, model):
    first = model.network[1]
    weights = first.weight.detach().numpy().reshape(-1, 3, 64, 64)[:6].mean(1)
    bound = abs(weights).max()

    def draw(T):
        fig, axes = plt.subplots(2, 3, figsize=T.pick((13, 8), (16, 10)))
        for index, (ax, values) in enumerate(zip(axes.flat, weights)):
            image = ax.imshow(values, cmap=T.diverging, vmin=-bound, vmax=bound)
            ax.set_title(f'Neuron {index}', **T.pick({}, {'fontsize': 20}))
            ax.axis('off')
        fig.colorbar(image, ax=list(axes.flat), shrink=.65, label='Mean weight over the three channels')
        fig.suptitle('The first Dense layer assigns a weight to every position')
        return fig
    return both(context, draw, '07_first_layer_weights',
                'Source: first six neurons, baseline seed 42. Mean RGB weight per position, common scale centred at zero. These are neither photographs nor causal importance maps.')


@torch.inference_mode()
def activation_figure(context, model, tensors):
    # Submuestra fija por orden: diagnóstico descriptivo, no ajuste de decisiones.
    images = tensors['train'][:512]
    activation = model.network[2](model.network[1](images.flatten(1)))
    fraction = (activation > 0).float().mean(0).numpy()

    def draw(T):
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.bar(range(len(fraction)), fraction, color=T.primary)
        ax.set(xlabel='First-layer neuron index', ylabel='Fraction of inputs\nwith ReLU > 0', ylim=(0, 1))
        fig.suptitle('Which neurons activate on a fixed train sample')
        return fig
    return both(context, draw, '08_activations',
                'Source: first 512 train images, baseline seed 42. Activation after ReLU in evaluation mode. Do not generalise inactivity in this sample to the whole population.')
