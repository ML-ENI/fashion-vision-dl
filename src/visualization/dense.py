"""Figuras de 04 con el estilo del proyecto ML; sin decisiones de modelado."""
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.visualization.style import BLUE, ORANGE, SPLIT_COLORS, add_source, apply_style, format_number, format_percent


def save(fig, root, name, note):
    folder = Path(root) / 'results/figures/04_dense'
    folder.mkdir(parents=True, exist_ok=True)
    if fig._suptitle is not None:
        fig._suptitle.set_text(textwrap.fill(fig._suptitle.get_text(), max(45, int(fig.get_figwidth()*5))))
    add_source(fig, textwrap.fill(note, int(fig.get_figwidth()*10)))
    fig.savefig(folder / (name+'.png'), dpi=300, bbox_inches='tight')
    fig.savefig(folder / (name+'.svg'), bbox_inches='tight')
    return fig


def input_figure(context, raw):
    row = context['rows']['train'].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    with Image.open(context['processed']/row.image_path) as image:
        axes[0].imshow(image)
    axes[1].imshow(raw['train'][0].transpose(1, 2, 0))
    for ax, title in zip(axes, ['Almacenada: RGB224', 'Entrada Dense: RGB64']):
        ax.axis('off')
        ax.set_title(title, fontsize=21)
    fig.suptitle('La misma prenda con una entrada más pequeña')
    return save(fig, context['root'], '01_input',
                f'Fuente: primer artículo de train {row.article_id}, {row.label}. Se conserva el encuadre preparado; no se abre test.')


def curves_figure(context, directory):
    fig, axes = plt.subplots(1, 2, figsize=(19, 7))
    for seed in context['config']['training']['seeds']:
        history = pd.read_csv(directory/f'baseline_seed{seed}_history.csv')
        for split, label in [('train', 'Train'), ('validation', 'Validation')]:
            for ax, metric in zip(axes, ['loss', 'macro_f1']):
                ax.plot(history.epoch, history[split+'_'+metric], color=SPLIT_COLORS[split],
                        alpha=.75, linestyle={42: '-', 123: '--', 2026: ':'}[seed],
                        label=label if seed == 42 else None)
    axes[0].set(ylabel='Entropía cruzada', xlabel='Época', title='Pérdida: menor es mejor')
    axes[1].set(ylabel='Macro-F1', xlabel='Época', title='Macro-F1: mayor es mejor', ylim=(0, 1))
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc='best')
    fig.suptitle('Cómo aprende la Dense sin dropout')
    return save(fig, context['root'], '02_learning_curves',
                'Fuente: entrenamiento local. Una línea por semilla: 42 continua, 123 discontinua, 2026 punteada. Todas las curvas se miden en modo evaluación, sin dropout.')


def dropout_figure(context, records):
    fig, ax = plt.subplots(figsize=(11, 7))
    for seed in context['config']['training']['seeds']:
        row = records.loc[records.seed.eq(seed)].set_index('variant')
        values = row.loc[['baseline', 'dropout_030'], 'validation_macro_f1'].to_numpy()
        ax.plot([0, 1], values, color=BLUE, marker='o', alpha=.8,
                linestyle={42:'-',123:'--',2026:':'}[seed], label=f'Semilla {seed}')
    ax.set_xticks([0, 1], ['Sin dropout', 'Dropout 0,30'])
    ax.set(ylabel='Macro-F1 en validation', ylim=(0, 1))
    ax.legend(loc='best')
    fig.suptitle('Una sola modificación: apagar neuronas al entrenar')
    return save(fig, context['root'], '03_dropout_pairs',
                'Fuente: tres pares con los mismos datos, inicialización por semilla y orden de minibatches. Cada punto usa el checkpoint elegido con validation; test reservado.')


def confusion_figure(context, cm):
    names = context['names']
    fig, axes = plt.subplots(1, 2, figsize=(25, 11))
    proportions = cm / cm.sum(1, keepdims=True)
    for ax, values, title, vmax in zip(axes, [cm, proportions], ['Recuentos', 'Proporción por clase real'], [None, 1]):
        im = ax.imshow(values, cmap='Blues', vmin=0, vmax=vmax)
        ax.set_xticks(range(len(names)), names, rotation=65, ha='right')
        ax.set_yticks(range(len(names)), names)
        ax.set(xlabel='Clase predicha', ylabel='Clase real', title=title)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=.65)
    fig.suptitle('Qué clases confunde la Dense de referencia')
    return save(fig, context['root'], '04_confusion',
                'Fuente: validation, baseline Dense, semilla 42 prefijada. Diagonal = aciertos. Cada fila de la matriz derecha suma uno; no son resultados de test.')


def per_class_figure(context, dense_report, colour_report):
    names = context['names']
    dense = np.array([dense_report[name]['f1-score'] for name in names])
    colour = colour_report.loc[names, 'f1-score'].to_numpy()
    fig, ax = plt.subplots(figsize=(13, 10))
    y = np.arange(len(names))
    ax.barh(y-.18, colour, height=.34, color=ORANGE, label='Logística de color (03)')
    ax.barh(y+.18, dense, height=.34, color=BLUE, label='Dense (semilla 42)')
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set(xlabel='F1 por clase en validation', xlim=(0, 1))
    # Las dos primeras clases dejan libre esta esquina; abajo taparía Vest top.
    ax.legend(loc='upper right')
    fig.suptitle('La mejora global puede variar entre clases')
    return save(fig, context['root'], '05_per_class',
                'Fuente: mismas imágenes y etiquetas de validation. Una semilla Dense prefijada; la variabilidad de entrenamiento se informa aparte. No aísla el efecto exclusivo de la arquitectura.')


def errors_figure(context, predictions, count=8):
    rows = context['rows']['validation'].copy()
    rows['prediction'] = predictions.predicted_label_id.to_numpy()
    rows['confidence'] = predictions.confidence.to_numpy()
    rows['label_id'] = rows.label_id.astype(int)
    errors = rows.loc[rows.prediction.ne(rows.label_id)].sort_values(['confidence', 'article_id'], ascending=[False, True]).head(count)
    fig, axes = plt.subplots(2, 4, figsize=(21, 12))
    for ax, row in zip(axes.flat, errors.itertuples()):
        with Image.open(context['processed']/row.image_path) as image:
            ax.imshow(image)
        label = context['names'][row.prediction]
        ax.set_title(f'Real: {row.label}\nPred.: {label}\nSoftmax: {format_percent(row.confidence)}', fontsize=17, fontweight='normal')
        ax.axis('off')
    for ax in axes.flat[len(errors):]:
        ax.axis('off')
    fig.suptitle('Errores con mayor confianza: también hay que mirarlos')
    return save(fig, context['root'], '06_confident_errors',
                'Fuente: validation, Dense baseline semilla 42. Selección automática: los ocho errores de mayor softmax. No es una muestra representativa ni una revisión de etiquetas confirmada.')


def weight_figure(context, model):
    first = model.network[1]
    weights = first.weight.detach().numpy().reshape(-1, 3, 64, 64)[:6].mean(1)
    bound = abs(weights).max()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for index, (ax, values) in enumerate(zip(axes.flat, weights)):
        image = ax.imshow(values, cmap='RdBu_r', vmin=-bound, vmax=bound)
        ax.set_title(f'Neurona {index}', fontsize=20)
        ax.axis('off')
    fig.colorbar(image, ax=list(axes.flat), shrink=.65, label='Peso medio de los tres canales')
    fig.suptitle('La primera Dense asigna un peso a cada posición')
    return save(fig, context['root'], '07_first_layer_weights',
                'Fuente: seis primeras neuronas, baseline semilla 42. Media de pesos RGB por posición, escala común centrada en cero. No son fotografías ni mapas causales de importancia.')


@torch.inference_mode()
def activation_figure(context, model, tensors):
    # Submuestra fija por orden: diagnóstico descriptivo, no ajuste de decisiones.
    images = tensors['train'][:512]
    activation = model.network[2](model.network[1](images.flatten(1)))
    fraction = (activation > 0).float().mean(0).numpy()
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(range(len(fraction)), fraction, color=BLUE)
    ax.set(xlabel='Índice de neurona de la primera capa', ylabel='Fracción de entradas\ncon ReLU > 0', ylim=(0, 1))
    fig.suptitle('Qué neuronas se activan en una muestra fija de train')
    return save(fig, context['root'], '08_activations',
                'Fuente: primeras 512 imágenes de train, baseline semilla 42. Activación tras ReLU en modo evaluación. No generalizar inactividad de esta muestra a toda la población.')
