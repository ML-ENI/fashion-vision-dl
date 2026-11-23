"""Tablas legibles para los notebooks: presentación sin modificar los datos."""
from __future__ import annotations

import numbers
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from src.visualization.style import BLUE, ORANGE, INK, GRID, format_number


LABELS = {
    'variant': 'Modelo', 'baseline': 'Sin dropout', 'dropout_030': 'Dropout 0,30',
    'seed': 'Semilla', 'selected_epoch': 'Época elegida', 'epochs_run': 'Épocas realizadas',
    'train_macro_f1': 'Macro-F1 · train', 'validation_macro_f1': 'Macro-F1 · validación',
    'validation_accuracy': 'Accuracy · validación', 'train_seconds': 'Entrenamiento (s)',
    'inference_ms_per_image': 'Inferencia (ms/imagen)', 'parameters': 'Parámetros',
    'approx_linear_flops_per_image': 'FLOPs aprox./imagen',
    'macro_F1_media': 'Macro-F1 medio', 'macro_F1_std': 'Desv. típica F1',
    'accuracy_media': 'Accuracy media', 'accuracy_std': 'Desv. típica accuracy',
    'tiempo_total_s': 'Tiempo total (s)', 'semillas': 'Semillas',
    'diferencia_dropout_menos_baseline': 'Diferencia de macro-F1',
    'macro_F1': 'Macro-F1', 'accuracy': 'Accuracy', 'precision': 'Precisión',
    'recall': 'Recall', 'f1-score': 'F1', 'support': 'Imágenes',
    'label': 'Clase', 'split': 'Partición', 'product_type_name': 'Tipo de prenda',
    'garment_group_name': 'Grupo de prendas', 'index_group_name': 'Segmento comercial',
    'articulos': 'Artículos', 'sin_imagen': 'Sin imagen', 'article_id': 'ID del artículo',
    'reason': 'Motivo', 'count': 'Número', 'mean': 'Media', 'std': 'Desv. típica',
    'min': 'Mínimo', 'max': 'Máximo', 'median': 'Mediana',
    'variacion_intensidad': 'Variación de intensidad', 'esquinas_contenido': 'Brillo de las esquinas',
    'brillo': 'Brillo', 'Proporción de la clase real': 'Fracción de la clase real',
    'accuracy_IC95_inf': 'Accuracy · IC95 inferior', 'accuracy_IC95_sup': 'Accuracy · IC95 superior',
    'F1_IC95_inf': 'F1 · IC95 inferior', 'F1_IC95_sup': 'F1 · IC95 superior',
    'ajuste_s': 'Ajuste (s)', 'prediccion_ms_imagen': 'Predicción (ms/imagen)',
}


def label(value):
    return LABELS.get(value, value) if isinstance(value, str) else value


def table_styles():
    return [
        {'selector': '', 'props': [('border-collapse', 'collapse'), ('border-spacing', '0'),
            ('font-family', '"Latin Modern Roman", "LM Roman 10", Georgia, serif'),
            ('font-size', '15px'), ('line-height', '1.45'), ('color', INK),
            ('background-color', '#FFFFFF'), ('margin', '12px 0 22px'), ('max-width', '100%')]},
        {'selector': 'th, td', 'props': [('padding', '9px 13px'), ('border', 'none'),
            ('border-bottom', f'1px solid {GRID}'), ('vertical-align', 'top'),
            ('text-align', 'left'), ('max-width', '28rem'), ('overflow-wrap', 'anywhere')]},
        {'selector': 'thead th', 'props': [('background-color', BLUE), ('color', '#FFFFFF'),
            ('font-weight', '600'), ('text-align', 'left'), ('max-width', '12rem'),
            ('white-space', 'normal')]},
        {'selector': 'thead tr:last-child th', 'props': [('border-bottom', f'3px solid {ORANGE}')]},
        {'selector': 'tbody th', 'props': [('font-weight', '600'), ('white-space', 'nowrap')]},
        {'selector': 'tbody tr:nth-child(even) th, tbody tr:nth-child(even) td',
            'props': [('background-color', '#F1F5FA')]},
        {'selector': 'tbody tr:nth-child(odd) th, tbody tr:nth-child(odd) td',
            'props': [('background-color', '#FFFFFF')]},
        {'selector': 'td', 'props': [('font-variant-numeric', 'tabular-nums')]},
    ]


def table_number(value, digits=3, integer=False):
    if value is None or (isinstance(value, numbers.Real) and not np.isfinite(value)):
        return '—'
    if isinstance(value, (bool, np.bool_)):
        return 'Sí' if value else 'No'
    if not isinstance(value, numbers.Number):
        return str(label(value))
    if integer or isinstance(value, numbers.Integral):
        return format_number(value, 0)
    if 0 < abs(value) < 10**(-max(digits, 1)):
        return f'{value:.2e}'.replace('.', ',')
    return format_number(value, digits)


def report_table(frame, digits=3):
    """Enteros sin decimales, métricas redondeadas y cabeceras comprensibles."""
    if frame.__class__.__name__ == 'Styler':
        frame = frame.data
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    data = frame.copy()
    data.index.names = [label(name) for name in data.index.names]
    data.columns.names = [label(name) for name in data.columns.names]
    if data.index.nlevels == 1 and (data.index.name is not None or not is_integer_dtype(data.index.dtype)):
        # El índice significativo pasa a una columna normal: evita una segunda
        # fila de cabecera casi vacía y hace explícitos Clase, Modelo o Semilla.
        name = data.index.name or 'Referencia'
        if name in data.columns:
            name += ' (índice)'
        data = data.rename_axis(name).reset_index()
    style = data.style.set_table_styles(table_styles())
    formats = {}
    numeric = []
    for column in data.columns:
        integer = is_integer_dtype(data[column]) or column in {'support', 'semillas'}
        formats[column] = lambda value, integer=integer: table_number(value, digits, integer)
        if column in {'seed', 'Semilla'}:
            formats[column] = lambda value: str(int(value))
        if is_numeric_dtype(data[column]):
            numeric.append(column)
    style = style.format(formats, escape='html', na_rep='—')
    style = style.format_index(lambda x: label(x), axis=1, escape='html')
    index_format = (lambda x: str(int(x))) if data.index.name == 'Semilla' else (lambda x: table_number(x, 0))
    style = style.format_index(index_format, axis=0, escape='html')
    if numeric:
        style = style.set_properties(subset=numeric, **{'text-align': 'right !important', 'white-space': 'nowrap'})
    # Quitar índices automáticos; conservar clase, semilla y demás índices con significado.
    if data.index.nlevels == 1 and data.index.name is None and is_integer_dtype(data.index.dtype):
        style = style.hide(axis='index')
    return style
