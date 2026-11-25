"""Regla exploratoria de encuadre: píxeles no claros en el borde real.

No utiliza CLIP ni un modelo entrenable. La caché es independiente del umbral.
"""
from __future__ import annotations

import gzip
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, __version__ as pillow_version

RECIPE = 'border-foreground-rgb64-bicubic-cutoff078-band4-v1'


def border_foreground(image, width, height):
    """Retirar el padding conocido de 02 antes de inspeccionar el encuadre."""
    if min(width, height) <= 0:
        raise ValueError('Dimensiones originales no válidas')
    scale = image.width / max(width, height)
    w, h = max(1, round(width*scale)), max(1, round(height*scale))
    left, top = (image.width-w)//2, (image.height-h)//2
    content = image.convert('RGB').crop((left, top, left+w, top+h))
    pixels = np.asarray(content.resize((64, 64), Image.Resampling.BICUBIC), dtype=float)/255
    # Esto identifica píxeles suficientemente alejados de un fondo claro;
    # no es una segmentación semántica de tela.
    foreground = pixels.min(axis=2) < .78
    return float(max(foreground[:4].mean(), foreground[-4:].mean(),
                     foreground[:, :4].mean(), foreground[:, -4:].mean()))


def _digest(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def cache_key(row):
    return f'{row.prepared_sha256}:{int(row.width)}:{int(row.height)}'


def save_cache(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending.gz')
    payload = {'recipe': RECIPE, 'pillow': pillow_version,
               'values': values, 'values_sha256': _digest(values)}
    with gzip.open(temporary, 'wt', encoding='utf-8') as stream:
        json.dump(payload, stream)
    temporary.replace(path)


def geometry_scores(frame, root):
    """Leer cada copia elegible una vez y reutilizar su medida en otras ejecuciones."""
    root = Path(root)
    path = root/'results/metrics/02_zoom_rule/geometry_cache.json.gz'
    values = {}
    if path.exists():
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            payload = json.load(stream)
        if payload.get('recipe') != RECIPE or payload.get('pillow') != pillow_version:
            raise ValueError('La receta de bordes cambió: conservar la caché anterior y revisar antes de recalcular')
        values = payload['values']
        if _digest(values) != payload['values_sha256']:
            raise ValueError('La caché de bordes ha sido modificada')
    new = 0
    started = time.perf_counter()
    rows = list(frame.itertuples())
    def measure(row):
        key = cache_key(row)
        image_path = (root/'data/processed'/row.image_path).resolve()
        if not image_path.is_relative_to((root/'data/processed').resolve()):
            raise ValueError('Ruta de imagen fuera de data/processed')
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != row.prepared_sha256:
            raise ValueError(f'Foto modificada: {row.article_id}')
        with Image.open(image_path) as image:
            score = border_foreground(image, int(row.width), int(row.height))
        return key, score

    # Cuatro lecturas pequeñas concurrentes; tareas acotadas a 256 fotos.
    # Solo el hilo principal modifica o guarda la caché.
    saved_at = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        for offset in range(0, len(rows), 256):
            block = rows[offset:offset+256]
            missing = [row for row in block if cache_key(row) not in values]
            for key, score in executor.map(measure, missing):
                values[key] = score
                new += 1
            if new-saved_at >= 1000:
                save_cache(path, values)
                saved_at = new
                completed = offset+len(block)
                remaining = sum(cache_key(row) not in values for row in rows[completed:])
                eta = remaining*(time.perf_counter()-started)/new/60
                print(f'Encuadre: {completed:,}/{len(frame):,}; medidas nuevas {new:,}; '
                      f'estimación restante {eta:.1f} min; sin CLIP.', flush=True)
    result = []
    for row in rows:
        value = values[cache_key(row)]
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Medida de borde no válida')
        result.append(value)
    if new:
        save_cache(path, values)
    print(f'Encuadre terminado: {len(frame):,} fotos; {new:,} medidas nuevas; '
          f'{len(frame)-new:,} reutilizadas ({time.perf_counter()-started:.1f} s).', flush=True)
    return np.asarray(result)
