"""Representaciones persistentes del 07, independientes del kernel y las métricas.

Una clave identifica los pesos y el preprocesamiento; otra, el JPEG preparado.
Cambiar la selección de prendas reutiliza las fotos ya calculadas. Cada lote
se confirma en disco: una interrupción no pierde los lotes anteriores.
Este módulo no conoce modelos, no descarga pesos y no entrena nada.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np


def identity(value):
    """Huella reproducible de una receta serializable; no incluye las métricas."""
    encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


class FeatureStore:
    """Caché local SQLite con un vector float32 por hash del JPEG.

    La apertura normal es de solo lectura. Solo la extracción autorizada usa
    ``create=True``. Los datos se validan también al leerlos, sin abrir fotos.
    """

    def __init__(self, directory, recipe, *, create=False):
        self.recipe = recipe
        self.directory = Path(directory) / identity(recipe)
        self.path = self.directory / 'features.sqlite'
        self.connection = None
        if not self.path.exists() and not create:
            return
        if create:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path)
            with self.connection:
                self.connection.execute(
                    'CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
                self.connection.execute(
                    'CREATE TABLE IF NOT EXISTS features ('
                    'image_sha TEXT PRIMARY KEY, vector BLOB NOT NULL, digest TEXT NOT NULL)')
                self.connection.execute(
                    'INSERT OR IGNORE INTO metadata VALUES (?, ?)',
                    ('recipe', json.dumps(recipe, sort_keys=True)))
        else:
            self.connection = sqlite3.connect(self.path.resolve().as_uri() + '?mode=ro', uri=True)
        saved = self.connection.execute('SELECT value FROM metadata WHERE key=?', ('recipe',)).fetchone()
        if saved is None or json.loads(saved[0]) != recipe:
            self.close()
            raise ValueError('La caché no corresponde a estos pesos/preprocesamiento')

    def keys(self):
        if self.connection is None:
            return set()
        return {row[0] for row in self.connection.execute('SELECT image_sha FROM features')}

    def put(self, hashes, vectors):
        """Guardar un lote entero o ninguno; nunca marcar como completo un vector parcial."""
        vectors = np.asarray(vectors, dtype='<f4')
        if vectors.shape != (len(hashes), self.recipe['dimension']) or not np.isfinite(vectors).all():
            raise ValueError('Representaciones con forma incorrecta o valores no finitos')
        if self.connection is None:
            raise RuntimeError('La extracción debe abrir la caché con create=True')
        records = []
        for digest, vector in zip(hashes, vectors):
            blob = vector.tobytes()
            records.append((digest, blob, hashlib.sha256(blob).hexdigest()))
        with self.connection:
            self.connection.executemany('INSERT OR REPLACE INTO features VALUES (?, ?, ?)', records)

    def read(self, hashes):
        """Recuperar vectores en el orden solicitado, comprobando contenido y tamaño."""
        if self.connection is None:
            raise FileNotFoundError('No hay representaciones guardadas. No se ha iniciado la extracción.')
        array = np.empty((len(hashes), self.recipe['dimension']), dtype=np.float32)
        for index, digest in enumerate(hashes):
            record = self.connection.execute(
                'SELECT vector, digest FROM features WHERE image_sha=?', (digest,)).fetchone()
            if record is None:
                raise FileNotFoundError(f'Falta una representación: {digest[:12]}. No se ha iniciado la extracción.')
            blob, expected = record
            if hashlib.sha256(blob).hexdigest() != expected:
                raise ValueError('Representación modificada o dañada en disco')
            vector = np.frombuffer(blob, dtype='<f4')
            if vector.shape != (self.recipe['dimension'],) or not np.isfinite(vector).all():
                raise ValueError('Representación guardada inválida')
            array[index] = vector
        return array

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
