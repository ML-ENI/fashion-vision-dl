"""Lectura local independiente del RAW; aumentos solo en entrenamiento."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps

ROOT = Path(__file__).resolve().parents[2]


class LocalImageDataset:
    def __init__(self, split, root=ROOT, run=None, augment=False, normalize=True, image_size=None, seed=42, normalization=None):
        if split not in {"train", "validation", "test"}:
            raise ValueError("Partición desconocida")
        if augment and split != "train":
            raise ValueError("Los aumentos solo se aplican a entrenamiento")
        self.root = Path(root)
        self.output = self.root / "data/processed"
        if run is None and not (self.output / 'CURRENT.json').exists():
            raise FileNotFoundError('Primero ejecuta el notebook 02 para preparar y publicar el conjunto vigente')
        self.run = Path(run) if run else self.output / json.loads((self.output / "CURRENT.json").read_text())["run_dir"]
        manifest = pd.read_csv(self.run / "image_manifest.csv.gz", dtype={"article_id": "string", "product_code": "string"}, keep_default_na=False)
        self.rows = manifest.loc[manifest.split.eq(split)].reset_index(drop=True)
        self.config = json.loads((self.run / "config.json").read_text())
        if 'max_images_per_class' in self.config:
            raise ValueError('Este manifiesto corresponde al cupo retirado; ejecutar el notebook 02 con la configuración vigente')
        self.image_size = image_size or self.config["image_size"]
        if normalize and self.image_size != self.config['image_size'] and normalization is None:
            raise ValueError('A esta resolución debes pasar normalization calculada con train; no reutilizar automáticamente las estadísticas RGB224')
        stats = normalization if normalization is not None else json.loads((self.run / "normalization.json").read_text())
        self.mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, None]
        self.std = np.asarray(stats["std"], dtype=np.float32)[:, None, None]
        self.augment, self.normalize = augment, normalize
        self.rng = np.random.default_rng(seed)
        self.split = split

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(self.output / row.image_path) as source:
            image = source.convert("RGB")
        if image.size != (self.image_size,) * 2:
            image = image.resize((self.image_size,) * 2, Image.Resampling.LANCZOS)
        if self.augment:
            cfg = self.config["augmentation"]
            if self.rng.random() < cfg["horizontal_flip_probability"]:
                image = ImageOps.mirror(image)
            image = ImageEnhance.Brightness(image).enhance(self.rng.uniform(1-cfg["brightness"], 1+cfg["brightness"]))
            image = ImageEnhance.Color(image).enhance(self.rng.uniform(1-cfg["colour"], 1+cfg["colour"]))
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255
        if self.normalize:
            array = (array - self.mean) / self.std
        return np.ascontiguousarray(array), int(row.label_id)

    def batches(self, batch_size=32, shuffle=False, seed=42):
        if batch_size < 1:
            raise ValueError("batch_size debe ser positivo")
        indexes = np.arange(len(self))
        if shuffle:
            np.random.default_rng(seed).shuffle(indexes)
        for start in range(0, len(indexes), batch_size):
            rows = [self[int(i)] for i in indexes[start:start+batch_size]]
            yield np.stack([r[0] for r in rows]), np.asarray([r[1] for r in rows], dtype=np.int64)


def make_torch_loaders(root=ROOT, run=None, batch_size=32, image_size=None, seed=42, augment_train=False, normalization=None):
    """Adaptador opcional: importar PyTorch solo cuando se vaya a entrenar.

    num_workers=0 evita cierres anidados no serializables en Windows y permite
    reproducir exactamente la secuencia del generador de aumentos del Dataset.
    """
    import torch
    from torch.utils.data import DataLoader
    generator = torch.Generator().manual_seed(seed)
    return {split: DataLoader(LocalImageDataset(split, root, run, augment=split == "train" and augment_train,
                                               image_size=image_size, seed=seed, normalization=normalization),
                             batch_size=batch_size, shuffle=split == "train", num_workers=0, generator=generator)
            for split in ["train", "validation", "test"]}
