#!/usr/bin/env python3
"""Validar imágenes y crear particiones por grupos de producto y contenido.

El manifiesto conserva incidencias. Solo imágenes decodificables reciben split.
Productos relacionados e imágenes con los mismos píxeles no cruzan particiones.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageOps

from data_contract import articles_path, load_articles, metadata, publish, sha256

ROOT = Path(__file__).resolve().parents[2]
ImageFile.LOAD_TRUNCATED_IMAGES = False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--articles-path", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--target", default="garment_group_name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--min-class-images", type=int, default=3)
    parser.add_argument("--include-class", action="append", default=[], help="Repetible: clases que se desean conservar")
    parser.add_argument("--parquet", action="store_true", help="Añade Parquet; requiere pyarrow")
    return parser.parse_args()


def image_index(images_dir):
    index = {}
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            if path.stem in index:
                raise ValueError(f"Nombre de imagen duplicado: {index[path.stem]} y {path}")
            index[path.stem] = path
    return index


def inspect_image(path):
    """verify comprueba estructura y load obliga a decodificar los píxeles."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                mode = image.mode
                if mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK"}:
                    raise ValueError(f"Modo no admitido: {mode}")
                # El hash detecta igualdad de contenido aunque cambie el formato.
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                width, height = rgb.size
                if min(width, height) < 1:
                    raise ValueError("Dimensiones vacías")
                digest = hashlib.sha256(f"{width}x{height}:RGB:".encode() + rgb.tobytes()).hexdigest()
        return {"image_valid": True, "reason": "", "width": width, "height": height,
                "original_mode": mode, "file_sha256": sha256(path), "pixel_sha256": digest}
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        return {"image_valid": False, "reason": f"invalid_image: {exc}", "width": 0,
                "height": 0, "original_mode": "", "file_sha256": sha256(path), "pixel_sha256": ""}


def add_groups(frame):
    """Une transitivamente artículos del mismo producto o con idénticos píxeles."""
    result = frame.sort_values("article_id").copy()
    parent = {str(i): str(i) for i in result["article_id"]}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for column in ["product_code", "pixel_sha256"]:
        first = {}
        for article, value in zip(result["article_id"], result[column]):
            article = str(article)
            if not value:
                raise ValueError(f"Falta {column} para agrupar {article}")
            if value in first:
                a, b = find(article), find(first[value])
                parent[max(a, b)] = min(a, b)
            else:
                first[value] = article
    result["group_id"] = [find(str(i)) for i in result["article_id"]]
    conflicts = result.groupby("group_id")["label"].nunique()
    if (conflicts > 1).any():
        raise ValueError(f"Grupos de producto/contenido con etiquetas incompatibles: {conflicts[conflicts > 1].index.tolist()[:10]}")
    return result


def grouped_splits(frame, seed, val_size, test_size):
    if not (0 < val_size < 1 and 0 < test_size < 1 and val_size + test_size < 1):
        raise ValueError("Proporciones positivas con suma menor que 1")
    rng = np.random.default_rng(seed)
    result = frame.copy()
    result["split"] = ""
    # Se reparten grupos por clase. Las proporciones de imágenes serán aproximadas.
    # ceil garantiza al menos un grupo por clase en validación y en prueba.
    for label, part in result.groupby("label", sort=True):
        groups = np.array(sorted(part["group_id"].unique()), dtype=object)
        n_val, n_test = math.ceil(len(groups) * val_size), math.ceil(len(groups) * test_size)
        n_train = len(groups) - n_val - n_test
        if min(n_train, n_val, n_test) < 1:
            raise ValueError(f"Clase {label!r}: {len(groups)} grupos no permiten train/val/test "
                             f"con las proporciones solicitadas ({n_train}/{n_val}/{n_test}). "
                             "Selecciona otras clases, más datos o proporciones distintas.")
        rng.shuffle(groups)
        assignment = {g: "train" for g in groups[:n_train]}
        assignment.update({g: "validation" for g in groups[n_train:n_train + n_val]})
        assignment.update({g: "test" for g in groups[n_train + n_val:]})
        result.loc[part.index, "split"] = part["group_id"].map(assignment)
    for key in ["group_id", "product_code", "pixel_sha256"]:
        if result.groupby(key)["split"].nunique().gt(1).any():
            raise ValueError(f"Fuga de grupos entre particiones: {key}")
    return result


def prepare_manifest(articles, images_dir, target, seed, val_size, test_size,
                     min_class_images=3, include_classes=()):
    if target not in articles or articles[target].isna().any():
        raise ValueError("Target ausente o con etiquetas vacías")
    if min_class_images < 1:
        raise ValueError("min-class-images debe ser positivo")
    images = image_index(images_dir)
    rows = []
    for number, record in enumerate(articles[["article_id", "product_code", target]].itertuples(index=False, name=None), 1):
        article, product, label = record
        path = images.get(article)
        row = {"article_id": article, "product_code": product, "label": str(label),
               "image_path": path.relative_to(images_dir).as_posix() if path else "",
               "image_exists": path is not None}
        row.update(inspect_image(path) if path else {
            "image_valid": False, "reason": "missing_image", "width": 0, "height": 0,
            "original_mode": "", "file_sha256": "", "pixel_sha256": ""})
        rows.append(row)
        if number % 5000 == 0:
            print(f"Revisados {number:,} artículos".replace(",", "."), flush=True)
    manifest = pd.DataFrame(rows)
    # Los filtros actúan sobre imágenes realmente utilizables.
    usable_mask = manifest["image_valid"].copy()
    if include_classes:
        unknown = set(include_classes) - set(manifest["label"])
        if unknown:
            raise ValueError(f"Clases solicitadas inexistentes: {sorted(unknown)}")
        excluded = usable_mask & ~manifest["label"].isin(include_classes)
        manifest.loc[excluded, "reason"] = "class_not_selected"
        usable_mask &= manifest["label"].isin(include_classes)
    counts = manifest.loc[usable_mask, "label"].value_counts()
    small = counts[counts < min_class_images].index
    manifest.loc[usable_mask & manifest["label"].isin(small), "reason"] = "class_too_small"
    usable_mask &= ~manifest["label"].isin(small)
    usable = manifest.loc[usable_mask].copy()
    if usable.empty:
        raise ValueError("No hay imágenes utilizables para generar particiones. Añade las imágenes y revisa los filtros.")
    usable = grouped_splits(add_groups(usable), seed, val_size, test_size)
    labels = {name: number for number, name in enumerate(sorted(usable["label"].unique()))}
    manifest["label_id"] = manifest["label"].map(labels).astype("Int64")
    manifest["split"] = "excluded"
    manifest["group_id"] = ""
    manifest.loc[usable.index, "split"] = usable["split"]
    manifest.loc[usable.index, "group_id"] = usable["group_id"]
    # Pesos calculados SOLO con entrenamiento. El entrenador deberá utilizarlos.
    train_counts = usable.loc[usable["split"] == "train", "label"].value_counts()
    weights = (train_counts.sum() / (len(train_counts) * train_counts)).to_dict()
    manifest["training_weight"] = 1.0
    train_mask = manifest["split"].eq("train")
    manifest.loc[train_mask, "training_weight"] = manifest.loc[train_mask, "label"].map(weights)
    summary = {
        "articles": len(manifest), "images_found": int(manifest["image_exists"].sum()),
        "images_valid": int(manifest["image_valid"].sum()), "images_usable": len(usable),
        "classes_in_metadata": int(manifest["label"].nunique()), "classes_usable": len(labels),
        "classes_without_valid_images": sorted(set(manifest["label"]) - set(manifest.loc[manifest["image_valid"], "label"])),
        "classes_excluded": sorted(set(manifest["label"]) - set(labels)),
        "excluded_reasons": manifest.loc[manifest["split"] == "excluded", "reason"].value_counts().to_dict(),
        "split_counts": usable["split"].value_counts().to_dict(),
        "class_distribution_by_split": pd.crosstab(usable["label"], usable["split"]).to_dict(),
        "duplicate_pixel_rows": int(usable["pixel_sha256"].duplicated().sum()),
        "orphan_images": len(set(images) - set(articles["article_id"])),
        "split_unit": "connected product_code and decoded-pixel hash groups, stratified by label",
        "image_path_base": str(images_dir.resolve()),
        "balance_policy": "inverse frequency training weights; no resampling; trainer must apply weights",
    }
    return manifest, labels, weights, summary


def main():
    args = parse_args()
    source = articles_path(ROOT, args.raw_dir, args.articles_path)
    articles = load_articles(source)
    images_dir = (args.images_dir or args.raw_dir / "images").resolve()
    if not images_dir.is_dir():
        raise ValueError(f"Falta la carpeta de imágenes: {images_dir}")
    manifest, labels, weights, summary = prepare_manifest(
        articles, images_dir, args.target, args.seed, args.val_size,
        args.test_size, args.min_class_images, args.include_class)
    provenance = metadata(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        {"articles": source}, [Path(__file__), Path(__file__).with_name("data_contract.py")])
    final = publish(args.output_dir, {"image_manifest": manifest},
                    {"summary": summary, "labels": labels, "training_class_weights": weights,
                     "provenance": provenance}, args.parquet)
    print(json.dumps(summary, indent=2))
    print("Ejecución publicada:", final)


if __name__ == "__main__":
    main()
