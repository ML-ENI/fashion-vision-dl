"""Publicar imágenes locales pequeñas y manifiestos inmutables desde una auditoría."""
from __future__ import annotations

import hashlib
import io
import importlib.metadata
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageStat

from audit_images import ROOT, read_audit
from data_contract import metadata, publish, sha256
from process_data import grouped_splits

VERSION = "1.0.0"
DEFAULT_CONFIG = {
    "preparation_version": VERSION,
    "target": "product_type_name",
    "include_classes": ["Trousers", "Dress", "Sweater", "T-shirt", "Cardigan", "Blouse",
                        "Jacket", "Shorts", "Shirt", "Vest top", "Skirt", "Hoodie"],
    "min_side": 64, "min_class_images": 3,
    "image_size": 224, "jpeg_quality": 90, "padding_rgb": [255, 255, 255],
    "seed": 42, "val_size": 0.15, "test_size": 0.15,
    "split_exposure_path": None,
    "augmentation": {"horizontal_flip_probability": 0.5, "brightness": 0.1, "colour": 0.1},
}


def connect_groups(frame, hashes):
    """Componentes transitivas: producto, píxeles RAW y píxeles preparados."""
    parent = {str(a): str(a) for a in frame.article_id}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for column in ["product_code", *hashes]:
        first = {}
        for article, value in zip(frame.article_id, frame[column]):
            article = str(article)
            if not value:
                raise ValueError(f"Falta {column} en {article}")
            if value in first:
                a, b = find(article), find(first[value])
                parent[max(a, b)] = min(a, b)
            else:
                first[value] = article
    return pd.Series([find(str(a)) for a in frame.article_id], index=frame.index)


def read_exposure(config, root=ROOT):
    path = config.get("split_exposure_path")
    if not path:
        return pd.DataFrame(columns=["article_id", "split"])
    result = pd.read_csv(root / path, dtype={"article_id": "string"})
    if result.article_id.duplicated().any() or not result.split.isin(["train", "validation", "test"]).all():
        raise ValueError("Registro de exposición inválido")
    return result


def exposure_by_group(frame, column="history_split"):
    priority = {"": 0, "test": 1, "validation": 2, "train": 3}
    rank = frame[column].map(priority)
    if rank.isna().any():
        raise ValueError("Partición de exposición desconocida")
    maxima = rank.groupby(frame.group_id).transform("max")
    return maxima.map({v: k for k, v in priority.items()})


def select_candidates(audit, config, root=ROOT):
    if "max_images_per_class" in config or "visual_audit_groups_per_class" in config:
        raise ValueError("La selección ya no admite cupos ni reservas por clase")
    frame = audit.copy()
    target = config["target"]
    frame["label"] = frame[target].astype(str)
    frame["exclusion_reason"] = frame["reason"]
    ok = frame.image_valid.astype(bool)
    for mask, reason in [
        (~frame.label.isin(config["include_classes"]), "class_not_selected"),
        (frame[["width", "height"]].min(axis=1) < config["min_side"], "too_small"),
    ]:
        frame.loc[ok & mask, "exclusion_reason"] = reason
        ok &= ~mask
    candidate = frame.loc[ok].copy()
    candidate["group_id"] = connect_groups(candidate, ["pixel_sha256"])
    candidate["raw_group_id"] = candidate["group_id"]
    anchors = read_exposure(config, root).set_index("article_id").split
    candidate["history_split"] = candidate.article_id.map(anchors).fillna("")
    candidate["history_split"] = exposure_by_group(candidate)
    conflicts = candidate.groupby("group_id").label.nunique()
    bad = candidate.group_id.isin(conflicts[conflicts.gt(1)].index)
    frame.loc[candidate.index[bad], "exclusion_reason"] = "conflicting_group_labels"
    candidate = candidate.loc[~bad].sort_values("article_id")
    duplicate = candidate.pixel_sha256.duplicated(keep="first")
    frame.loc[candidate.index[duplicate], "exclusion_reason"] = "exact_pixel_duplicate"
    candidate = candidate.loc[~duplicate]
    frame.loc[candidate.index, "exclusion_reason"] = ""
    return frame, candidate


def assign_splits(candidate, config):
    """Dividir toda la cohorte por grupos, respetando exposiciones anteriores.

    Los grupos nuevos completan 70/15/15 por clase aproximadamente. Un producto
    ya utilizado en train no puede reaparecer como validación ni test.
    """
    val, test = config['val_size'], config['test_size']
    if not (0 < val < 1 and 0 < test < 1 and val+test < 1):
        raise ValueError('Proporciones positivas con suma menor que uno')
    result = candidate.copy()
    result["history_split"] = exposure_by_group(result)
    result["split"] = ""
    for label, part in result.groupby("label", sort=True):
        groups = part.groupby("group_id").history_split.first()
        ordered = sorted(groups.index, key=lambda g: hashlib.sha256(f'split:{config["seed"]}:{label}:{g}'.encode()).hexdigest())
        if len(ordered) < 3:
            raise ValueError(f"{label}: se necesitan al menos tres grupos independientes")
        assignment = {g: groups[g] for g in ordered if groups[g]}
        free = [g for g in ordered if not groups[g]]
        for split, fraction in [("test", config["test_size"]), ("validation", config["val_size"])]:
            target = max(1, int(np.ceil(len(ordered)*fraction)))
            missing = max(0, target-sum(v == split for v in assignment.values()))
            take, free = free[:missing], free[missing:]
            assignment.update({g: split for g in take})
        assignment.update({g: "train" for g in free})
        result.loc[part.index, "split"] = part.group_id.map(assignment)
        if set(assignment.values()) != {"train", "validation", "test"}:
            raise ValueError(f"{label}: la exposición previa impide crear tres particiones; revisar grupos")
    return result


def materialize(record, raw_base, output, config):
    path = raw_base / record["raw_path"]
    raw_content = path.read_bytes()
    if hashlib.sha256(raw_content).hexdigest() != record["file_sha256"]:
        raise ValueError(f'RAW modificado: {record["article_id"]}')
    with Image.open(io.BytesIO(raw_content)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = ImageOps.pad(image, (config["image_size"],) * 2, method=Image.Resampling.LANCZOS,
                             color=tuple(config["padding_rgb"]), centering=(0.5, 0.5))
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=config["jpeg_quality"], optimize=True, subsampling=0)
    content = stream.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    relative = f"images/{digest[:2]}/{digest}.jpg"
    asset = output / relative
    asset.parent.mkdir(parents=True, exist_ok=True)
    # Varias versiones apuntan al mismo contenido. La escritura es idempotente.
    if not asset.exists() or sha256(asset) != digest:
        temporary = asset.with_name(f'.{digest}.{record["article_id"]}.pending')
        temporary.write_bytes(content)
        temporary.replace(asset)
    with Image.open(io.BytesIO(content)) as decoded:
        rgb = decoded.convert("RGB")
        pixels = hashlib.sha256(f"{rgb.width}x{rgb.height}:RGB:".encode() + rgb.tobytes()).hexdigest()
    return {"article_id": record["article_id"], "image_path": relative,
            "prepared_sha256": digest, "prepared_pixel_sha256": pixels, "prepared_bytes": len(content)}


def channel_statistics(manifest, output):
    sums, squares, pixels = np.zeros(3), np.zeros(3), 0
    for row in manifest.loc[manifest.split.eq("train")].itertuples():
        with Image.open(output / row.image_path) as image:
            stats = ImageStat.Stat(image)
            sums += np.array(stats.sum)
            squares += np.array(stats.sum2)
            pixels += image.width * image.height
    mean = sums / pixels / 255
    std = np.sqrt(np.maximum(squares / pixels / 255**2 - mean**2, 0))
    if (std <= 0).any():
        raise ValueError("Canal con varianza nula")
    return {"mean": mean.tolist(), "std": std.tolist(), "pixels": pixels,
            "partition": "train", "space": "RGB [0,1]", "method": "pixel-weighted population moments, prepared JPEG"}


def verify_run(run, root=ROOT, verify_assets=True):
    output = root / "data/processed"
    for name, digest in json.loads((run / "checksums.json").read_text()).items():
        if sha256(run / name) != digest:
            raise ValueError(f"Salida modificada: {name}")
    manifest = pd.read_csv(run / "image_manifest.csv.gz", dtype={"article_id": "string", "product_code": "string", "group_id": "string", "raw_group_id": "string"}, keep_default_na=False)
    validate_manifest(manifest, output, verify_assets)
    return manifest


def validate_manifest(manifest, output, verify_assets=True):
    usable = manifest.loc[manifest.split.ne("excluded")]
    if usable.empty:
        raise ValueError("Publicación vacía")
    for key in ["group_id", "raw_group_id", "product_code", "pixel_sha256", "prepared_pixel_sha256"]:
        if usable.groupby(key).split.nunique().gt(1).any():
            raise ValueError(f"Fuga entre particiones: {key}")
    if usable.prepared_pixel_sha256.duplicated().any():
        raise ValueError("Duplicados exactos en el conjunto preparado")
    if 'history_split' in usable:
        seen = usable.loc[usable.history_split.ne('')]
        if not seen.history_split.eq(seen.split).all():
            raise ValueError('La partición contradice la exposición previa del producto')
    if verify_assets:
        for row in usable.itertuples():
            path = output / row.image_path
            if not path.resolve().is_relative_to(output.resolve()) or sha256(path) != row.prepared_sha256:
                raise ValueError(f"Imagen preparada modificada: {row.article_id}")


def prepare(config=None, root=ROOT, workers=4):
    config = dict(DEFAULT_CONFIG if config is None else config)
    audit_run, audit, audit_summary = read_audit(root)
    output = root / "data/processed"
    sources = [Path(__file__), Path(__file__).with_name("process_data.py"),
               Path(__file__).with_name("local_dataset.py"), Path(__file__).with_name("data_contract.py")]
    signature = {"config": config, "audit_checksums": sha256(audit_run / "checksums.json"),
                 "code_sha256": {p.name: sha256(p) for p in sources},
                 "packages": {name: importlib.metadata.version(name) for name in ["Pillow", "numpy", "pandas"]}}
    if config.get("split_exposure_path"):
        signature["split_exposure_sha256"] = sha256(root / config["split_exposure_path"])
    if (output / "CURRENT.json").exists():
        pointer = json.loads((output / "CURRENT.json").read_text())
        current = output / pointer["run_dir"]
        if json.loads((current / "signature.json").read_text()) == signature:
            verify_run(current, root)
            print(f"Versión vigente verificada: {current}", flush=True)
            return current
    frame, candidate = select_candidates(audit, config, root)
    raw_base = Path(audit_summary["raw_image_base"])
    if not raw_base.is_dir():
        raise FileNotFoundError("Reconecta el RAW para generar una versión nueva")
    records = candidate.to_dict("records")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        assets = []
        for i, result in enumerate(pool.map(lambda row: materialize(row, raw_base, output, config), records), 1):
            assets.append(result)
            if i % 1000 == 0:
                print(f"Preparadas {i:,}/{len(records):,}", flush=True)
    asset_frame = pd.DataFrame(assets).set_index("article_id")
    for column in asset_frame:
        candidate[column] = candidate.article_id.map(asset_frame[column])
    # Mantener también las conexiones transitivas anteriores a eliminar copias:
    # quitar un representante no debe separar un componente del origen.
    candidate["group_id"] = connect_groups(candidate, ["raw_group_id", "pixel_sha256", "prepared_pixel_sha256"])
    candidate["history_split"] = exposure_by_group(candidate)
    conflicts = candidate.groupby("group_id").label.nunique()
    bad = candidate.group_id.isin(conflicts[conflicts.gt(1)].index)
    frame.loc[candidate.index[bad], "exclusion_reason"] = "prepared_conflicting_group_labels"
    candidate = candidate.loc[~bad].sort_values("article_id")
    duplicate = candidate.prepared_pixel_sha256.duplicated()
    frame.loc[candidate.index[duplicate], "exclusion_reason"] = "prepared_pixel_duplicate"
    candidate = candidate.loc[~duplicate].copy()
    if set(candidate.label) != set(config["include_classes"]) or candidate.groupby("label").size().min() < config["min_class_images"]:
        raise ValueError("La limpieza final no permite el conjunto de clases configurado")
    candidate = assign_splits(candidate, config)
    labels = {name: i for i, name in enumerate(sorted(candidate.label.unique()))}
    counts = candidate.loc[candidate.split.eq("train"), "label"].value_counts()
    weights = (counts.sum() / (len(counts) * counts)).to_dict()
    frame["split"] = "excluded"
    frame["label_id"] = frame.label.map(labels).astype("Int64")
    frame["training_weight"] = 0.0
    for column in ["image_path", "prepared_sha256", "prepared_pixel_sha256", "group_id", "raw_group_id", "history_split"]:
        frame[column] = ""
    frame["prepared_bytes"] = 0
    for column in ["split", "image_path", "prepared_sha256", "prepared_pixel_sha256", "group_id", "raw_group_id", "prepared_bytes", "history_split"]:
        frame.loc[candidate.index, column] = candidate[column]
    train = frame.split.eq("train")
    frame.loc[train, "training_weight"] = frame.loc[train, "label"].map(weights)
    frame["reason"] = frame.exclusion_reason
    selected = frame.loc[frame.split.ne("excluded")]
    stats = channel_statistics(frame, output)
    summary = {"preparation_version": VERSION, "articles": len(frame), "images_usable": len(selected),
               "classes_usable": len(labels), "split_counts": selected.split.value_counts().to_dict(),
               "class_distribution_by_split": pd.crosstab(selected.label, selected.split).to_dict(),
               "excluded_reasons": frame.loc[frame.split.eq("excluded"), "reason"].value_counts().to_dict(),
               "prepared_bytes": int(selected.prepared_bytes.sum()), "raw_bytes": audit_summary["raw_bytes"],
               "image_path_base": "data/processed (relative to project root)", "image_size": config["image_size"],
               "split_unit": "connected product_code, RAW pixel hash and prepared pixel hash",
               "balance_policy": "No class caps or balancing; keep all eligible images. Optional loss weights derived from train only.",
               "split_exposure_policy": "Preserve prior development exposure; conflicts inherit train > validation > test; new groups complete approximate class-stratified fractions",
               "raw_required_for_training": False, "raw_archive_moved_externally": False,
               "audit_run": audit_run.relative_to(root).as_posix()}
    provenance = metadata(config, {"audit_checksums": audit_run / "checksums.json"}, sources)
    # Verificar el conjunto completo ANTES de permitir que publish cambie CURRENT.
    validate_manifest(frame, output, verify_assets=True)
    run = publish(output, {"image_manifest": frame}, {"summary": summary, "labels": labels,
                  "training_class_weights": weights, "normalization": stats, "config": config,
                  "signature": signature, "provenance": provenance})
    verify_run(run, root)
    print(json.dumps(summary, indent=2), flush=True)
    return run


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/local_images_v1.json")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(prepare(json.loads(args.config.read_text(encoding="utf-8")), workers=args.workers))
