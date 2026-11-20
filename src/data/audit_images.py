"""Auditoría completa, reanudable y versionada del origen H&M."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import io
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from data_contract import articles_path, load_articles, metadata, publish, sha256, write_json
from process_data import inspect_image

ROOT = Path(__file__).resolve().parents[2]
AUDIT_VERSION = "1.0.0"


def inspect_record(task):
    """Misma validación que inspect_image, sin copias RGB redundantes.

    Cada proceso abre su archivo; no se comparten descriptores de lectura.
    """
    record, base = task
    path = Path(base) / record["raw_path"]
    content = path.read_bytes()
    file_hash = hashlib.sha256(content).hexdigest()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as check:
                check.verify()
            with Image.open(io.BytesIO(content)) as image:
                image.load()
                mode = image.mode
                if mode not in {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK"}:
                    raise ValueError(f"Modo no admitido: {mode}")
                oriented = ImageOps.exif_transpose(image) if image.getexif().get(274, 1) != 1 else image
                rgb = oriented if oriented.mode == "RGB" else oriented.convert("RGB")
                width, height = rgb.size
                if min(width, height) < 1:
                    raise ValueError("Dimensiones vacías")
                digest = hashlib.sha256(f"{width}x{height}:RGB:".encode())
                digest.update(rgb.tobytes())
                result = {"image_valid": True, "reason": "", "width": width, "height": height,
                          "original_mode": mode, "file_sha256": file_hash, "pixel_sha256": digest.hexdigest()}
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        result = {"image_valid": False, "reason": f"invalid_image: {exc}", "width": 0, "height": 0,
                  "original_mode": "", "file_sha256": file_hash, "pixel_sha256": ""}
    return {**record, **result}


def inventory(images_dir):
    rows = []
    for directory, folders, files in os.walk(images_dir):
        folders.sort()
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            stat = path.stat()
            rows.append({"article_id": path.stem,
                         "raw_path": path.relative_to(images_dir).as_posix(),
                         "file_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    frame = pd.DataFrame(rows).sort_values("raw_path").reset_index(drop=True) if rows else pd.DataFrame()
    if frame.empty:
        raise ValueError(f"No hay imágenes en {images_dir}")
    if frame.article_id.duplicated().any():
        raise ValueError("Hay nombres de imagen repetidos; revisar antes de continuar")
    return frame


def read_audit(root=ROOT):
    directory = root / "data/audit"
    pointer = json.loads((directory / "CURRENT.json").read_text(encoding="utf-8"))
    run = directory / pointer["run_dir"]
    for name, digest in json.loads((run / "checksums.json").read_text()).items():
        if sha256(run / name) != digest:
            raise ValueError(f"Auditoría modificada: {name}")
    frame = pd.read_csv(run / "image_audit.csv.gz", dtype={"article_id": "string", "product_code": "string"}, keep_default_na=False)
    return run, frame, json.loads((run / "summary.json").read_text(encoding="utf-8"))


def ensure_audit(images_dir=None, workers=4, refresh=False, root=ROOT):
    images_dir = Path(images_dir or root / "data/raw/images").resolve()
    source = articles_path(root, root / "data/raw", None)
    output = root / "data/audit"
    sources = [Path(__file__), Path(__file__).with_name("process_data.py"), Path(__file__).with_name("data_contract.py")]
    code_hashes = {p.name: sha256(p) for p in sources}
    articles_hash = sha256(source)
    if (output / "CURRENT.json").exists() and not refresh:
        run, frame, summary = read_audit(root)
        provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
        if provenance["inputs"]["articles"]["sha256"] != articles_hash or provenance["source_sha256"] != code_hashes:
            raise ValueError("Han cambiado metadatos/código: repetir auditoría con --refresh")
        if not images_dir.exists():
            print("RAW desconectado: se reutiliza la auditoría verificada", flush=True)
            return run
        listing = inventory(images_dir)
        signature = hashlib.sha256(listing.to_csv(index=False).encode()).hexdigest()
        if signature != summary["inventory_sha256"]:
            raise ValueError("Ha cambiado el inventario RAW: repetir con --refresh")
        print(f"Auditoría vigente verificada: {run}", flush=True)
        return run
    articles = load_articles(source)
    listing = inventory(images_dir)
    signature = hashlib.sha256(listing.to_csv(index=False).encode()).hexdigest()
    # Los checkpoints registran la versión semántica del inspector. Cambiar
    # procesos/hilos no invalida hashes de píxeles ya verificados.
    fingerprint = hashlib.sha256(json.dumps([signature, articles_hash, AUDIT_VERSION], sort_keys=True).encode()).hexdigest()
    checkpoint = output / "checkpoints" / fingerprint
    checkpoint.mkdir(parents=True, exist_ok=True)
    parts = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(listing), 1000):
            path = checkpoint / f"{start:06d}.csv.gz"
            if path.exists():
                part = pd.read_csv(path, dtype={"article_id": "string"}, keep_default_na=False)
            else:
                tasks = [(record, str(images_dir)) for record in listing.iloc[start:start + 1000].to_dict("records")]
                part = pd.DataFrame(pool.map(inspect_record, tasks, chunksize=16))
                temporary = path.with_suffix(".pending")
                part.to_csv(temporary, index=False, compression="gzip")
                os.replace(temporary, path)
            parts.append(part)
            print(f"Auditadas {min(start + 1000, len(listing)):,}/{len(listing):,}", flush=True)
    inspected = pd.concat(parts, ignore_index=True)
    merged = articles.merge(inspected, on="article_id", how="left", validate="one_to_one")
    merged["image_exists"] = merged["raw_path"].notna()
    merged["image_valid"] = merged["image_valid"].fillna(False).astype(bool)
    merged["reason"] = merged["reason"].fillna("missing_image")
    for name in ["raw_path", "original_mode", "file_sha256", "pixel_sha256"]:
        merged[name] = merged[name].fillna("")
    for name in ["width", "height", "file_bytes", "mtime_ns"]:
        merged[name] = merged[name].fillna(0).astype("int64")
    valid = merged.loc[merged.image_valid]
    summary = {"audit_version": AUDIT_VERSION, "articles": len(articles), "images_found": int(merged.image_exists.sum()),
               "images_valid": len(valid), "images_invalid": int((merged.image_exists & ~merged.image_valid).sum()),
               "missing_images": int((~merged.image_exists).sum()), "orphan_images": int((~inspected.article_id.isin(articles.article_id)).sum()),
               "raw_bytes": int(listing.file_bytes.sum()), "duplicate_pixel_rows": int(valid.pixel_sha256.duplicated().sum()),
               "inventory_sha256": signature, "raw_image_base": str(images_dir),
               "invalid_reasons": merged.loc[merged.image_exists & ~merged.image_valid, "reason"].value_counts().to_dict()}
    inputs = {"articles": source}
    archive = root / "data/raw/images.zip"
    if archive.is_file():
        inputs["images_archive"] = archive
    print("Calculando hashes del origen y publicando auditoría", flush=True)
    provenance = metadata({"audit_version": AUDIT_VERSION, "workers": workers, "raw_image_base": str(images_dir)}, inputs, sources)
    run = publish(output, {"image_audit": merged, "orphan_images": inspected.loc[~inspected.article_id.isin(articles.article_id)]},
                  {"summary": summary, "provenance": provenance})
    print(json.dumps(summary, indent=2), flush=True)
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    print(ensure_audit(args.images_dir, args.workers, args.refresh))
