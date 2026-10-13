#!/usr/bin/env python3
"""Create a labelled, reproducibly split image manifest for Fashion Vision DL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--target", default="garment_group_name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--include-missing-images", action="store_true")
    return parser.parse_args()


def image_index(images_dir: Path) -> dict[str, Path]:
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    return {
        path.stem: path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    }


def stratified_splits(data: pd.DataFrame, seed: int, val_size: float, test_size: float) -> pd.Series:
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("val-size and test-size must be positive and sum to less than 1")
    train_idx, remainder_idx = train_test_split(
        data.index, test_size=val_size + test_size, random_state=seed, stratify=data["label"]
    )
    val_idx, test_idx = train_test_split(
        remainder_idx,
        test_size=test_size / (val_size + test_size),
        random_state=seed,
        stratify=data.loc[remainder_idx, "label"],
    )
    split = pd.Series("", index=data.index, dtype="string")
    split.loc[train_idx], split.loc[val_idx], split.loc[test_idx] = "train", "validation", "test"
    return split


def dataset_report(filename: str, data: pd.DataFrame) -> str:
    """Return a human-readable inventory similar to the raw dataset report."""
    lines = [
        "=" * 80,
        f"ARCHIVO: {filename}",
        "=" * 80,
        f"Número de filas: {len(data):,}",
        "",
        "CABECERAS:",
    ]
    lines.extend(f"  {number}: {column}" for number, column in enumerate(data.columns))
    lines.extend(["", "PRIMERA FILA:"])
    if data.empty:
        lines.append("  El dataset no contiene datos.")
    else:
        lines.extend(f"  {column}: {value}" for column, value in data.iloc[0].items())
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    articles = pd.read_csv(
        args.raw_dir / "articles.csv",
        dtype={"article_id": "string", "product_code": "string"},
    )
    if args.target not in articles.columns:
        raise ValueError(f"Unknown target column: {args.target}")
    if articles["article_id"].duplicated().any() or articles[args.target].isna().any():
        raise ValueError("article IDs must be unique and target labels cannot be empty")

    images = image_index(args.raw_dir / "images")
    manifest = articles[["article_id", args.target]].rename(columns={args.target: "label"}).copy()
    manifest["image_path"] = manifest["article_id"].map(
        lambda article_id: str(images[article_id].relative_to(ROOT)) if article_id in images else ""
    )
    manifest["image_exists"] = manifest["image_path"].ne("")
    usable = manifest if args.include_missing_images else manifest[manifest["image_exists"]].copy()
    usable["split"] = ""
    if len(usable):
        too_small = usable["label"].value_counts().loc[lambda counts: counts < 3]
        if len(too_small):
            raise ValueError(f"Classes need at least 3 images for stratification: {too_small.to_dict()}")
        usable["split"] = stratified_splits(usable, args.seed, args.val_size, args.test_size)

    label_names = sorted(manifest["label"].unique())
    label_to_id = {name: number for number, name in enumerate(label_names)}
    manifest["label_id"] = manifest["label"].map(label_to_id)
    split_map = usable.set_index("article_id")["split"]
    manifest["split"] = manifest["article_id"].map(split_map).fillna("")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_dir / "image_manifest.csv", index=False)
    (args.output_dir / "dataset_processed.txt").write_text(
        dataset_report("image_manifest.csv", manifest), encoding="utf-8"
    )
    (args.output_dir / "labels.json").write_text(
        json.dumps(label_to_id, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = {
        "articles": len(manifest),
        "images_found": int(manifest["image_exists"].sum()),
        "missing_images": int((~manifest["image_exists"]).sum()),
        "classes": len(label_to_id),
        "target": args.target,
        "split_counts": usable["split"].value_counts().to_dict(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not images:
        print("WARNING: data/raw/images is empty; copy the H&M images there and rerun.")


if __name__ == "__main__":
    main()
