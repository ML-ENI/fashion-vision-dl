
# Fashion Vision DL

Deep Learning project for classifying H&M fashion articles from product images and comparing Dense, CNN, LSTM and Transformer-based architectures.

## Overview

This project studies visual representation learning for fashion products using the image collection and article metadata from the **H&M Personalized Fashion Recommendations** dataset.

The main task is to predict a garment category from an article image while comparing different Deep Learning architectures under a common experimental setup.

The project is developed for the **PDRL & MLLB** Deep Learning reports.

## Objective

Given an H&M article image, predict its garment category using increasingly expressive neural architectures.

The study is designed to compare:

- **Dense networks** as a simple baseline,
- **Convolutional Neural Networks (CNNs)** for spatial feature learning,
- **LSTM-based models** for sequential representations,
- **Transformer / Vision Transformer approaches** when computationally feasible.

Ablation studies are included to evaluate the effect of relevant architectural and training decisions.

## Dataset

Main sources:

| Source | Purpose |
|---|---|
| H&M article images | Visual input |
| `articles.csv` | Class labels and product metadata |

The exact target category will be selected from the available article taxonomy, prioritizing a manageable and sufficiently balanced set of classes.

The original H&M dataset is not redistributed in this repository.

## Data Pipeline

```text
Article images + metadata
          ↓
Image / label matching
          ↓
Class selection
          ↓
Train / validation / test split
          ↓
Resize and normalization
          ↓
Optional data augmentation
          ↓
Dense / CNN / LSTM / Transformer
          ↓
Evaluation and ablation studies
```

## Project Structure

```text
fashion-vision-dl/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── eda/
│   └── experiments/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── evaluation/
├── results/
│   ├── figures/
│   ├── metrics/
│   └── models/
├── README.md
└── requirements.txt
```

## Experimental Design

All models should use equivalent data partitions and evaluation criteria whenever possible.

The workflow includes:

- baseline definition,
- controlled model comparison,
- hyperparameter refinement,
- final evaluation on unseen data,
- mandatory ablation studies.

Potential ablations include:

- data augmentation on/off,
- image resolution,
- dropout,
- batch normalization,
- network depth,
- optimizer configuration,
- pretrained vs. from-scratch training.

## Evaluation

Main metrics:

- accuracy,
- macro F1-score,
- per-class precision and recall,
- confusion matrix.

Training behaviour is also analysed using learning curves and validation loss.

## Reproducibility

Relevant experiments should record:

- architecture,
- hyperparameters,
- preprocessing configuration,
- augmentation strategy,
- random seed,
- data split,
- training history,
- evaluation metrics.

Place the H&M images under `data/raw/images/` (nested folders are supported),
then generate the labelled and stratified image manifest with:

```bash
python src/data/process_data.py
```

The default target is `garment_group_name`; use `--target` to select another
column from `articles.csv`. The same seed always produces the same split.

## Reports

The same project is documented from two complementary perspectives:

- **PDRL:** architecture principles, theoretical foundations, assumptions, limitations and critical interpretation.
- **MLLB:** implementation, preprocessing, training pipeline, tools, tuning, experiment tracking and reproducibility.

## License

This repository contains project code and documentation only. The H&M dataset and images remain subject to their original terms of use.


## Preparación validada de datos (versión 2)

El comando `python src/data/process_data.py` valida entradas y publica una ejecución
en `data/processed/runs/`. **Lee `data/processed/CURRENT.json` para localizar la
ejecución vigente**. Los CSV antiguos de la raíz se conservan como archivos heredados;
no contienen las correcciones de la versión 2. Nunca uses el resultado de PREVIEW
para un experimento completo.

Cada ejecución incluye `summary.json`, `provenance.json`, `schema.json` y
`checksums.json`. Se registran parámetros, versiones, hashes de entradas y código.
El cambio de CURRENT es atómico: un fallo no publica una ejecución a medias.

La copia canónica de `articles.csv`, cuando ambos repos están juntos, es
`../fashion-demand-ml/data/raw/articles.csv`. Se puede indicar otra ubicación con
`--articles-path`. En un clon independiente se usa `data/raw/articles.csv`.
No se borran copias locales; el hash y la ruta efectiva quedan registrados.
`src/data/data_contract.py` es el mismo contrato versionado en ambos repos;
si se modifica, hay que actualizar y comprobar ambas copias.

Para añadir Parquet, instala `python -m pip install -r requirements-parquet.txt`
y ejecuta el procesador con `--parquet`. Se conserva CSV comprimido como exportación.
Sin esa opción no se requiere pyarrow. Las lecturas CSV deben conservar los IDs como texto.

Pruebas: `python -m unittest discover -s tests -v`.
Consulta `docs/data-preparation-audit.md` para decisiones y limitaciones.


Se comprueban nombres repetidos, decodificación completa, dimensiones, modos de
color y hashes de archivo y píxeles. Los nombres duplicados provocan un error;
las imágenes ausentes o corruptas permanecen en el manifiesto con `split=excluded`
y un motivo. Nunca se asigna split a una imagen ausente. Sin imágenes utilizables
el proceso falla sin sustituir CURRENT.

`--include-class` (repetible) y `--min-class-images` controlan la selección.
Las particiones se estratifican por clase sobre grupos conectados por `product_code`
o hash de píxeles. Un grupo no cruza particiones. Las proporciones se aplican a
grupos y son aproximadas en imágenes: se publica la distribución real. Se exige
al menos un grupo por clase en cada partición con el redondeo configurado.
Grupos con etiquetas contradictorias provocan un error que debe revisarse.

El balance se propone mediante `training_weight` y `training_class_weights.json`,
calculados solo con entrenamiento. El futuro entrenador debe aplicar esos pesos;
generarlos no equilibra por sí solo un entrenamiento. El manifiesto usa rutas
relativas a `image_path_base` del resumen. Las métricas distinguen clases de
metadatos, clases con imágenes válidas y clases seleccionadas para modelado.
No se deducen campañas ni fechas a partir de las imágenes.
