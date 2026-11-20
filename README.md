
# Fashion Vision DL

## Estado de ejecución y flujo local (notebooks 01–03)

Las figuras usan el mismo módulo de estilo que ML, copiado en
`src/visualization/style.py`: Latin Modern, fondo blanco, texto grande y colores
azul/naranja ETSIT-UPM. La rúbrica está en `docs/visualization-rubric.md`.
Para modificar solo su presentación y guardar las figuras y outputs actualizados:
`python scripts/render_data_figures.py` (requiere los notebooks ya ejecutados).

Los notebooks `01_data_understanding.ipynb` y `02_data_preparation.ipynb` contienen
la auditoría y preparación del conjunto local. Los notebooks 01–03 están ejecutados y guardan sus salidas: auditoría, preparación,
EDA visual y baselines clásicos medidos en validation. Los notebooks 04–12 son
planes de redes y extensiones, todavía sin entrenamiento ejecutado.

1. Ejecutar el notebook 01 para revisar el origen, la taxonomía, incidencias y
   selección. La auditoría inicial decodifica todas las imágenes y guarda
   checkpoints; las ejecuciones posteriores verifican y reutilizan la publicación.
2. Ejecutar el notebook 02 para preparar imágenes RGB de 224×224 y publicar
   manifiestos, splits, estadísticas de train y procedencia. La configuración
   está en `config/local_images_v1.json`: doce tipos de prenda y hasta 1.500
   imágenes por clase, conservando grupos completos de producto.
3. Leer siempre `data/processed/CURRENT.json`. Cada versión guarda sus tablas e
   informes en `data/processed/runs/`; los JPEG se comparten por hash en
   `data/processed/images/` y las rutas del manifiesto son relativas a processed.
4. Los loaders de `src/data/local_dataset.py` leen exclusivamente imágenes
   preparadas. El adaptador PyTorch es opcional y se valida al instalar ese
   framework; los notebooks comprueban los lotes NumPy sin instalarlo.

Si el código del preparador cambia desde la auditoría registrada, ejecutar
`python src/data/audit_images.py --refresh` para publicar una auditoría nueva antes
de reconstruir los datos; las publicaciones y salidas anteriores conservan su procedencia.

Equivalentes por consola, desde este directorio:

```bash
python src/data/audit_images.py --workers 6
python src/data/prepare_local.py --config config/local_images_v1.json
python -m unittest discover -s tests -v
```

La auditoría se publica en `data/audit/runs/` con su propio `CURRENT.json`.
Las miniaturas reservadas para inspección visual (24 de las clases elegidas y
dos ejemplos de la categoría Top descartada), excluidas del modelado,
permiten consultar el notebook 01 sin el RAW una vez completada la primera ejecución.
Los duplicados exactos se reducen a un representante. Los grupos con etiquetas
contradictorias se excluyen; las variantes del mismo producto y los píxeles
idénticos, antes y después de preparar, nunca cruzan particiones.

**El RAW sigue preservado localmente.** Antes de liberarlo hay que elegir un
destino externo, copiarlo y verificar su hash. No se borran automáticamente ni
originales ni versiones anteriores. Para recrear imágenes a más de 224×224
será necesario recuperar el RAW. No se necesita para leer la versión preparada.

El resto del README documenta el diseño general y el preparador original
`process_data.py`, que sigue disponible para manifiestos sobre imágenes RAW.
Su objetivo por defecto es `garment_group_name`; el nuevo flujo local usa
`product_type_name`, justificado en el notebook 01.

Deep Learning project for classifying H&M fashion articles from product images and comparing Dense, CNN, LSTM and Transformer-based architectures.

## Protocolo previo al entrenamiento

`config/experimental_protocol_v1.json` fija métricas, presupuestos, pérdida inicial
sin pesos y ablaciones previstas. El dataset equilibrado es exploratorio y no
representa por sí solo la prevalencia del catálogo. El test sigue reservado.

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
          â†“
Image / label matching
          â†“
Class selection
          â†“
Train / validation / test split
          â†“
Resize and normalization
          â†“
Optional data augmentation
          â†“
Dense / CNN / LSTM / Transformer
          â†“
Evaluation and ablation studies
```

## Project Structure

```text
fashion-vision-dl/
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/
â”‚   â””â”€â”€ processed/
â”œâ”€â”€ notebooks/
â”‚   â”œâ”€â”€ eda/
â”‚   â””â”€â”€ experiments/
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ training/
â”‚   â””â”€â”€ evaluation/
â”œâ”€â”€ results/
â”‚   â”œâ”€â”€ figures/
â”‚   â”œâ”€â”€ metrics/
â”‚   â””â”€â”€ models/
â”œâ”€â”€ README.md
â””â”€â”€ requirements.txt
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
