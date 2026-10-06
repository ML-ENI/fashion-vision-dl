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

## Reports

The same project is documented from two complementary perspectives:

- **PDRL:** architecture principles, theoretical foundations, assumptions, limitations and critical interpretation.
- **MLLB:** implementation, preprocessing, training pipeline, tools, tuning, experiment tracking and reproducibility.

## License

This repository contains project code and documentation only. The H&M dataset and images remain subject to their original terms of use.
