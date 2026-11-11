# Enunciado del trabajo de Deep Learning — Clasificación visual de moda (H&M)

**Asignaturas:** PDRL y MLLB  
**Proyecto:** Fashion Vision DL  
**Fuente de datos:** H&M Personalized Fashion Recommendations (imágenes de artículo + `articles.csv`)  
**Alcance:** clasificación de imágenes con Dense, CNN, LSTM/GRU, Transformers y transfer learning; estudios de ablación; estrategias de aprendizaje (contrastivo, zero-shot) y modelos generativos.

> **Cómo interpretar este documento.** Se basa en la guía del profesorado
> `Notebooks_GuideLines_for_DeepLearnig_Report.pptx` y en los notebooks y presentaciones de
> la asignatura (FeedForward, CNN, RNN, Transfer Learning, Attention y Transformers, CLIP,
> VAE). La guía pide **diseñar experimentos sencillos propios** que ayuden a entender los
> fundamentos (módulos Dense, CNN y RNN, nociones de Attention/Transformers, efecto de
> dropout, batch normalization, etc.) y **relacionar las estrategias de aprendizaje**
> (transfer learning y fine-tuning, self-supervised, contrastive, zero-shot, in-context,
> meta learning, modelos generativos). La adaptación al caso H&M es una propuesta nuestra.

## Objetivo general

Dada la imagen de producto de un artículo de H&M, predecir su categoría de prenda
(por defecto `garment_group_name`, conjunto de clases justificado en el notebook 01)
y comparar arquitecturas de Deep Learning con un protocolo común: mismas particiones
agrupadas por `product_code`, mismas métricas (accuracy, F1 macro, precisión y recall
por clase, matriz de confusión) y curvas de aprendizaje.

## Organización de notebooks

| Bloque de la guía | Notebook | Resultado principal |
|---|---|---|
| Datos | `01_data_understanding.ipynb` | Inventario de imágenes y metadatos, elección justificada de clases |
| Preparación | `02_data_preparation.ipynb` | Split agrupado y estratificado, `Dataset`/`DataLoader`, preprocesado y augmentation |
| EDA | `03_eda_images.ipynb` | Análisis exploratorio de imágenes y baselines clásicos |
| Dense Layers | `04_dense_mlp.ipynb` | MLP (flatten + Dense + softmax) como baseline de DL |
| Convolutional Layers | `05_cnn.ipynb` | CNN desde cero, BatchNorm, bloques residuales, feature maps |
| Recurrent Neural Networks | `06_rnn_lstm.ipynb` | Imagen como secuencia de columnas de píxeles con RNN/LSTM/GRU |
| Transfer Learning | `07_transfer_learning.ipynb` | Feature extraction vs. fine-tuning de un backbone preentrenado |
| Transformers | `08_vision_transformer.ipynb` | Mini-ViT desde cero y ViT preentrenado, mapas de atención |
| Experimentos propios | `09_ablation_studies.ipynb` | Ablaciones con varias semillas |
| Evaluación final | `10_final_evaluation.ipynb` | Evaluación única en test, comparación e interpretabilidad |
| Estrategias de aprendizaje | `11_clip_zero_shot.ipynb` | CLIP zero-shot, linear probe y relación entre estrategias |
| Modelos generativos | `12_generative_models.ipynb` | Autoencoder y VAE sobre prendas |

Los notebooks están en `notebooks/` con numeración fija. Por ahora solo contienen
introducción, objetivos y una vista previa del trabajo; el código reutilizable debe
ir en `src/` (`data/`, `models/`, `training/`, `evaluation/`).

## Recordatorios de la guía

- Priorizar entender los conceptos frente a abordar problemas grandes o complejos.
- Incluir la primera tarea (First Assignment) revisada en los informes finales de PRDL y MLLB.
- En el informe PRDL, elaborar un diagrama propio de relaciones entre las estrategias de aprendizaje.
- Ejecutar en Google Colab con GPU y registrar qué GPU se ha asignado (`nvidia-smi`).
