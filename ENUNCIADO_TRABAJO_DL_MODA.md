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

## Actualización obligatoria del profesorado: correo compartido el 20/11/2026

Esta actualización procede del correo facilitado por la estudiante. Prevalece sobre
nuestra interpretación anterior de los apuntes cuando hay diferencias de alcance.
El correo no incluye fecha de envío ni año de entrega; por el contexto del curso,
planificamos el borrador para el **1 de diciembre de 2026**. Las revisiones se harán
en las sesiones de diciembre. No se especifica hora ni plataforma de entrega.

- **Resumen al comienzo del informe DL, máximo tres páginas:** alcance, metodología
  central, principales hallazgos y conclusiones. Verificar el límite en el PDF final;
  un recuento de palabras no garantiza el número de páginas. En el borrador, separar
  resultados medidos de experimentos pendientes.
- **Arquitecturas fundamentales:** aplicar CNN, RNN (LSTM y GRU), RNN + atención,
  self-attention y Transformers a ejemplos sencillos. Dense sigue como baseline propio.
- **Estrategias de aprendizaje:** estudiar y aplicar representaciones de modelos
  preentrenados de forma autosupervisada; relacionarlas con transferencia, fine-tuning
  y zero-shot. Un backbone supervisado de ImageNet o CLIP por sí solos no documentan
  toda esta cobertura: explicar el objetivo y los datos de cada checkpoint.
- **GenAI básica:** estudiar y aplicar al menos un generador autorregresivo de
  secuencias, mediante una RNN o un Transformer causal pequeño. Autoencoder y VAE
  quedan como ampliaciones opcionales y no sustituyen este requisito.
- Priorizar análisis, fundamentos y claridad metodológica. Los modelos masivos,
  segmentación médica avanzada, SAM o variantes complejas de YOLO no son el objetivo.
  La falta de GPU potente no debe bloquear el trabajo. Se pueden adaptar ejemplos
  prácticos de clase; identificar la fuente y las modificaciones propias.

### Estructura de entrega y evidencia mínima

1. Resumen inicial de hasta tres páginas con los cuatro contenidos exigidos.
2. Problema, datos, alcance y limitaciones del muestreo; protocolo común y baselines.
3. Arquitecturas: pregunta, módulo, experimento sencillo, evidencia y explicación.
4. Estrategias: objetivo autosupervisado, transferencia, fine-tuning y zero-shot,
   diferenciando lo aplicado de las conexiones conceptuales.
5. GenAI: siguiente token, entrenamiento autorregresivo y ejemplos de generación.
6. Discusión y conclusiones: qué se ha entendido, límites y coste local.
7. Referencias y anexos técnicos breves; evitar repetir salidas de notebooks en el cuerpo.

Esta estructura es nuestra propuesta para cubrir el correo; el profesor exige el
resumen y los tres bloques, no estos títulos concretos. El borrador del 1 de diciembre
mostrará el estado real de cada bloque. Actualmente solo 01–03 tienen salidas ejecutadas.

| Requisito | Evidencia prevista | Estado actual |
|---|---|---|
| CNN y baseline Dense | 04–05: curvas, contraste controlado y errores | Pendiente |
| LSTM, GRU y RNN + atención | 06: mismo encoder recurrente con pooling frente a atención | Pendiente |
| Self-attention y Transformer | 08: ejemplo Q/K/V y mini-ViT; máscara causal en 12 si se amplía | Pendiente |
| Preentrenamiento autosupervisado | 07: encoder congelado y adaptación pequeña, con objetivo documentado | Pendiente |
| Zero-shot y conexión entre estrategias | 11: CLIP, prompts fijados con validation y diagrama | Pendiente |
| Generación autorregresiva básica | 12: siguiente carácter/token y muestreo secuencial | Pendiente |
| Resumen inicial ≤3 páginas | Verificación visual del informe exportado | Pendiente |

## Objetivo general

Dada la imagen de producto de un artículo de H&M, predecir su categoría de prenda
(`product_type_name` en la versión visual vigente; selección de doce clases exploratoria y documentada en el notebook 01)
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
| Recurrent Neural Networks | `06_rnn_lstm.ipynb` | Imagen como secuencia: RNN, LSTM, GRU y RNN + atención |
| Transfer Learning | `07_transfer_learning.ipynb` | Transferencia y fine-tuning; incluir un encoder autosupervisado |
| Transformers | `08_vision_transformer.ipynb` | Mini-ViT desde cero y ViT preentrenado, mapas de atención |
| Experimentos propios | `09_ablation_studies.ipynb` | Ablaciones con varias semillas |
| Evaluación final | `10_final_evaluation.ipynb` | Evaluación única en test, comparación e interpretabilidad |
| Estrategias de aprendizaje | `11_clip_zero_shot.ipynb` | CLIP zero-shot, linear probe y relación entre estrategias |
| Modelos generativos | `12_generative_models.ipynb` | Generador autorregresivo pequeño; autoencoder/VAE opcionales |

Los notebooks están en `notebooks/` con numeración fija. Los notebooks 01–03 están
ejecutados y guardan sus salidas: auditoría, preparación y EDA con baselines clásicos
sobre validation. Los notebooks 04–12 conservan planes y todavía no presentan
resultados de entrenamiento de redes. El código reutilizable vive en `src/`.
El protocolo previo al entrenamiento está en `config/experimental_protocol_v1.json`: macro-F1 primaria, pérdida inicial sin pesos, comparación controlada RGB64 y ablaciones previstas. La muestra equilibrada no representa por sí sola la prevalencia original del catálogo.

## Recordatorios de la guía

- Priorizar entender los conceptos frente a abordar problemas grandes o complejos.
- Incluir la primera tarea (First Assignment) revisada en los informes finales de PRDL y MLLB.
- En el informe PRDL, elaborar un diagrama propio de relaciones entre las estrategias de aprendizaje.
- En este proyecto se trabaja en local. Registrar hardware y GPU si se utiliza; Colab queda como alternativa de cómputo. Esta elección de entorno es nuestra, no un nuevo requisito del profesor.
