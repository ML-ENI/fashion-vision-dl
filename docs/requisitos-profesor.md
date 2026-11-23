# Revisión de DL — actualizada el 23 de noviembre de 2026

## Material consultado

- Proyecto local: README, los doce notebooks, preparador de imágenes, contrato de datos, pruebas y auditoría de preparación.
- `C:/Users/irfer/Downloads/PRDL_MLLB_Presentation_III_Learning_and_Evaluation.pptx`: en particular diapositivas 35, 43–44, 46–51, 60–63, 73–79 y 97.
- `C:/Iratxe/UPM/master - sandra/Apuntes IPAD/2º MUIT/PRDL - MLLB/1. Notebook for DL Reports.pdf`: 36 páginas, revisadas completas. Es material de la carpeta de Sandra; para la evaluación actual se ha contrastado con la presentación del curso disponible en Descargas.

## Actualización prioritaria: correo facilitado el 20/11/2026

El nuevo correo concreta el alcance y sustituye las interpretaciones anteriores
que dejaban Transformer y GenAI como ampliaciones. Exige:

- Resumen estructurado al comienzo del informe DL, **máximo tres páginas**, con
  alcance, metodología central, principales hallazgos y conclusiones.
- Aplicación de CNN, LSTM, GRU, RNN + atención, self-attention y Transformers a
  ejemplos sencillos.
- Estudio y aplicación de modelos preentrenados de manera autosupervisada, conectados
  con transferencia, fine-tuning y zero-shot.
- GenAI básica con **al menos un generador autorregresivo de secuencias**, RNN o
  Transformer causal. Un VAE no sustituye ese mínimo.
- Prioridad al análisis y la claridad metodológica, sin depender de GPU potente.
- Borrador inicial el **1 de diciembre**; planificado para 2026 por el contexto del
  curso, ya que el texto no explicita el año. Revisión en clases de diciembre.

La matriz de cobertura y la estructura propuesta están en
`../ENUNCIADO_TRABAJO_DL_MODA.md`; los experimentos nuevos siguen pendientes.

## Requisitos de materiales anteriores que se conservan

- Formular objetivos e hipótesis; justificar los datos y realizar EDA.
- La diapositiva 61 proponía Dense, CNN y LSTM y condicionaba Transformer a la viabilidad. El correo posterior incluye explícitamente Transformers, GRU y atención; usar ejemplos pequeños para cubrirlos. Dense se conserva como baseline.
- Hacer estudios de ablación: son obligatorios. Se valoran experimentos sencillos que permitan comprender los fundamentos, con entrenamiento, evaluación y refinamiento justificados.
- Autoencoder y VAE se conservan como ampliaciones opcionales. La generación autorregresiva básica pasa a ser obligatoria por el nuevo correo.
- Explicar en PRDL los fundamentos, hiperparámetros, limitaciones y resultados; en MLLB, implementación, herramientas y metodología. Entregar el código, preferiblemente mediante GitHub.
- La presentación indica trabajo de DL en equipo, con informe individual y contribuciones personales identificadas (diapositiva 63).
- Documentar uso y validación de asistentes de IA, incluidos errores detectados, cómo se comprobaron y su corrección. Preparar la defensa oral de las decisiones y del código.
- La guía de notebooks también pide revisar la primera tarea sobre estrategias de aprendizaje e incorporar conexiones entre conceptos y lecturas de investigación a los informes finales.

## Estado del proyecto

Los notebooks 01–04 están desarrollados y adaptados a una única propuesta que conserva todas las fotos elegibles, sin cupos por clase. Están pendientes de reejecución por la estudiante: se han retirado sus salidas anteriores para no confundirlas con evidencia del diseño vigente. 05–12 siguen como planes y comparten el protocolo actualizado. 06–08 y 12 cubren en su diseño atención, autosupervisión y generación autorregresiva; 11 aborda zero-shot. El VAE es opcional.

El preparador original `process_data.py` agrupa duplicados sin retirarlos. El nuevo flujo `audit_images.py` + `prepare_local.py`, usado por los notebooks 01 y 02, conserva todas las incidencias en el manifiesto y retiene un único representante de los píxeles idénticos para modelar. Los grupos con etiquetas contradictorias se excluyen. Las variantes relacionadas nunca cruzan particiones.

La auditoría del origen ha validado 105.100 imágenes. La preparación del conjunto completo se detuvo antes de publicar, a petición de la estudiante, que prefiere ejecutar personalmente. No hay un total final publicado ni métricas predictivas vigentes de esa cohorte. Consulta `local-image-dataset.md` y `local-preparation-verification.json` para el estado y las operaciones pendientes.

## Ubicación de las imágenes

- ZIP trasladado desde Descargas: `data/raw/images.zip` (30.036.309.564 bytes).
- Contenido del ZIP: 105.100 imágenes JPG; 30.557.038.537 bytes sin comprimir.
- Cruce de identificadores del ZIP con los 105.542 artículos de la copia canónica de `articles.csv`: 105.100 coincidencias, 442 artículos sin imagen y ninguna imagen sin metadatos. Es una comprobación de nombres e identificadores, no de decodificación.
- Destino requerido por el código: `data/raw/images/<prefijo>/<article_id>.jpg`.
- Se conserva el ZIP original. Tanto el ZIP como la carpeta de imágenes están excluidos de Git.
- La extracción comprueba el CRC de las entradas; esa comprobación de integridad no equivale a validar visualmente o decodificar todas las imágenes con el preparador.
- Extracción y comprobación terminadas: 105.100 archivos verificados, sin errores pendientes. Se reextrajeron 4.176 archivos durante la recuperación de una incidencia de lectura concurrente; todos superaron la comprobación CRC. Registro: `docs/image-extraction-verification.json`.

## Revisión del 23/11/2026: alcance, calidad y coste

El notebook 04 explica modelo, pérdida, parada y selección, variabilidad, intervalos,
errores, coste y limitaciones. Su código utiliza toda la cohorte elegible y mide
el coste antes de decidir por recursos; no iguala clases mediante filtros.
La reejecución queda a cargo de la estudiante. Diseño: `config/dense_baseline_v1.json`.
Estado pendiente: `dense-baseline-verification.json`. H1 con CNN sigue pendiente.
