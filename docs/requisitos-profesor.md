# Revisión de DL — 19 de noviembre de 2026

## Material consultado

- Proyecto local: README, los doce notebooks, preparador de imágenes, contrato de datos, pruebas y auditoría de preparación.
- `C:/Users/irfer/Downloads/PRDL_MLLB_Presentation_III_Learning_and_Evaluation.pptx`: en particular diapositivas 35, 43–44, 46–51, 60–63, 73–79 y 97.
- `C:/Iratxe/UPM/master - sandra/Apuntes IPAD/2º MUIT/PRDL - MLLB/1. Notebook for DL Reports.pdf`: 36 páginas, revisadas completas. Es material de la carpeta de Sandra; para la evaluación actual se ha contrastado con la presentación del curso disponible en Descargas.

## Lo que pide el profesor

- Formular objetivos e hipótesis; justificar los datos y realizar EDA.
- Probar como mínimo Dense, CNN y LSTM; Transformer si es posible (diapositiva 61).
- Hacer estudios de ablación: son obligatorios. Se valoran experimentos sencillos que permitan comprender los fundamentos, con entrenamiento, evaluación y refinamiento justificados.
- Los modelos generativos son una ampliación valorada; no sustituyen las arquitecturas básicas ni las ablaciones.
- Explicar en PRDL los fundamentos, hiperparámetros, limitaciones y resultados; en MLLB, implementación, herramientas y metodología. Entregar el código, preferiblemente mediante GitHub.
- La presentación indica trabajo de DL en equipo, con informe individual y contribuciones personales identificadas (diapositiva 63).
- Documentar uso y validación de asistentes de IA, incluidos errores detectados, cómo se comprobaron y su corrección. Preparar la defensa oral de las decisiones y del código.
- La guía de notebooks también pide revisar la primera tarea sobre estrategias de aprendizaje e incorporar conexiones entre conceptos y lecturas de investigación a los informes finales.

## Estado del proyecto

Los notebooks 01 y 02 ya están desarrollados y ejecutados con las imágenes reales: auditoría completa, decisiones de selección y publicación local versionada. Los notebooks 03–12 siguen siendo planes, sin experimentos de entrenamiento ejecutados. La estructura prevista cubre las familias básicas, transfer learning, ablaciones y evaluación final. CLIP y VAE son ampliaciones.

El preparador original `process_data.py` agrupa duplicados sin retirarlos. El nuevo flujo `audit_images.py` + `prepare_local.py`, usado por los notebooks 01 y 02, conserva todas las incidencias en el manifiesto y retiene un único representante de los píxeles idénticos para modelar. Los grupos con etiquetas contradictorias se excluyen. Las variantes relacionadas nunca cruzan particiones.

La auditoría real ha validado 105.100 imágenes. La publicación preparada contiene 18.000 imágenes de doce tipos de prenda en 139,3 MB, con particiones y controles ejecutados. Falta desarrollar el EDA detallado de entrenamiento y los experimentos de los notebooks siguientes. Todavía no se ha entrenado ningún modelo. Consulta `local-image-dataset.md` y `local-preparation-verification.json` para resultados y comprobaciones.

## Ubicación de las imágenes

- ZIP trasladado desde Descargas: `data/raw/images.zip` (30.036.309.564 bytes).
- Contenido del ZIP: 105.100 imágenes JPG; 30.557.038.537 bytes sin comprimir.
- Cruce de identificadores del ZIP con los 105.542 artículos de la copia canónica de `articles.csv`: 105.100 coincidencias, 442 artículos sin imagen y ninguna imagen sin metadatos. Es una comprobación de nombres e identificadores, no de decodificación.
- Destino requerido por el código: `data/raw/images/<prefijo>/<article_id>.jpg`.
- Se conserva el ZIP original. Tanto el ZIP como la carpeta de imágenes están excluidos de Git.
- La extracción comprueba el CRC de las entradas; esa comprobación de integridad no equivale a validar visualmente o decodificar todas las imágenes con el preparador.
- Extracción y comprobación terminadas: 105.100 archivos verificados, sin errores pendientes. Se reextrajeron 4.176 archivos durante la recuperación de una incidencia de lectura concurrente; todos superaron la comprobación CRC. Registro: `docs/image-extraction-verification.json`.
