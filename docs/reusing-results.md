# Reutilizar el 02 ejecutado

La selección vigente es `20261125T152902-c912cdcc`: 48.376 fotos, 25 clases;
train 33.618, validation 7.515 y test 7.243. No conserva Baby/Children.
`results/data_preparation/CURRENT.json` señala su copia verificada compartida.
El análisis completo original también se conserva y no se sobrescribe.

Una compañera con el mismo código y configuración puede cargar las puntuaciones
y **la selección terminada con sus estadísticas**, sin ejecutar CLIP ni medir
bordes de nuevo. Las fotos preparadas locales sí son necesarias para las galerías,
loaders y entrenamientos: la copia compartida contiene metadatos, no JPEG.

`CREAR_ANALISIS_CLIP=False` sigue siendo el modo normal. Las correcciones claras
están en `config/training_visual_review.csv`; las dudas humanas no se transforman
en decisiones definitivas. La regla exploratoria, sus errores y límites se
explican en 02 y `results/metrics/02_zoom_rule/validation_report.json`.

Si se cambia el umbral, las medidas guardadas en
`results/metrics/02_zoom_rule/geometry_cache.json.gz` se reutilizan. Una nueva
selección puede requerir recalcular estadísticas de train; eso no es CLIP.
No se reasignan productos ni se borran fotografías.

El 02 tiene sus 14 celdas ejecutadas sin errores. 03 también está ejecutado y validado (7/7 celdas, mismos IDs y manifiesto). 04 está ejecutado y validado sobre este conjunto (18/18 celdas), experimento
`143ad91452347fef`. Sus seis ajustes usan el mismo manifiesto y validation que 03. El 04
mantiene su entrenamiento desactivado por defecto y requiere pesos locales para
las visualizaciones de activaciones y pesos.
