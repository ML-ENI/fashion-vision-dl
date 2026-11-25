# Ejecutar de nuevo sin repetir las horas de cálculo

## 02: el modo normal lee resultados

1. Carga la tabla completa de puntuaciones guardadas.
2. Recupera el manifiesto y los cálculos existentes cuando no cambia la selección.
3. Muestra tablas, gráficos y ejemplos con las fotos locales.

La celda opcional `CREAR_ANALISIS_CLIP` está en `False`. Si faltan resultados
compatibles, el cuaderno se detiene y explica qué falta. No arranca CLIP por sí solo.
Solo se activa esa celda cuando se decide generar un análisis nuevo o reanudar
uno incompleto. Cambiar Markdown o una gráfica no requiere activarla.

La carga de puntuaciones no exige RAW, auditoría de 01, Transformers ni descargar
un modelo. Para las galerías y lotes sí se necesitan las fotos preparadas locales.

## Archivos para compartir

`results/data_preparation/CURRENT.json` señala la copia completa guardada en
`results/data_preparation/20261124T185853-e7045c19`.
Son 12 archivos, aproximadamente 33,5 MB: tablas, puntuaciones y metadatos,
sin imágenes ni pesos de CLIP. Los checksums se han verificado al copiar.

Esta copia contiene las 74.525 puntuaciones del análisis terminado. El conjunto
publicado conserva 74.126 fotos: train 51.506, validation 11.532 y test 11.088.
La regla visual se aplica a los tres conjuntos. Las líneas infantiles se mantienen;
no se ha cambiado el alcance para retirarlas.

Una compañera necesita el código, esta carpeta de resultados y las fotos
preparadas para ejecutar las galerías y los loaders. No necesita tu kernel ni
repetir CLIP. La copia de puntuaciones de 02 ya está publicada. Los resultados nuevos de
03–04 forman parte de esta revisión local; este commit no realiza push.

## Cambiar una selección o una métrica

`finalize_saved_analysis` recibe puntuaciones ya calculadas. No contiene ninguna
llamada al analizador CLIP. Si cambia una selección explícita, recalcula los
recuentos y los estadísticos necesarios; si no cambia train, reutiliza también
sus medias y desviaciones. Las publicaciones originales permanecen intactas.
No elimina clases automáticamente si una selección las deja sin ejemplos.

En 04, `ENTRENAR_SI_FALTA=False` carga un experimento terminado para el mismo
manifiesto, arquitectura y configuración de entrenamiento. Modificar métricas,
Bootstrap, figuras o texto no inicia otro entrenamiento. Las métricas nuevas se
calculan en las celdas de análisis a partir de las predicciones guardadas (`y` y
`pred`), sin modificar los resultados históricos del entrenamiento.

Si el dataset cambió desde el entrenamiento anterior, no se presentan esos
resultados como si fueran del nuevo dataset: el cuaderno avisa antes de preparar
arrays. Entrenar el modelo sobre ese dataset por primera vez requiere activar
la bandera de la celda inicial. Después se reutiliza.

Los arrays normalizados tienen también una caché verificada: no se reescriben
si no cambian los arrays de entrada ni la normalización.

## Validación realizada

Ocho pruebas con datos ficticios, sin CLIP ni entrenamiento real: carga repetida,
copia compartida sin RAW ni imágenes, incompatibilidad, archivos alterados,
selección sin inferencia, reutilización de estadísticas y arrays normalizados,
y comprobación de que las llamadas costosas están protegidas en los notebooks.
Las ocho pasan. No se ha vuelto a ejecutar el análisis real de las fotografías.

El 04 ya terminó sobre la publicación filtrada: experimento `67338715b6e6941e`.
Su bandera queda en `False`. Los checkpoints siguen en local: son necesarios
para la visualización de pesos y activaciones, aunque las predicciones y métricas
permitan revisar resultados sin entrenar. El 03 reutiliza la extracción de color
guardada, pero vuelve a ajustar sus dos baselines clásicos y calcular intervalos;
no ejecuta CLIP ni entrena las redes de 04.
