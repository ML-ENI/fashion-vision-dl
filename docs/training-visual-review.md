# Filtro de fotos de detalle en el notebook 02

El filtro se aplica a todas las fotos elegibles de las 26 clases: train,
validation y test. Se utiliza la misma regla en los tres conjuntos, sin mover
las fotos conservadas entre particiones ni borrar originales.

## Criterio visual

Apartar primeros planos que solo muestran tejido, cuello, costura, cierre u otro
detalle sin suficiente estructura para reconocer la prenda. Conservar las vistas
parciales reconocibles y las dudosas.

CLIP compara cada foto con tres pares de textos fijados en la configuración.
Cada par contrapone una vista reconocible a un detalle aislado. Solo se excluye
automáticamente si los tres prefieren detalle. Los desacuerdos y empates se
conservan. Los pares no son pruebas independientes; sus similitudes no son
probabilidades. Esta aproximación está pendiente de validación en nuestras fotos.

## Alcance y trazabilidad

El reparto por productos se hace antes del filtro. Se guarda la partición
original en `assigned_split`, incluso para una foto excluida. Los motivos son
`automatic_detail_image` o `detail_only_image` para una decisión manual.

La tabla `automatic_visual_scores.csv.gz` contiene las puntuaciones, los votos y
la partición de todas las fotos analizadas. El 02 presenta recuentos y miniaturas
separados de train, validation y test. No se cambia la regla buscando mejores
resultados predictivos en test. La evaluación describirá el conjunto filtrado.
Si una clase pierde todos sus ejemplos de una partición, la publicación se detiene.

El registro `config/training_visual_review.csv` permite corregir decisiones en
cualquiera de las tres particiones. Su nombre histórico no limita su alcance.
La decisión manual, ligada al hash del original, prevalece sobre CLIP.

## Ejecución

Se necesita `requirements-visual-filter.txt`. La primera vez se descargan los
pesos de `openai/clip-vit-base-patch32`, revisión
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. Las fotos se procesan localmente.
Se reutilizan JPEG compatibles verificados y se guardan predicciones por lotes
con checksum en `tmp/automatic_visual_filter`. Una interrupción conserva la
publicación anterior y permite reanudar con la misma configuración.

El cambio de alcance invalida la caché anterior de solo train. La publicación y
las comparaciones siguientes deben ejecutarse con el nuevo alcance. 03 registra
el manifiesto y 04 comprueba que sus referencias correspondan a los mismos datos.

Pendiente de ejecución y validación por la estudiante. El asistente no ha
realizado descargas, inferencia, entrenamiento ni ejecución de pruebas.

Fuentes: [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) y
[API de Transformers utilizada](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/clip/modeling_clip.py).