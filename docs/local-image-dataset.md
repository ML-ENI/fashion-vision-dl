# Conjunto local de imágenes: alcance y calidad antes que cantidad

## Qué define el conjunto

El objetivo es `product_type_name`. Incluimos todos los tipos de las familias
`Garment Upper body`, `Garment Lower body` y `Garment Full body` de H&M:
26 etiquetas del catálogo actual. No elegimos tipos individualmente por frecuencia.
La configuración declara familias y etiquetas; el preparador exige que correspondan
a la lista completa. Top, conjuntos y disfraces también están incluidos.

Interior, baño, ropa de dormir, accesorios y calzado quedan fuera de esas familias.
La frontera es un alcance acordado del problema, no un juicio sobre su calidad.
La familia sirve para delimitar; la clase que predecimos sigue siendo el tipo concreto.

No hay un límite de imágenes por clase, una reserva de fotos válidas para galería
ni un objetivo de tamaño total. Conservamos todas las imágenes que superan los
controles. Las diferencias de frecuencia entre prendas forman parte del problema.

## Orden de las operaciones

1. Enlazar artículos e imágenes conservando los identificadores como texto.
2. Verificar decodificación completa, orientación, dimensiones y hashes.
3. Aplicar el alcance de prendas y un lado mínimo común de 64 píxeles.
4. Conectar variantes por producto y píxeles; excluir grupos de etiqueta conflictiva.
5. Conservar un representante por copia exacta, manteniendo la conexión del grupo.
6. Preparar todas las fotos elegibles: RGB224, proporción conservada, blanco y JPEG90.
7. Volver a resolver conexiones, conflictos y duplicados después de preparar.
8. Separar toda la cohorte en grupos, aproximadamente 70/15/15 por clase, respetando
   la exposición de desarrollo registrada. Publicar las proporciones por imágenes.
9. Calcular normalización y pesos opcionales solo con train.
10. Medir tiempo y memoria de entrenamiento antes de justificar cualquier muestra
    computacional. Si fuese necesaria, afectaría únicamente a train.

La regla de calidad no se confunde con una limitación de cálculo. Una foto válida
fuera de una eventual muestra de train seguiría siendo válida y permanecería en
el conjunto preparado.

## Productos vistos y test

`config/split_exposure.csv` evita mover al test artículos ya usados en train o
validation. Incluye como train los dos ejemplos de Top ya inspeccionados antes de
ampliar el alcance. Los grupos nuevos completan las proporciones de clase. Si aparecen
conexiones nuevas entre exposiciones, se adopta la más utilizada: train antes que
validation, y validation antes que test. Las galerías muestran train; el test
predictivo permanece reservado.

Las proporciones se refieren a grupos, por lo que no son exactas por imagen.
La evaluación describe el repertorio elegido después de los controles, no todas
las categorías de H&M, la demanda comercial ni otras marcas.

## Una publicación activa

`config/local_images_v1.json` es la configuración vigente. El publicador crea
una ejecución interna con manifiesto, esquema, estadísticas y checksums, y solo
después actualiza `data/processed/CURRENT.json`. Las fotos se identifican por hash
y se comparten cuando coinciden sus bytes. La trazabilidad interna no constituye
dos alternativas experimentales.

Los loaders consumen exclusivamente el conjunto preparado. No se eliminan
originales ni respaldos al cambiar la propuesta. Para entradas de resolución
superior a las copias guardadas habrá que recuperar los originales.

## Estado al 24/11/2026

La estudiante ha ejecutado 01 y 02. Ambos conservan contadores de ejecución en
sus celdas de código y no contienen errores guardados. La publicación vigente es
`runs/20261123T181635-92a8fcea`: 74.525 imágenes de 26 clases, con 51.791 en train,
11.583 en validation y 11.151 en test. Las copias preparadas ocupan 570.836.280 bytes.
Los originales y el ZIP siguen locales; no se ha liberado ese espacio.

La revisión de estos registros no ha vuelto a ejecutar la preparación ni pruebas.
El EDA y los entrenamientos siguientes siguen pendientes de ejecución completa y
validación. No se presentan métricas de la selección anterior como evidencia nueva.