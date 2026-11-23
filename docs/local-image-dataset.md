# Conjunto local de imágenes: alcance y calidad antes que cantidad

## Qué define el conjunto

El objetivo es `product_type_name` y el repertorio contiene doce tipos de prenda,
explicados uno por uno en 01. Accesorios, calzado, ropa interior, baño, conjuntos
y otras categorías quedan fuera del alcance definido. No se afirma que sus fotos
sean malas. La etiqueta genérica Top queda fuera del repertorio específico; dos
ejemplos ilustrativos no prueban una propiedad de toda su categoría.

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
validation. Los grupos nuevos completan las proporciones de clase. Si aparecen
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

## Estado al 23/11/2026

La reconstrucción se ha detenido a petición de la estudiante para que ejecute
el código personalmente. La configuración y los notebooks están adaptados;
la preparación completa, el EDA y el entrenamiento sobre esta cohorte están
pendientes de ejecución integral.

El puntero preparado anterior se ha conservado en un respaldo local y ya no
identifica datos activos. Las salidas anteriores de 01–04 y sus métricas se han
retirado de la propuesta vigente para no presentarlas como resultados nuevos.
Los originales y las copias parciales siguen conservados. No se declara un
número final de imágenes hasta publicar correctamente el conjunto completo.

Consultar el README para ejecutar 01, 02, 03 y 04 en orden. La preparación puede
tardar y la medición de coste de 04 precede a los seis ajustes. Solo se guardan
notebooks completados sin errores.
