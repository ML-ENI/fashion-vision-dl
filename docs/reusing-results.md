# Reutilizar los resultados del 02

El modo normal carga las puntuaciones CLIP existentes y, si no cambia la selección,
reutiliza también el manifiesto, los pesos opcionales y la normalización. La carga
y los cálculos están en celdas distintas. `CREAR_ANALISIS_CLIP=False` impide lanzar
un análisis nuevo por sorpresa; si faltan resultados compatibles, aparece un aviso.

Los resultados completos están en `results/data_preparation/20261124T185853-e7045c19`,
con su CURRENT.json. Son 12 archivos y 33.498.172 bytes, sin fotografías ni pesos.
Contienen las puntuaciones de 74.525 fotos. El conjunto filtrado conserva 74.126:
51.506 en train, 11.532 en validation y 11.088 en test. Los checksums protegen la
copia; .gitattributes conserva sus bytes entre sistemas.

Una compañera puede cargar esas tablas sin RAW, auditoría de 01 ni Transformers.
Para mostrar fotos o entregar lotes sí necesita las copias preparadas locales.
No se ha implementado una descarga automática de imágenes ni se ha cambiado
su ubicación. Las líneas Baby/Children se mantienen; no se ha autorizado retirarlas.

Una selección explícita puede usar estas puntuaciones sin volver a llamar a CLIP.
Los cálculos se actualizan solo cuando cambia su entrada. Las publicaciones fuente
se conservan y no se eliminan clases silenciosamente si una selección las deja vacías.

Validación: 14 celdas del 02 guardadas por la estudiante sin errores; mensajes de
carga de puntuaciones y reutilización de cálculos confirmados. Pruebas pequeñas
pertinentes al 02: cinco de carga/reutilización, cuatro del filtro con puntuaciones
simuladas y catorce del preparador con imágenes sintéticas. Ninguna ejecuta CLIP
real ni entrenamiento. Los cambios de 03–04 se revisan por separado.