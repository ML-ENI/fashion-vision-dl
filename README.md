# Fashion Vision DL

> **02 ejecutado y validado.** La publicación `20261125T152902-c912cdcc` conserva 48.376 fotos y 25 clases (train 33.618, validation 7.515, test 7.243), sin Baby/Children. El filtro exploratorio de bordes añadió 999 exclusiones; las revisiones humanas tienen prioridad. 03 está ejecutado y validado con esta selección; 04 sigue pendiente.

Clasificación visual de prendas H&M para los informes de PRDL y MLLB. La prioridad
es explicar los fundamentos y comparar experimentos sencillos en local.

## Una sola propuesta de datos

1. **Definir qué prendas queremos clasificar.** Incluimos todos los tipos de las
   familias `Garment Upper body`, `Garment Lower body` y `Garment Full body` de H&M:
   **26 etiquetas**, incluidas Top, polos, abrigos, blazers, conjuntos y disfraces.
   Predecimos el tipo concreto, no solo una de las tres familias. Interior, baño,
   pijamas, calzado y accesorios quedan fuera por alcance, no por mala calidad.
2. **Aplicar controles iguales:** imagen enlazada y decodificable, lado mínimo de
   64 píxeles, grupos sin etiquetas contradictorias y un representante por copia
   exacta. Los controles de duplicados se repiten después de preparar las imágenes.
3. **Conservar todas las fotos elegibles.** No hay cupos por clase ni un total
   prefijado. Si hay más pantalones que cárdigans, esa diferencia permanece.
4. **Separar grupos de producto completos.** Los grupos nuevos completan
   aproximadamente 70/15/15 por clase. Se publican las proporciones reales por fotos.
5. **Medir el coste antes de reducir train.** Una muestra adicional solo tendría
   sentido si el coste lo exige y con una justificación registrada. Validation y
   test no se recortan para equilibrarlos.

`config/local_images_v1.json` contiene las familias y todas sus etiquetas. El preparador
comprueba que la lista coincida con todos los tipos de esas familias, sin descartar
etiquetas a mano. Este archivo contiene las decisiones vigentes; su nombre identifica
el archivo de configuración, no dos propuestas de dataset. `data/processed/CURRENT.json`
se crea al terminar y verificar la preparación. Las publicaciones internas anteriores
son trazabilidad y no se presentan como experimentos paralelos.

## Ejecutar sin repetir CLIP ni entrenamientos

El modo normal de 02 carga las puntuaciones completas guardadas. CLIP está
apagado por defecto: si faltan resultados compatibles, muestra un aviso antes
de iniciar cualquier análisis. La carga y los cálculos están en celdas distintas.

Los resultados de las 74.525 fotos están protegidos en
`results/data_preparation/20261124T185853-e7045c19` (aprox. 33,5 MB, sin fotos ni
pesos). Esa carpeta permite compartir las puntuaciones con el proyecto.

El 04 carga resultados compatibles antes de preparar arrays; no entrena
por sorpresa. Las métricas y figuras usan predicciones guardadas. Las fotos
preparadas siguen siendo necesarias para galerías y cargas de imágenes.

[Instrucciones y comprobaciones](docs/reusing-results.md). Ocho pruebas pequeñas
con datos ficticios pasan; no se ha repetido CLIP ni entrenado redes reales.

## Estado de la selección actual

**02 terminado: 14/14 celdas ejecutadas, sin errores.** Las puntuaciones CLIP,
las medidas de bordes y los cálculos finales están guardados. La copia compartida
vigente está en `results/data_preparation/20261125T152902-c912cdcc`; conserva
los archivos exactos y sus hashes. No contiene JPEG ni pesos de modelos.

El modo normal instala la selección compartida y reutiliza las estadísticas,
sin repetir CLIP ni medir bordes. Las fotos locales siguen siendo necesarias
para galerías y entrenamiento. Costumes no tiene validation y Tailored Waistcoat
no tiene test; sus soportes cero se declaran, sin mover fotografías.

03 está ejecutado y validado sobre esta selección: 7/7 celdas, sin errores.
04 sigue pendiente de ejecutar y validar; sus resultados anteriores son históricos.

## Ejecución anterior (antes del cambio de selección)

01–04 tienen ejecuciones completas guardadas y sin errores. El 02 conserva
74.126 fotos: train 51.506, validation 11.532 y test 11.088. El filtro visual
se aplica a las tres particiones; en esa ejecución se mantenían las líneas infantiles y no había cupos.
El 03 y el experimento Dense `67338715b6e6941e` usan esa misma publicación
`20261124T185853-e7045c19`. Las seis ejecuciones Dense terminaron sin evaluar test.

05–12 contienen planes, sin código ni experimentos terminados. Su revisión
comprueba coherencia y cobertura docente; no equivale a ejecución. La revisión
del 25/11 prepara un commit local en `dev/iratxe`; no realiza push.

Los pesos y las fotografías siguen en local. El 04 necesita sus checkpoints
locales para las visualizaciones de pesos y activaciones; las predicciones y
métricas compartidas permiten revisar resultados sin volver a entrenar.

## Qué hace cada notebook

| Notebook | Trabajo |
|---|---|
| 01 | Alcance de prendas, auditoría, calidad y explicación de las copias |
| 02 | Preparación completa, particiones, integridad y loaders |
| 03 | EDA de train, frecuencias observadas y baselines clásicos en validation |
| 04 | Dense RGB64, medición de coste, tres semillas y ablación de dropout |
| 05–08 | CNN, LSTM/GRU con atención, transferencia y self-attention/Transformer |
| 09–10 | Ablaciones y evaluación final con test reservado |
| 11–12 | Zero-shot y generador autorregresivo básico |

## Evitar que un producto ya visto se convierta en test

`config/split_exposure.csv` registra productos usados anteriormente en desarrollo
mediante sus artículos. No es un filtro de calidad ni un cupo. Permite conservar
su partición al incorporar todas las fotos elegibles.

Las conexiones de producto y píxeles se resuelven antes de dividir. Si una conexión
nueva une exposiciones diferentes, prevalece train sobre validation y validation
sobre test. Los productos ya usados no reaparecen como una prueba supuestamente
nueva. La inspección visual utiliza el lado de train, sin excluir fotos válidas
solo para completar una galería.

## Preparación y entrenamiento son decisiones distintas

Guardamos copias RGB224 con proporción conservada, relleno blanco y JPEG calidad
90. Esto reduce el espacio de **las mismas fotografías**. No significa que debamos
elegir menos fotos ni que todos los modelos tengan que recibir RGB224.

La adaptación prevista de Dense recibirá RGB64: Flatten → Dense(80) → ReLU → Dense(32) → ReLU → 26 logits,
con 986.570 parámetros. Usa entropía cruzada sin pesos, sin aumentos y Adam.
Las frecuencias desiguales hacen necesario leer macro-F1, accuracy y resultados
por clase juntos. Los pesos de clase se reservan para una ablación explícita.
Los loaders no activan aumentos por defecto. Al cambiar la resolución se pasan
estadísticos calculados con train a ese tamaño, evitando mezclar RGB64 y RGB224.

Los arrays numéricos de 04 están respaldados por disco para limitar las copias en
RAM. Antes de los seis ajustes se mide una época piloto completa y se descartan
sus pesos. El presupuesto operativo previsto es de 90 minutos para los seis
ajustes; la estimación incluye treinta épocas máximas y un margen del 25 %.
Si la estimación supera ese presupuesto, el código se detiene para revisar el diseño: no elimina fotos
automáticamente ni decide por la puntuación del piloto.

La época piloto y los seis ajustes están pendientes. La CNN de 05 deberá mantener
los mismos productos, particiones y presupuesto para el contraste controlado.

## Cómo ejecutarlo cuando decidas hacerlo

Desde la raíz del repositorio, con el kernel **Python (Fashion Vision DL local)**:

```powershell
.\.venv\Scripts\python.exe scripts/execute_data_notebooks.py 01_data_understanding.ipynb
.\.venv\Scripts\python.exe scripts/execute_data_notebooks.py 02_data_preparation.ipynb
.\.venv\Scripts\python.exe scripts/execute_data_notebooks.py 03_eda_images.ipynb
.\.venv\Scripts\python.exe scripts/execute_data_notebooks.py 04_dense_mlp.ipynb
```

También puedes abrir los notebooks y ejecutar sus celdas en ese orden. El 02
transforma todas las imágenes elegibles y puede tardar; el 04 entrena realmente.
El ejecutor solo guarda un notebook como completado si termina sin errores.
Si el notebook estaba abierto antes del cambio a 26 etiquetas, reinicia su kernel
para recargar el código y la configuración antes de volver a ejecutar.

El entorno `.venv` local ya existe. Las dependencias están en `requirements.txt`
y `requirements-dl.txt`. El entrenamiento aislado se inicia con
`.venv/Scripts/python scripts/run_dense_baseline.py`, después de preparar los datos.

## Archivos y estilo

- `data/raw/`: originales y ZIP preservados, excluidos de Git. Sigue pendiente
  elegir y verificar un destino externo; no se ha liberado ese espacio.
- `data/processed/`: fotos filtradas, manifiestos y publicación vigente cuando termine.
- `results/metrics/`: evidencia numérica de las ejecuciones; no contiene resultados
  vigentes de los notebooks reconstruidos hasta que se vuelvan a ejecutar.
- `results/models/04_dense/` y `tmp/`: checkpoints, cachés y respaldos locales.
- `src/visualization/`: tipografía del proyecto ML, azul ETSIT y naranja como base,
  tablas legibles y colores coherentes de particiones.

El resumen inicial del informe DL tendrá **máximo tres páginas**: alcance,
metodología, hallazgos y conclusiones. El borrador se planifica para el 1 de
diciembre de 2026, con año inferido del contexto del curso. Consulta
`ENUNCIADO_TRABAJO_DL_MODA.md` para la cobertura docente.
