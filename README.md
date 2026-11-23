# Fashion Vision DL

Clasificación visual de prendas H&M para los informes de PRDL y MLLB. La prioridad
es explicar los fundamentos y comparar experimentos sencillos en local.

## Una sola propuesta de datos

1. **Definir qué prendas queremos clasificar.** El notebook 01 explica las doce
   categorías incluidas y el alcance de las exclusiones. Una categoría fuera del
   problema no contiene necesariamente fotografías de mala calidad.
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

`config/local_images_v1.json` contiene las decisiones vigentes; su nombre identifica
el archivo de configuración, no dos propuestas de dataset. `data/processed/CURRENT.json`
se crea al terminar y verificar la preparación. Las publicaciones internas anteriores
son trazabilidad y no se presentan como experimentos paralelos.

## Estado: preparado para que lo ejecutes tú

Los notebooks 01–04 se han adaptado al criterio anterior y **están pendientes de
volver a ejecutarse**. Se han retirado sus salidas anteriores: aquellas métricas
correspondían a otra selección y no serían evidencia de este diseño. Los notebooks
05–12 siguen como planes y comparten el mismo protocolo actualizado.

La preparación que estaba en marcha se detuvo a petición de la estudiante.
También se detuvieron las colas de entrenamiento y notebooks. Los originales,
las copias generadas y los respaldos locales se conservan. El puntero del conjunto
anterior se retiró para evitar consumirlo por error como si fuera el nuevo.

No hay un número final publicado de imágenes ni resultados Dense de esta cohorte
hasta completar las ejecuciones. Los informes de estado están en
`docs/notebook-execution-status.json`, `docs/local-preparation-verification.json`
y `docs/dense-baseline-verification.json`.

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

Dense recibe RGB64: Flatten → Dense(80) → ReLU → Dense(32) → ReLU → doce logits,
con 986.108 parámetros. Usa entropía cruzada sin pesos, sin aumentos y Adam.
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
