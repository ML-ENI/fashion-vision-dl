# Dataset de imágenes para trabajo local

## Qué ejecutan los notebooks

`01_data_understanding.ipynb` audita el origen completo y compara las taxonomías.
Justifica la selección y muestra una reserva visual que queda excluida de todos
los conjuntos de modelado. `02_data_preparation.ipynb` publica el conjunto local
y comprueba integridad, separación de grupos, estadísticas y lotes.

El objetivo es `product_type_name`. `garment_group_name` mezcla materiales,
secciones comerciales y tipos de prenda. Se seleccionan doce tipos frecuentes
de prendas de vestir; accesorios y ropa interior quedan fuera de este primer
experimento. La categoría heterogénea Top se excluye tras examinar dos productos
reservados que mostraban prendas con capucha y de punto; se incorpora Cardigan.
Se mantienen diferencias difíciles como Sweater/Cardigan. No se afirma
que las clases excluidas tengan mala calidad ni que los nombres de H&M sean una
taxonomía universal.

La selección, con semilla 42, reserva dos grupos por clase para inspección y
limita cada clase a 1.500 imágenes conservando grupos completos. Esto define una
muestra aproximadamente equilibrada: las futuras métricas no estimarán el
rendimiento sobre la prevalencia original de todo el catálogo.

## Filtros y particiones

1. Enlace por article_id conservado como texto; inventario de ausencias y huérfanas.
2. Verificación y decodificación completa de todas las imágenes del origen;
   dimensiones, modo, hash de bytes y hash de píxeles RGB con orientación EXIF.
3. Selección de clases y lado mínimo de 64 píxeles.
4. Conexiones transitivas entre variantes de producto y píxeles idénticos;
   exclusión de componentes con etiquetas contradictorias.
5. Un representante por imagen de píxeles idénticos; reserva visual y cupo por grupos.
6. RGB, proporción conservada, relleno blanco, 224×224 y JPEG calidad 90.
7. Nueva comprobación de duplicados/conflictos después de preparar. Se mantienen
   las conexiones originales aunque se haya eliminado una copia intermedia.
8. División estratificada por clase, aproximadamente 70/15/15 por grupos.
   Las proporciones por imágenes son aproximadas y se publican las reales.
9. Normalización ponderada por píxeles y pesos inversos de clase usando solo train.

No se deducen campañas ni fechas de estas imágenes. Los hashes no detectan todos
los casi duplicados. Se reconoce el posible ruido semántico de las etiquetas.
No se usa test para elegir modelos, resolver errores de clasificación ni ajustar
los aumentos. Las comprobaciones de integridad cubren todos los conjuntos.

## Organización y versiones

```text
config/local_images_v1.json             decisiones editables y versionables
data/raw/source.json                    hash del ZIP y estado de traslado externo
data/raw/images.zip                    RAW preservado, todavía local
data/raw/images/                       originales, todavía locales
data/audit/checkpoints/                 reanudación de la auditoría inicial
data/audit/runs/<id>/                   auditoría, resumen, procedencia, checksums
data/audit/CURRENT.json                 auditoría vigente
data/audit/visual_samples/              26 miniaturas reservadas con hashes
data/processed/images/<prefijo>/<hash>  JPEG compartidos entre versiones
data/processed/runs/<id>/               manifiesto completo e informes
data/processed/CURRENT.json             versión preparada vigente
results/figures/data/                   figuras PNG y SVG de ambos notebooks
```

Se reutiliza el publicador atómico del proyecto ML sin modificar el contrato
compartido `data_contract.py`. Cada ejecución nueva tiene manifiesto, resumen,
etiquetas, pesos de train, normalización, configuración, firma, procedencia,
esquema y checksums. La firma incluye configuración, auditoría, código y versiones
de Pillow, NumPy y pandas. Si coincide, se verifican y reutilizan los resultados.

Los checksums de la publicación cubren las tablas y los informes. Los hashes de
los JPEG están en el manifiesto y se comprueban por separado. Las rutas son
relativas a `data/processed/`. No se copian todas las imágenes para cada versión.
No se eliminan automáticamente versiones ni archivos históricos.

## Uso sin RAW y traslado pendiente

Los loaders leen exclusivamente la publicación preparada. Los momentos de train
se guardan para no volver a necesitar los originales. Las miniaturas reservadas
permiten volver a consultar la galería de auditoría sin el RAW.

Para una nueva selección se necesitará acceder de nuevo al RAW. También hará
falta para experimentar con resoluciones superiores a 224×224. Los futuros
backbones preentrenados deben usar el preprocesamiento propio de sus pesos.

**Todavía no se ha liberado el espacio del RAW.** Falta elegir un destino externo,
copiar el ZIP y comprobar allí su SHA-256 contra `provenance.json` de la auditoría.
Solo después se podrá retirar la copia local. Una carpeta distinta de C: en el
mismo disco no libera espacio; una carpeta de nube sincronizada debe quedar
realmente disponible solo en línea si se pretende liberar el almacenamiento local.

## Ejecución y límites de esta etapa

Desde la raíz de `fashion-vision-dl`:

```powershell
python src/data/audit_images.py --workers 6
python scripts/execute_data_notebooks.py 01_data_understanding.ipynb 02_data_preparation.ipynb
python -m unittest discover -s tests -v
```

La auditoría inicial puede tardar: lee y decodifica todo el origen. Guarda progreso
cada mil archivos. En Windows, un entorno restringido puede necesitar autorización
para crear procesos de Python y conexiones del kernel Jupyter.

Los lotes NumPy permiten verificar la preparación sin instalar PyTorch. Existe
un adaptador opcional `make_torch_loaders`, con `num_workers=0` para Windows;
su comprobación con PyTorch queda pendiente hasta instalar ese framework.
Las pruebas unitarias de independencia del RAW se realizan sobre imágenes de
prueba; los notebooks comprueban manifiestos y archivos del conjunto real.

Esta entrega prepara datos. No ejecuta entrenamiento, selección de arquitecturas
ni evaluación predictiva sobre test. El EDA detallado de entrenamiento y los
baselines corresponden al notebook 03.


## Resultado comprobado el 19 de noviembre de 2026

Publicación: `data/processed/runs/20261119T140252-527f781b`. Los dos notebooks están ejecutados, con tablas y figuras inline.

- Origen auditado: 105.100 imágenes válidas, ninguna ilegible, 442 artículos sin imagen y ninguna imagen huérfana. Se detectaron 206 repeticiones exactas de píxeles en el origen completo.
- Selección final: 18.000 imágenes de doce clases, 1.500 por clase.
- Particiones reales: 12.486 train, 2.898 validation y 2.616 test.
- Imágenes preparadas: 139.260.989 bytes (139,3 MB decimales), RGB de 224×224.
- Exclusiones anteriores al cupo: 365 imágenes con conflicto de etiquetas dentro de un grupo y 128 duplicados exactos en las clases consideradas. Además, 51 filas de grupos reservados para auditoría quedan fuera del modelado.
- Sin cruces entre particiones por producto, componentes originales/preparados o píxeles idénticos; no quedan JPEG sin referencia en el almacén local preparado.
- Dieciséis pruebas unitarias pasan. Lotes NumPy reales comprobados; PyTorch no está instalado.
- El RAW permanece local, pendiente de destino externo. La reducción del conjunto preparado todavía no equivale a espacio RAW liberado.
