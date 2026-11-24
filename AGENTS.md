# Preferencias de trabajo de la estudiante

- No subir cambios pendientes de ejecución o validación. Preparar los cambios
  en local, dejar que la estudiante los ejecute, corregir los errores y publicar
  solo cuando la versión correspondiente esté ejecutada y validada. Un diff
  revisado o resultados de una versión anterior no sustituyen esa validación.
- Editar y explicar el código sin lanzar por iniciativa propia preparaciones,
  notebooks, entrenamientos ni pruebas. La estudiante prefiere ejecutarlos ella.
  Solo ejecutar código del proyecto si lo pide o si es estrictamente necesario;
  explicar antes el motivo concreto de esa excepción. Lectura, edición y gestión
  autorizada de Git no equivalen a ejecutar el experimento.
- No dejar trabajos de ejecución en segundo plano ni colas automáticas sin una
  petición explícita. Al cambiar datos o protocolo, retirar salidas obsoletas y
  marcar los notebooks como pendientes; no inventar resultados ni contadores.
- Mantener una sola propuesta activa de datos: definir las prendas, aplicar
  controles comunes y conservar todas las fotos elegibles. No fijar cupos por
  clase ni un total deseado. Una eventual muestra de train requiere coste medido
  y justificación; no se recortan validation/test para equilibrar clases.
- Alcance acordado: todos los tipos de Garment Upper body, Garment Lower body y
  Garment Full body del catálogo H&M (26 etiquetas actualmente), incluida Top.
  Las familias delimitan el problema; la predicción es product_type_name.
  No volver a elegir etiquetas sueltas por conveniencia. Ampliar o reducir el
  alcance gradualmente con la estudiante, conservando todas las fotos válidas.
- Mantener juntos los productos relacionados y respetar su exposición previa:
  datos ya utilizados en desarrollo no deben convertirse en un supuesto test nuevo.
- Escribir explicaciones sencillas y autoexplicativas en español. Conservar las
  mejoras de redacción de la estudiante al cambiar el contenido técnico.
- Usar principalmente azul ETSIT y naranja con los módulos visuales existentes.
  Introducir más colores cuando hagan falta más grupos y mantener la coherencia
  de las particiones. Tablas legibles, números alineados y precisión justificada.
- El filtro visual de detalles se aplica en 02 después de asignar particiones,
  a TODAS las fotos elegibles: train, validation y test, por petición expresa.
  CLIP usa la misma regla y textos fijados para las tres particiones;
  se excluyen los detalles unánimes y se conservan dudas. Las correcciones
  manuales ligadas al hash del original prevalecen. No borrar fotos ni reasignar
  productos. Distinguir análisis automático completo de validación humana.
  Las fotos conservadas no cambian de partición. La evaluación se referirá al
  conjunto filtrado; no ajustar este filtro según los resultados de test.
  Todas las fotos se normalizan, pero medias y desviaciones se calculan solo
  con train y se reutilizan para validation/test.
- El 04 debe explicar cada paso con la claridad de 02, frases cortas y una
  indicación de cómo leer cada resultado. Mostrar fotos pequeñas, figuras con
  ancho limitado y matrices separadas. El código largo puede ser desplegable.

- Run All debe cargar resultados existentes por defecto. CLIP y entrenamientos
  nuevos requieren activar su bandera explícita; nunca arrancan por faltar una
  caché. Separar la lectura de puntuaciones de los cálculos posteriores. Mantener
  la copia compartible results/data_preparation y los originales inmutables.
