# Decisiones

Cada decisión tomada ante una ambigüedad del brief, con su motivo. Las pruebas que las sostienen
están en `docs/notas_verificacion/`. Fecha de trabajo: 24-sep-2026.

## Repositorio y entorno

| # | Decisión | Motivo |
|---|---|---|
| D01 | El repositorio vive en `Presentaciones/IAaSP/sesiones/S7/cnn-sesion-final/` y es un repositorio git propio. | La carpeta de trabajo (`MASTER IA PROFESOR`) no era un repositorio y contiene cientos de MB de material de otras asignaturas (un `Flowise.zip` de 468 MB, PPTX, PDF). Inicializar git ahí mezclaba todo. La asignatura del encargo es IASP y su última sesión es la S7 (24-sep), así que el repo va con el resto de sesiones de IASP. |
| D02 | Verificación local con Python 3.12 (`.venv`, uv) **y** prueba de humo del NB1 con Python 3.13 y las versiones exactas de Colab (`.venv-colab`). | El brief pide emular 3.11–3.12, pero Colab pasó a Python 3.13.15 el 25-ago-2026 (colabtools#6081) con TF 2.21.0, Keras 3.13.2, numpy 2.1.3 y pandas 2.2.3 (googlecolab/backend-info, 23-sep-2026). Se cumple el brief y además se prueba contra lo que usarán los alumnos. El NB2 no admite ese espejo en macOS: ultralytics 8.4.161 excluye numpy 2.0–2.3.4 solo en macOS. |
| D03 | Versiones fijas en las celdas de instalación: `medmnist==3.0.2`, `ultralytics==8.4.161`, `supervision==0.30.5`; `lap>=0.5.12` y `shapely>=2.0.0` explícitos. Las versiones que ya trae Colab se fijan con un archivo de restricciones para que pip no las actualice. | Son las versiones contra las que se verificaron las APIs. Las *solutions* de Ultralytics cambian a menudo. Sin `lap` y `shapely` preinstalados, Ultralytics los instala a mitad de clase. Actualizar numpy o TF en Colab obliga a reiniciar la sesión. |
| D04 | La instalación comprueba primero con `importlib.util.find_spec` y solo llama a pip si falta algo. | En Colab evita reinstalar; en el `.venv` local (uv, sin pip) evita un fallo. |
| D05 | `scripts/verificar.py` simula el formulario reescribiendo la línea `nombre = valor  # @param` antes de ejecutar con nbclient. | Es lo que hace Colab al tocar un control. Papermill inyecta una celda aparte y solo admite una celda de parámetros; aquí hay parámetros en seis celdas. |

## Notebook 1

| # | Decisión | Motivo |
|---|---|---|
| D10 | Convoluciones con `padding="valid"`. | Es el padding de Tiny VGG en CNN Explainer, que los alumnos acaban de ver (64→62→60→30…). Además hace real el caso del brief «la imagen se queda en 0×0»: con 28 px y pooling caben 3 bloques con kernel 3 y 2 con kernel 5. |
| D11 | Un bloque = convolución [+ BatchNorm] + ReLU [+ max-pooling 2×2]. Tras aplanar: [densa ReLU] → [dropout] → neurona de salida (`logit`) → sigmoide. | La arquitectura más corta que cubre todos los controles del formulario. El `logit` separado permite Grad-CAM para las dos clases (+logit / −logit). |
| D12 | Diagrama de arquitectura propio en matplotlib; no se usa visualkeras. | visualkeras 0.2.0 (última en PyPI) falla con Keras 3 (`'InputLayer' object has no attribute 'output_shape'`), su vista es pseudo-3D (el brief pide «nada de 3D gratuito») y su leyenda sale en inglés. El brief prevé este caso. |
| D13 | `BatchNormalization(momentum=0.9)`. | Con el valor por defecto (0,99) la red colapsa a «todo Neumonía» en 8 épocas en 3 de 3 semillas; los alumnos concluirían que BatchNorm estropea la red. |
| D14 | Descarga propia de los `.npz` (urllib, timeout 60 s, MD5, archivo temporal) y lectura con `medmnist(..., download=False)`. Espejo solo para 28 px: Hugging Face `albertvillanova/medmnist-v2`, revisión fijada `f6dd981c…`. | La descarga de medmnist usa `urlopen` sin timeout: una descarga de Zenodo se quedó colgada más de 10 min en la fase 1. El espejo no es oficial (MedMNIST solo reconoce Zenodo), pero es idéntico byte a byte (MD5 verificado). No hay espejo del archivo de 64 px: si Zenodo falla, el mensaje propone volver a 28 px. |
| D15 | Lote 128, Adam 1e-3, semilla 42 en cada entrenamiento. | Mismas métricas que lote 64 dentro del ruido entre semillas y la mitad de pasos, que es lo que cuenta en la CPU de Colab. Con la semilla dentro del entrenamiento, re-entrenar da exactamente lo mismo en CPU. |
| D16 | Umbrales antes de entrenar: más de 20 M de parámetros se bloquea sin estimar; si no, tiempo estimado con un mini-cronometraje (×1,2): hasta 90 s nada, 90–600 s aviso, más de 600 s bloqueo. | La estimación se mide en el hardware real (Colab o local). El peor caso del formulario (103 M parámetros) tardaría 30–50 min; no cabe en un bloque de 25 min. |
| D17 | Plan B entrena 7 configuraciones: por defecto, A con 1 y con 3 bloques, B sin pooling, C con dropout 0 y 0,5, y D con compensación. | Los retos A y C comparan dos variantes cada uno; B y D se comparan con la configuración por defecto. «Las 4 configuraciones de referencia de los retos» no alcanza para responder a los cuatro retos. |
| D18 | Grad-CAM sobre la salida de la última ReLU; si el mapa de la clase «Normal» sale vacío, se explica en lugar de mostrar una imagen en blanco, y se prefieren ejemplos con mapa. | Definición estándar (activaciones tras la ReLU). Con la red por defecto el mapa de «Normal» sale vacío en torno a la mitad de los casos. |
| D19 | CSV con `;`, coma decimal y UTF-8 con BOM. La línea resumen para la tabla común va separada por tabuladores. | Abre bien en Excel y Sheets en español. Los tabuladores reparten la línea en columnas al pegarla en una hoja. |

## Notebook 2

| # | Decisión | Motivo |
|---|---|---|
| D30 | Segunda demo: `subway.mp4` (andén de metro, interior), empezando en el segundo 22. No se ofrece `grocery-store.mp4`. | Es el único vídeo de interior con varias personas y con una historia de ocupación (el andén se vacía al subir al tren). `grocery-store` tiene una sola persona y su MD5 en supervision 0.30.5 está roto (la descarga falla siempre tras bajar 200 MB). `market-square` es exterior y YOLO26n apenas detecta a nadie a 640 px. |
| D31 | Todos los fotogramas se reducen a 1280 px de lado largo antes de procesar. | Las zonas en % valen igual, el vídeo de salida cumple el máximo de 720p y el proceso es más rápido (el andén es 4K vertical). |
| D32 | `linea_puerta = "auto"` por defecto: usa la línea recomendada para cada demo (personas caminando y vídeo propio: horizontal al 50 %; andén: diagonal junto al borde del tren, `62,15; 22,100`). | Medido con fotogramas limpios (10 s, salto 2, confianza 0,35): en la demo 1 la línea al 50 % da 8 entradas y 8 salidas. En el andén, una horizontal al 60 % no cuenta a nadie (0/0) y la diagonal da 9/3. Una línea fija no sirve para las dos demos. Una medición previa de la fase 1 (4/6 y 0/2) se descartó: reutilizaba fotogramas ya pintados por RegionCounter. |
| D33 | Casilla `invertir_sentido`. | Ultralytics decide el sentido por la geometría (línea horizontal: entrada = hacia abajo; vertical: hacia la derecha). En un vídeo propio la puerta puede estar al revés. |
| D34 | Tracker ByteTrack en lugar del BoT-SORT por defecto; `imgsz=640`. | Mismos recuentos con cámara fija, algo más rápido y sin compensación de movimiento. A 480 px se pierden personas lejanas (15 frente a 20 en el mismo fotograma). |
| D35 | Telemetría de Ultralytics desactivada en el notebook. | Coherente con el aviso de privacidad. |
| D36 | Vídeo inline con `<video>` HTML y base64, tras recodificar a H.264 (`libx264`, `yuv420p`, CRF 28, `+faststart`). ffmpeg del sistema o, si no hay, el de `imageio-ffmpeg`. | Es lo que hace `mediapy` de Google para Colab y mantiene el texto alternativo en castellano. Colab trae ffmpeg 6.1.1 con libx264; el Mac de verificación no tiene ffmpeg del sistema. |

## Documentación y licencias

| # | Decisión | Motivo |
|---|---|---|
| D50 | No se añade un archivo de licencia al repositorio. Se deja como pendiente para el profesor: si se publica, lo prudente es publicar el código abierto y declarar AGPL-3.0 al menos para el NB2 y su generador. | Ultralytics es AGPL-3.0 y considera que el software que lo usa queda cubierto. Es una interpretación, no asesoramiento jurídico, y la decisión de publicar es del profesor. |
| D51 | CNN Explainer solo lo abre el profesor, compartiendo pantalla. | La web sigue cargando un script de `polyfill.io`, dominio comprometido en junio de 2024; no se ha comprobado qué sirve hoy. |
