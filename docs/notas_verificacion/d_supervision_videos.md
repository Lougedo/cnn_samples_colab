# Frente d) supervision 0.30.5: vídeos de demostración

Verificado el 24-sep-2026 con el venv del repo: Python 3.12.7, supervision 0.30.5, opencv 5.0.0.93, ultralytics 8.4.161 y los pesos `yolo26n.pt` del frente c. La 0.30.5 es la última versión en PyPI a fecha de hoy (publicada el 22-sep-2026).
Máquina: M3 Pro con 11 hilos y otros agentes ocupando la CPU a la vez. **Los tiempos son orientativos y tienen ruido.**
Lo que va sin marca se ha ejecutado aquí o sale de una fuente primaria. 🟡 = inferido o sin verificar ejecutando. 🔴 = no consta.
Scripts y fotogramas de prueba en `verificacion/d/` (`dl.py`, `zonas.py`, `frames/*.jpg`). Los vídeos están en `verificacion/d/videos/`, que git ignora.

---

## 1. Catálogo `VideoAssets` y `download_assets()`

URL base: `https://media.roboflow.com/supervision/video-examples/<fichero>`. El catálogo vive en `supervision/assets/list.py`.

| Miembro | Fichero | MB | md5 del catálogo | ¿Coincide con el servidor? |
|---|---|---|---|---|
| `VEHICLES` | vehicles.mp4 | 35,3 | 8155ff4e4de08cfa25f39de96483f918 | sí (por etag) |
| `MILK_BOTTLING_PLANT` | milk-bottling-plant.mp4 | 2,7 | 9e8fb6e883f842a38b3d34267290bdc7 | sí (por etag) |
| `VEHICLES_2` | vehicles-2.mp4 | 29,8 | 830af6fba21ffbf14867a7fea595937b | sí (por etag) |
| `GROCERY_STORE` | grocery-store.mp4 | 103,3 | 48608fb4a8981f1c2469fa492adeec9c | ❌ **NO**: el servidor sirve 11402e7b861c1980527d3d74cbe3b366 |
| `SUBWAY` | subway.mp4 | 128,5 | 453475750691fb23c56a0cffef089194 | sí (descargado) |
| `MARKET_SQUARE` | market-square.mp4 | 21,3 | 859179bf4a21f80a8baabfdb2ed716dc | sí (descargado) |
| `PEOPLE_WALKING` | people-walking.mp4 | 7,6 | 0574c053c8686c3f1dc0aa3743e45cb9 | sí (descargado) |
| `BEACH` | beach-1.mp4 | 175,4 | 4175d42fec4d450ed081523fd39e0cf8 | sí (por etag) |
| `BASKETBALL` | basketball-1.mp4 | 27,2 | 60d94a3c7c47d16f09d342b088012ecc | sí (por etag) |
| `SKIING` | skiing.mp4 | 8,9 | d30987cbab1bbc5934199cdd1b293119 | sí (por etag) |

«Por etag» quiere decir que el vídeo no se descargó y que se comparó el etag de la cabecera HTTP, que en GCS es el md5 del objeto. También hay `ImageAssets`: `PEOPLE_WALKING` (`people-walking.jpg`) y `SOCCER` (`soccer.jpg`).

Cómo se usa la API (ejecutado):

```python
from supervision.assets import download_assets, VideoAssets
ruta = download_assets(VideoAssets.PEOPLE_WALKING, directory="videos")   # -> str con la ruta absoluta
VideoAssets.list()            # ['vehicles.mp4', ..., 'skiing.mp4']
VideoAssets("subway.mp4")     # -> VideoAssets.SUBWAY ; VideoAssets.SUBWAY.value == 'subway.mp4'
```

- **Firma:** `download_assets(asset_name: Assets | str, directory: str | Path | None = None) -> str`. Acepta el enum o el nombre del fichero como texto.
- **Dónde guarda:**
  - Sin `directory`, en el directorio de trabajo, y devuelve **solo el nombre** (`'people-walking.mp4'`).
  - Con `directory`, crea la carpeta si hace falta y devuelve la **ruta absoluta** (`Path.resolve()`). Funciona con espacios en la ruta.
- **Caché:** si el fichero ya existe, recalcula su md5 (128 MB en 0,23 s) y no descarga. Aun así escribe `[INFO] ... asset download complete.`
- ⚠️ **Si el md5 no coincide, borra el fichero y lo vuelve a descargar.** Un vídeo bajado a mano con el mismo nombre y otro md5 desaparece.
- **Descarga:**
  - `requests` en streaming con `timeout=30`.
  - Escribe en un fichero temporal y lo mueve con `os.replace`, así que la operación es atómica. El fichero queda con permisos 0600, algo que no afecta.
  - Tiempos medidos (dependen de la red): people-walking 0,8 s, market-square 2,8 s, subway 12,3 s.
- **Error de md5 tras descargar:**
  - Escribe `WARNING File corrupted. Re-downloading...` y reintenta **una** vez.
  - Si vuelve a fallar, borra el fichero y lanza `ValueError: Downloaded asset 'grocery-store.mp4' failed MD5 verification.`
  - **Con `GROCERY_STORE` falla siempre en la 0.30.5:** baja 2 × 98,5 MB, tarda unos 20 s y termina en error.
- **Nombre inválido:** lanza `ValueError: Invalid asset. It should be one of the following: vehicles.mp4, ...`, en inglés.
- **Qué imprime:**
  - El logger `supervision.assets.downloader` escribe en stdout `[2026-09-24 00:42:14] [INFO] supervision.assets.downloader - Downloading people-walking.mp4 assets`, en inglés.
  - Además aparece una barra `tqdm.auto` morada, que en Colab se muestra como widget.
  - El log se silencia con `logging.getLogger("supervision.assets.downloader").setLevel(logging.WARNING)`, probado.
  - La barra solo se silencia con `TQDM_DISABLE=1` puesto **antes de que nada importe tqdm**. Si lo pones después no hace efecto, también probado.
  - Recomendación: deja la barra, que para el alumno es útil, y silencia solo el log.

---

## 2. Vídeos candidatos: descargados, medidos con cv2 y mirados fotograma a fotograma

Los recuentos YOLO usan `yolo26n.pt`, `classes=[0]`, `conf=0.35` e `imgsz=640` (por defecto), sobre fotogramas de inicio, mitad y final.

| | people-walking | subway | market-square | grocery-store |
|---|---|---|---|---|
| Resolución | 1920×1080 | **2160×3840 vertical** | **2160×3840 vertical** | 3840×2160 |
| fps / fotogramas | 25 / 341 | 29,97 / 1298 | **60** / 474 | 29,97 / 1002 |
| Duración / tamaño | 13,6 s / 7,6 MB | 43,3 s / 128,5 MB | **7,9 s** / 21,3 MB | 33,4 s / 103,3 MB |
| Códec | H.264 High, yuv420p, sin audio | H.264 High + AAC | H.264 High + AAC | H.264 High + AAC |
| Interior / exterior | Interior: gran vestíbulo de suelo de mármol, en blanco y negro | Interior: andén de metro | Exterior: plaza peatonal con fuente y **terrazas de cafetería** al fondo | Interior: pasillo de supermercado |
| Personas (a ojo) | 40-50 | 20-40 hasta que suben al tren | ~100 | 1 |
| YOLO26n @640 | 17-26 por fotograma (31-39 con `imgsz=1280`) | 6-11 de 0 a 30 s; **0-2 desde el segundo 32** | **1-2** (27-38 con `imgsz=1280`) | 1 |
| Ángulo | Cenital alto, ligeramente oblicuo | Oblicuo alto desde una pasarela | Cenital muy alto | A la altura de los ojos, cámara en el pasillo |
| Caras | Pequeñas y en B/N, poco reconocibles 🟡 | **Las del tercio inferior se ven de frente o de lado**; muchas mascarillas, algunas personas podrían ser reconocibles | No reconocibles | Una persona con mascarilla pasa en primer plano al final |
| ¿Parece cafetería o comercio? | No: vestíbulo o estación | No, pero es aforo de un espacio interior | Terrazas, pero en exterior | Sí (tienda), pero con 1 persona no hay nada que contar |

Más detalles:

- **La orientación vertical es real.** cv2 y ffmpeg dan 2160×3840 y no hay `displaymatrix` ni rotación en los metadatos. cv2 5.0 lo lee derecho.
- **`CAP_PROP_FRAME_COUNT` es exacto** en people-walking y market-square: coincide con el recuento real por `grab()`.
- **Decodificación con cv2 en local (11 hilos, 10 s de vídeo):**
  - people-walking a 640 fps, subway 4K a 162 fps y market-square 4K a 200 fps.
  - En Colab, con 2 vCPU, será bastante menos. 🟡 No medido allí.
- **Rótulos del subway:** están en cirílico («Держите дистанцию»). 🟡 Probablemente Moscú, pero no lo afirmes en clase.
- **La plaza** 🟡 parece Amagertorv (Copenhague). Tampoco lo afirmes.
- **Cronología del subway** (recuento YOLO cada 2 s): `0s:9 2s:8 4s:7 6s:8 8s:7 10s:6 12s:7 14s:6 16s:7 18s:7 20s:8 22s:9 24s:9 26s:11 28s:10 30s:7 32s:2 34s:1 36s:2 38s:1 40s:1 42s:0`.
  - Hacia el segundo 6 llega el tren y entre 12 y 18 baja gente.
  - Entre 20 y 32 sube gente y a partir de 34 el andén queda vacío.
- **people-walking** (cada segundo): entre 18 y 26 personas, estable. El flujo va sobre todo en vertical, hacia la cámara o alejándose.

### Recomendación: segunda demo = `SUBWAY` («andén de metro, interior») empezando en el segundo 22

- Es el único vídeo de interior con gente suficiente y con una **historia de aforo**: el andén se llena y se vacía cuando sube la gente. Además, pasar del aforo de un andén al de una tienda o cafetería es justo el ejercicio de extrapolación de la asignatura, y enlaza con la S6 (Transporte).
- **Hay que empezar en el segundo 22.** Si se empieza en 0, los 10 primeros segundos son un recuento plano de entre 6 y 9 personas.
  - Basta con un `inicio_s` fijo por demo dentro del diccionario de vídeos; no hace falta añadir un control al formulario.
  - Desde el segundo 22 quedan como máximo 21 s, así que `segundos_a_procesar` hay que recortarlo a `duración − inicio`.
- **Descartados:**
  - `GROCERY_STORE`: md5 roto en la 0.30.5, una sola persona y cámara a la altura de los ojos.
  - `MARKET_SQUARE`: con `imgsz=640` casi no detecta a nadie; necesitaría `imgsz=1280` (🟡 unas 4 veces más cómputo) y dura menos que los 10 s por defecto.
- **Aviso de privacidad para el subway:** algunas caras se ven. Refuerza el mensaje de la sección 2.0: solo salen agregados y no se exportan fotogramas.

### Zonas y línea «puerta» recomendadas, en % (validadas ejecutando)

Condiciones de la prueba:
- `RegionCounter` y `ObjectCounter` con `classes=[0]`, `conf=0.35` y el tracker por defecto (`botsort.yaml`).
- `saltar_fotogramas=2`.
- Fotogramas reducidos con `INTER_AREA` a 1280 px de lado mayor: 1280×720 en people-walking y 720×1280 en subway.

> ⚠️ **Corrección del coordinador (24-sep-2026):** la columna «Entradas / salidas» de esta tabla no vale. La prueba
> reutilizó los mismos fotogramas para `RegionCounter` y luego para `ObjectCounter`, y las *solutions* pintan sobre el
> fotograma que reciben, así que el contador de línea vio fotogramas ya anotados. Repetido con `frame.copy()` y ByteTrack
> (`verificacion/coord/lineas.py`): people-walking `0,50; 100,50` → 8/8, `0,65; 100,65` → 10/6; subway desde 22 s, 10 s:
> `62,15; 22,100` → 9/3, `30,60; 100,60` → 0/0. La recomendación de la diagonal en el andén se mantiene; la de evitar el 50 % en la demo 1, no.

| Vídeo / tramo | Zonas | Resultado zonas (inicio / mitad / final, máx.) | Puerta | Entradas / salidas |
|---|---|---|---|---|
| people-walking 0-10 s | izq. `0,0; 50,0; 50,100; 0,100` · der. `50,0; 100,0; 100,100; 50,100` | 10/7 · 12/10 · 9/11 · máx. 14/13 | **`0,65; 100,65`** (horizontal) | 4 / 6 |
| people-walking 0-10 s | (idem) | | `0,50; 100,50` | 0 / 2 → peor, **evitar** |
| subway 22-37 s | las mismas mitades. La izquierda es el lado del tren y la derecha el andén; renombrables «junto al tren» y «andén» | 4/5 · 3/5 · 1/0 · máx. 6/8 | **`62,15; 22,100`** (diagonal, paralela al borde del tren: la cruza quien sube o baja) | 7 / 3 |
| subway 22-37 s | | | `30,60; 100,60` (horizontal) | 0 / 0 → inútil |

- Tiempos (con ruido): people-walking, 125 fotogramas en 7,0 s de `RegionCounter` y 4,7 s de `ObjectCounter`. Subway, 225 fotogramas en 9,4 s y ~10 s.
- 🟡 Qué sentido cuenta como «entrada» o «salida» en una línea diagonal lo documenta el frente c. Aquí solo se comprobó que la línea cuenta.

### Problemas encontrados al validar (los detalla el frente c)

- **Las solutions y el tracking necesitan `shapely` y `lap`**, y no están en el venv.
  - Ultralytics intenta instalarlos solo con `pip` y falla porque el venv es de uv y no tiene pip.
  - Para no tocar el venv, aquí se usó el `--target` del frente c: `PYTHONPATH=verificacion/c/extra`.
  - En Colab la autoinstalación funcionará, pero hace ruido y va en inglés. Pon `pip install -q shapely lap` de forma explícita en la celda de preparación.
- **Ruido de las solutions:**
  - Imprimen `Ultralytics Solutions: ✅ {...config...}` aunque se pase `verbose=False`.
  - Escriben `WARNING ⚠️ No tracks found.` en **cada** fotograma sin personas, que en el final del subway son decenas de líneas.
  - Hay que silenciar las dos cosas.

---

## 3. Licencia y origen de los vídeos

- **El código de supervision es MIT** (`LICENSE.md` del wheel, «Copyright (c) 2022 Roboflow»). **Esa licencia cubre el código, no los vídeos.**
- **La documentación no dice de dónde vienen los vídeos ni con qué licencia.**
  - La página de assets ([supervision.roboflow.com/latest/assets](https://supervision.roboflow.com/latest/assets/)) solo lista nombres y URL, sin origen ni licencia.
  - Tampoco hay créditos en [los ejemplos del repo](https://github.com/roboflow/supervision/tree/develop/examples), en el `METADATA` del paquete ni en los issues.
- **people-walking.mp4 coincide con un clip de Pexels:** «Black And White Video Of People», de **Coverr** ([pexels.com/video/black-and-white-video-of-people-853889](https://www.pexels.com/video/black-and-white-video-of-people-853889/)).
  - La página indica 1920×1080, 25 fps, 14 s y subida el 30-mar-2017.
  - El MP4 da 1920×1080, 25 fps, 13,64 s y `creation_time 2017-03-30T09:54:21Z`.
  - 🟡 La coincidencia se deduce de los metadatos; Roboflow no la declara. La licencia sería la de Pexels.
- **subway, market-square y grocery-store: 🔴 origen no documentado.**
  - Sus `creation_time` son 2021-11-18, 2020-02-05 y 2020-11-18.
  - Llevan `handler_name: L-SMASH Video Handler` y `encoder: AVC Coding`, igual que people-walking. Es un patrón habitual en descargas de Pexels 🟡, pero **no prueba nada**.
- **Texto propuesto para el notebook:**
  > Vídeos de demostración alojados por Roboflow para la librería `supervision` (el código tiene licencia MIT; los vídeos, no necesariamente). Roboflow no publica su origen. El de «personas caminando» coincide con un clip de Pexels (autor: Coverr, licencia Pexels). Los usamos con fines docentes: **licencia según su fuente**.

---

## 4. Nota para el profesor: conseguir un clip de cafetería con licencia libre

**Qué dice la licencia de Pexels** ([pexels.com/license](https://www.pexels.com/license/)):

- **Permite:** uso gratuito, también comercial, sin atribución obligatoria y con modificaciones.
- **No permite:**
  - mostrar a personas identificables de forma ofensiva o dejándolas en mal lugar;
  - vender copias sin modificar;
  - dar a entender que las personas o marcas que aparecen respaldan algo;
  - redistribuir el material en otras plataformas de stock;
  - usarlo como marca.
- **Términos** ([pexels.com/terms-of-service](https://www.pexels.com/terms-of-service/), actualizados el 15-nov-2024): Pexels **no garantiza que existan consentimientos** (*model releases*) de las personas identificables. Esa comprobación es responsabilidad de quien usa el vídeo.
  - Da para una frase en clase: **licencia libre ≠ consentimiento de las personas grabadas**.

**Páginas de Pexels que existen** (comprobado el 24-sep-2026; los vídeos no se han descargado):

| Página | Autor | Duración | Resolución |
|---|---|---|---|
| [a-coffee-shop-restaurant-full-of-customers-3135924](https://www.pexels.com/video/a-coffee-shop-restaurant-full-of-customers-3135924/) | Nazim Zafri | 10 s | 1920×1080, 29,97 fps |
| [people-inside-the-coffee-shop-3135925](https://www.pexels.com/video/people-inside-the-coffee-shop-3135925/) | Nazim Zafri | 10 s | 1920×1080, 29,97 fps |
| [video-of-a-coffee-shop-interior-7652196](https://www.pexels.com/video/video-of-a-coffee-shop-interior-7652196/) | Kindel Media | 12 s | 3840×2160 |

- 🟡 **El ángulo no está verificado a ojo.** El texto de las páginas apunta a cámara a la altura de los ojos o algo elevada, no cenital. Eso implica más oclusiones y cambios de ID que en las demos, así que **el profesor debe mirarlo antes de clase**.
- **Búsqueda:** [pexels.com/search/videos/coffee shop](https://www.pexels.com/search/videos/coffee%20shop/), o los términos «cafe people», «coffee shop customers», «restaurant crowd» y «overhead people».

**Qué clip elegir:**
- Cámara fija, sin zoom ni movimiento, en plano alto u oblicuo y con cuerpos enteros.
- Horizontal, de 10 a 30 s.
- Descargar en **1280×720 o 1920×1080** (no 4K: más MB, más decodificación y más subida). Idealmente menos de ~50 MB.

**Cómo subirlo a Colab** 🟡 (no se puede ejecutar en local; lo verifica el frente e):
- Arrastrarlo al panel «Archivos».
- O bien `from google.colab import files; subidos = files.upload(); ruta = next(iter(subidos))`, que lo guarda en `/content/`.
- El almacenamiento de la sesión es efímero: si la sesión se reinicia, hay que volver a subirlo. Alternativa: montar Drive.

---

## 5. ¿`pip install supervision` rompe opencv o numpy en Colab?

- **Dependencias de la 0.30.5** (`Requires-Dist`): `av>=14.2`, `defusedxml>=0.7.1`, `matplotlib>=3.6`, `numpy>=1.21.2`, `pillow>=9.4`, `pydeprecate>=0.9,<0.12`, `pyyaml>=5.3`, `requests>=2.26`, `scipy>=1.10`, `tqdm>=4.62.3`. Python >=3.10.
- **No depende de opencv.** En la 0.30.x cv2 es opcional (`supervision/_cv2/__init__.py`: `try: import cv2 except (ImportError, OSError)`), así que **no puede tocar opencv**.
- numpy no tiene cota superior, así que tampoco lo baja de versión.
- **Resolución en seco** con `uv pip compile` para Linux x86_64 y Python 3.12:
  - Pines tipo Colab 🟡 (supuestos, no leídos de un Colab real): numpy 2.0.2, opencv-python/-headless 4.12.0.88, matplotlib 3.10.0, scipy 1.16.1 y pillow 11.3.0, más ultralytics 8.4.161.
  - Resultado: resuelve **sin mover ninguno** y solo añade `supervision 0.30.5`, `av 18.1.0` y `pydeprecate 0.11.0`.
- **Coste:** el wheel de PyAV `av-18.1.0-cp311-abi3-manylinux_2_28_x86_64.whl` pesa **35,8 MB**. Es lo más pesado de instalar supervision solo para bajar vídeos.
- **`import supervision`:** 0 bytes en stdout y stderr, incluso con `-W default`. Tarda 0,73 s e importa cv2 si está disponible.

---

## Recomendaciones para quien construya el notebook 2

1. Instala `supervision==0.30.5` (o `>=0.30,<0.31`) junto con `shapely` y `lap`. El comportamiento de `directory=` y de la verificación md5 se ha comprobado solo en la 0.30.5.
2. Diccionario de demos (la clave es el texto que ve el alumno):
   ```python
   DEMOS = {
       "Demo 1: personas caminando (vestíbulo)": dict(fichero="people-walking.mp4", inicio_s=0,  puerta="0,65; 100,65"),
       "Demo 2: andén de metro (interior)":      dict(fichero="subway.mp4",         inicio_s=22, puerta="62,15; 22,100"),
   }
   ZONAS_POR_DEFECTO = "0,0; 50,0; 50,100; 0,100 | 50,0; 100,0; 100,100; 50,100"   # izquierda | derecha
   ```
3. Descarga con un error en castellano. El respaldo por URL directa es opcional y cuesta una línea:
   ```python
   try:
       ruta = download_assets(fichero, directory="videos")
   except Exception:
       # plan B: urllib.request.urlretrieve(f"https://media.roboflow.com/supervision/video-examples/{fichero}", f"videos/{fichero}")
       raise RuntimeError("No se pudo descargar el vídeo de demostración. Reintenta la celda o elige «subir mi vídeo».")
   ```
   Si usas el plan B, no vuelvas a llamar después a `download_assets` sobre ese fichero: si el md5 no coincide, lo borra.
4. No ofrezcas `GROCERY_STORE` (md5 roto) ni `MARKET_SQUARE` (con `imgsz=640` no detecta a nadie).
5. Lleva el vídeo a `inicio_s` con `cap.set(cv2.CAP_PROP_POS_FRAMES, int(inicio_s*fps))` y recorta `segundos_a_procesar` a `duración − inicio_s`. El primer fotograma que se enseña debe ser el de `inicio_s`.
6. Reduce cada fotograma a ≤1280 px de lado mayor con `cv2.INTER_AREA` **antes** de pasarlo a las solutions y de escribir el vídeo. El subway en 4K vertical queda en 720×1280.
   - Para verlo inline, fija el ancho del reproductor en ~360-400 px. Si no, el vídeo vertical ocupa toda la pantalla del proyector.
7. Trabaja con porcentajes y convierte a píxeles con el tamaño **reducido**, no con el original.
8. Silencia el logger de supervision, el volcado de configuración de las solutions y `No tracks found`.
9. Textos visibles: la licencia de los vídeos es «según su fuente» (sección 3) y el aviso de caras visibles va en el subway.
