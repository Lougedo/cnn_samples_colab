# Frente c) Ultralytics 8.4.161: YOLO26n + RegionCounter / ObjectCounter + tracking + ffmpeg

Fecha: 24-sep-2026. Entorno: `.venv` (Python 3.12.7, macOS arm64, M3 Pro con 11 núcleos), ultralytics 8.4.161, torch 2.14.0, opencv 5.0.0, numpy 2.5.3.
Todo lo que no lleva 🟡 lo he comprobado **ejecutando código** o leyendo el fuente instalado (`.venv/lib/python3.12/site-packages/ultralytics/…`).
Scripts y salidas en `verificacion/c/`, que está en .gitignore. El prototipo completo de NB2 §2.3–2.7, verificado, es `verificacion/c/snippets_nb2.py`.
Vídeo de prueba: `people-walking.mp4` (descargado con `curl` de media.roboflow.com): 1920×1080, 25 fps, 341 fotogramas (13,6 s) y vista cenital.

> **Tiempos con ruido.** Mientras medía, otros agentes usaban la CPU (carga media entre 6 y 17 sobre 11 núcleos). Las cifras de «1 hilo / tiempo de CPU» son las más limpias.

---

## 0. Lo imprescindible para el builder (resumen)

1. **Pesos: `yolo26n.pt`** (5 544 453 bytes, 5,3 MB), descargados de `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt`. Pasa **ruta explícita**: `YOLO("pesos/yolo26n.pt")` crea la carpeta y descarga ahí. Con el nombre pelado, el fichero cae en el **cwd**.
2. **Faltan dos dependencias que ultralytics instala sola a mitad de ejecución**: `shapely>=2.0.0` (al construir cualquier *solution*) y `lap>=0.5.12` (al llamar al tracker por primera vez). En el venv **no están**. En la celda de preparación: `pip install "ultralytics==8.4.161" "lap>=0.5.12" "shapely>=2.0.0"` y `os.environ["YOLO_AUTOINSTALL"]="False"`.
3. **Silenciar**: `os.environ["YOLO_VERBOSE"]="False"` **antes** de `import ultralytics`. Además, `LOGGER.setLevel(logging.ERROR)` después del import, para que siga funcionando si alguien reejecuta la celda. `verbose=False` en la solution **no basta**: el banner `Ultralytics Solutions: ✅ {…}` y los `WARNING ⚠️ No tracks found.` se siguen imprimiendo.
4. **`region_counts` es la ocupación del fotograma actual, no un acumulado.** `in_count`/`out_count` sí se acumulan. Ambos usan el **centro de la caja**.
5. **Tracker: `tracker="bytetrack.yaml"`.** Por defecto se usa `botsort.yaml`, que añade compensación de movimiento de cámara (sparseOptFlow): unos 3,5 ms más por fotograma y un arranque más lento. Con cámara fija no aporta nada, y los recuentos salen idénticos.
6. **Coordenadas de zonas y línea: siempre `int`.** `ObjectCounter` con puntos float revienta en `cv2.circle`. `RegionCounter` los tolera, pero conviene no depender de eso.
7. **Las solutions pintan sobre el fotograma que reciben** (`r.plot_im is frame` → True). Si el mismo fotograma pasa por dos solutions, dale `frame.copy()` a cada una.
8. **ffmpeg**: cv2 `mp4v` → `libx264 yuv420p` + `-movflags +faststart` + `-an` + `scale=-2:'min(720,trunc(ih/2)*2)'`. Verificado: 10 s anotados a 720p ocupan unos 1,6 MB.
9. **Tiempo**: 10 s a 25 fps con salto 2 son 125 llamadas por pasada. En el M3 con 1 hilo, cada llamada tarda unos 44 ms (imgsz 640). Dos pasadas más la recodificación suman entre 25 y 32 s de reloj. Colab estimado: 60-90 s 🟡.

---

## 1. Modelo YOLO26 nano

| Dato | Valor | Cómo |
|---|---|---|
| Nombre | `yolo26n.pt` | `GITHUB_ASSETS_NAMES` en `utils/downloads.py` y descarga real |
| URL | `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt` | log de descarga |
| Tamaño | 5 544 453 B (5,3 MB) | `stat` |
| Clases | 80 COCO, `model.names[0] == 'person'` | ejecutado |
| Ficha oficial | mAP50-95 40,9 · 2,4 M parámetros · 5,5 GFLOPs · CPU ONNX 38,9 ms en «Intel Xeon CPU @ 2.00 GHz» · publicado en enero de 2026 | [docs YOLO26](https://docs.ultralytics.com/models/yolo26/) |

**Dónde cae el fichero** (`attempt_download_asset`): (1) si la ruta existe, la usa; (2) si no, busca en `SETTINGS["weights_dir"]/nombre`; (3) si no está en ninguna, descarga **a la ruta indicada, relativa al cwd**. Verificado:
- `YOLO("yolo26n.pt")` → `./yolo26n.pt` en el cwd.
- `YOLO("pesos/yolo26n.pt")` → crea `pesos/` y descarga dentro. **Recomendado**: en Colab, `/content/pesos/yolo26n.pt`; en local, una carpeta ignorada por git (`*.pt` ya está en `.gitignore`).
- `settings["weights_dir"]` solo se consulta para **buscar**, nunca para descargar. Aquí vale `<raíz git>/weights`, porque el repo ya es git. No lo toques: con una ruta explícita da igual.
- Al primer `import ultralytics` se crea un `settings.json` en la carpeta de usuario (`~/Library/Application Support/Ultralytics/` en Mac; en Colab, `~/.config/Ultralytics/` 🟡). Mientras no se pone `YOLO_VERBOSE=False`, imprime «Creating new Ultralytics Settings v0.0.8 file ✅». Se puede redirigir con `YOLO_CONFIG_DIR`, pero hace falta que la carpeta exista y se pueda escribir; si no, cae en `/tmp/Ultralytics`.
- La **barra de progreso de la descarga no se silencia** con `YOLO_VERBOSE=False`. Descarga los pesos en la celda de preparación, que es donde tiene sentido verla.

**¿End-to-end / sin NMS?** El modelo trae las dos cabezas (`one2one_cv2` presente, `yaml end2end: True`), pero **en 8.4.161 la predicción usa por defecto la cabeza uno-a-muchos con NMS**. Lo he verificado con `predictor.model.end2end == False` y lo confirma la documentación: «Prediction and validation default to the one-to-many head for accuracy» ([docs YOLO26](https://docs.ultralytics.com/models/yolo26/)). La inferencia sin NMS solo se activa con `model.predict(..., nms=False)`. Las solutions no pasan `nms`, así que usan NMS.
En el primer fotograma a 1920×1080 con conf 0,35 salen 21 personas con la cabeza por defecto (igual con `nms=True`) y 16 con `nms=False`: las confianzas cambian entre cabezas. **Para el texto de 2.3:** no digas «YOLO26 no usa NMS». Lo correcto es «puede funcionar sin NMS; por defecto se usa la variante con NMS».

**`predict(classes=[0], conf=…)` funciona como siempre.** Con `classes=[0]` solo vuelve cls 0. Sin filtro y con conf 0,05 aparece también la clase 24 (mochila). La entrada 1280×720 se letterboxea a **384×640** (log: `0: 384x640 17 persons`). El log de las solutions dice «640x640», pero es solo el `imgsz` nominal.

---

## 2. Solutions: constructor, formatos y resultados

### 2.1 Argumentos aceptados (`solutions/config.py`, dataclass `SolutionConfig`)
Cualquier otro kwarg lanza `ValueError: <kwarg> is not a valid solution argument` (verificado con `regiones=`). Los relevantes, con su valor por defecto:
`model=None` (→ `"yolo26n.pt"`, **descarga en el cwd**), `region=None`, `classes=None`, `conf=0.25`, `iou=0.7`, `imgsz=640`, `tracker="botsort.yaml"`, `device=None` (CUDA si hay, si no CPU; **no elige MPS** por su cuenta), `max_det=300`, `line_width=2`, `show=False`, `show_conf=True`, `show_labels=True`, `show_boxes=True`, `show_in=True`, `show_out=True`, `verbose=True`, `quantize=None` (`half` está obsoleto: si lo usas, avisa y lo convierte).
Trackers disponibles en `cfg/trackers/`: `botsort`, `bytetrack`, `deepocsort`, `fasttrack`, `ocsort` y `tracktrack`.

### 2.2 Formato de `region`
- **RegionCounter**: `dict {nombre: [(x,y), …]}` para varias zonas, o una lista de puntos, que se convierte en `{"Region#01": pts}`. Cada zona necesita **3 puntos o más**; con 2 lanza `ValueError: RegionCounter requires regions with at least 3 points…` (verificado). Pixeles absolutos. Si las zonas se solapan, una persona cuenta en todas las que contienen su centro (fuente).
- **ObjectCounter**: **2 puntos = línea** y 3 o más = polígono. Con `region=None` usa `[(10,200),(540,200),(540,180),(10,180)]`: pásala siempre.
- **RegionCounter no dibuja el nombre de la zona**, solo el número, en el centroide del polígono. El nombre hay que pintarlo aparte con `cv2.putText`, que solo admite ASCII: «barra» y «mesas» valen; con tildes saldría `??`.

### 2.3 Llamada y `SolutionResults`
`r = solucion(frame_bgr)` (`__call__` → `process`). Campos útiles:
- `r.plot_im`: fotograma anotado, BGR. **Es el mismo objeto que `frame`**: pinta encima.
- `r.total_tracks`: IDs presentes en **este** fotograma.
- `r.speed`: `{"track": ms, "solution": ms}`.
- RegionCounter: `r.region_counts = {nombre: n}` con **ocupación actual**. Se recalcula a cero en cada fotograma, incluidas las zonas vacías (verificado con un fotograma en blanco: `{'izquierda': 0, 'derecha': 0}`). La documentación dice lo mismo: «the counts reset before the next frame» ([guía region-counting](https://docs.ultralytics.com/guides/region-counting/)).
- ObjectCounter: `r.in_count` y `r.out_count` (**acumulados**) y `r.classwise_count = {nombre_clase: {"IN": n, "OUT": n}}`. La clave es `names[cls]`: si renombras la clase, cambia.

### 2.4 Criterio de conteo (fuente leído y comprobado con un recuento independiente)
- **Centro de la caja** `((x0+x1)/2, (y0+y1)/2)`, no los pies. Con cámara en perspectiva, las zonas de suelo tienen que cubrir la altura del torso.
- **Línea en ObjectCounter**: se cuenta cuando el segmento *centro anterior → centro actual* del mismo ID corta el **segmento** de la línea, no una recta infinita.
  - La orientación la decide `|dx| < |dy|` entre los extremos de la línea. **Vertical**: ENTRADA si el centro se mueve a la **derecha** (x crece) y SALIDA si va a la izquierda. **Horizontal**: ENTRADA si **baja** (y crece) y SALIDA si sube. No influye el orden de los puntos, y una diagonal exacta de 45° se trata como horizontal.
  - **Cada ID cuenta una sola vez** (`counted_ids`): quien cruza y vuelve no suma. Solo se olvida cuando el tracker retira el ID. Un **cambio de ID** hace que la misma persona cuente dos veces.
  - La documentación no dice qué sentido es «IN» ([guía object-counting](https://docs.ultralytics.com/guides/object-counting/)). Esto sale del fuente. Sugerencia: una casilla `invertir_sentido` en el formulario que intercambie las etiquetas al mostrarlas.
- **Recuento con línea horizontal al 50 % en people-walking** (10 s, conf 0,35, imgsz 640, frames reducidos a 1280×720):

| tracker | salto 1 | salto 2 | salto 3 | salto 4 | salto 5 |
|---|---|---|---|---|---|
| bytetrack (entradas/salidas) | 9/8 | **8/8** | 8/8 | 8/8 | 8/8 |
| botsort | 9/8 | 8/8 | 8/8 | — | — |
| IDs únicos (bytetrack) | 54 | 51 | 47 | 41 | 41 |

  Con línea **vertical** al 50 %: 1/1 (la gente camina sobre todo en vertical). **Comprobación independiente**: con los centros por ID recogidos de `oc.boxes`/`oc.track_ids`, el primer cruce de y=359 da 8/8, igual que `ObjectCounter`.
- **Zonas izquierda/derecha al 50 %** (máximo / media de personas): salto 1 → 14/10,97 y 13/9,64; salto 2 → 14/11,0 y 13/9,5; salto 5 → 13/10,6 y 12/9,0. **Saltar fotogramas apenas mueve los agregados.** El tracker no sabe que se saltan: `track_buffer=30` cuenta **llamadas**, así que con salto 2 a 25 fps un ID perdido aguanta 2,4 s reales.
- Un detalle para 2.3/2.6: con ByteTrack, una detección por debajo de `new_track_thresh=0.25` no abre un ID nuevo. Si `conf` baja de 0,25 en las solutions, esas cajas solo sirven para mantener IDs que ya existen. Con 0,35 no afecta.

---

## 3. IDs y cajas por fotograma (para el tiempo de permanencia sin otra inferencia)

Después de cada `r = rc(frame)`, en el propio objeto solution:
- `rc.boxes`: tensor CPU `(N,4)` en formato xyxy, en píxeles del fotograma. Si no hay tracks, es **una lista vacía `[]`**, no un tensor.
- `rc.track_ids`: `list[int]`. `rc.clss`: `list[float]` (0.0). `rc.confs`: `list[float]`.
- `rc.counting_regions[i]["prepared_polygon"]` y `rc.Point` (shapely) permiten aplicar **exactamente el mismo criterio** que el contador.
- `rc.track_history` solo lo llena ObjectCounter (hasta 30 puntos por ID). RegionCounter no guarda historial.
Medido en la misma pasada de 2.5 (10 s, salto 2): 26 IDs pasan por «izquierda» (mediana 3,5 s, máximo 9,8 s) y 27 por «derecha» (mediana 3,1 s, máximo 10,0 s). El máximo está limitado por la duración del clip. Hay que decirlo: una permanencia truncada en los bordes del clip no es una permanencia real.

---

## 4. Logs, avisos y autoinstalación

| Qué sale | `verbose=False` | `YOLO_VERBOSE=False` / `LOGGER.setLevel(ERROR)` |
|---|---|---|
| Banner `Ultralytics Solutions: ✅ {…dict entero…}` al construir | **sigue** | desaparece |
| Por fotograma: `0: 640x640 18.0ms, 17 person` + `Speed: …` | desaparece | desaparece |
| `WARNING ⚠️ No tracks found.` (fotograma sin personas) | **sigue** | desaparece |
| `WARNING … Environment does not support cv2.imshow()` (en Colab/Linux sin pantalla, por `check_imshow(warn=True)` en cada `__init__`) | **sigue** 🟡 en Colab: sale de leer el fuente (`assert not IS_COLAB`) | desaparece |
| Barra de progreso de la descarga de pesos | sigue | **sigue** |
| «Creating new Ultralytics Settings…» (primer import) | — | desaparece solo si la variable de entorno se fija antes del import |

Receta verificada:
```python
import os, logging
os.environ["YOLO_VERBOSE"] = "False"      # antes del import
os.environ["YOLO_AUTOINSTALL"] = "False"  # nada de pip a mitad de clase
from ultralytics import YOLO, solutions
from ultralytics.utils import LOGGER
LOGGER.setLevel(logging.ERROR)            # idempotente, por si la celda se reejecuta
```
**Autoinstalación** (`check_requirements`, que usa `uv pip install` si hay `uv` y si no, `pip`):
- `BaseSolution.__init__` → `check_requirements("shapely>=2.0.0")`. Sin shapely y con `YOLO_AUTOINSTALL=False` → `ModuleNotFoundError: No module named 'shapely'` (verificado).
- `trackers/utils/matching.py` → `check_requirements("lap>=0.5.12")` en la primera llamada a `track` (tanto bytetrack como botsort). Sin lap → `ModuleNotFoundError: No module named 'lap'` (verificado).
- Con autoinstalación activa, instala en caliente y avisa «Restart runtime or rerun command for updates to take effect» (fuente). En clase: **preinstalar**.
- **Preinstalar en NB2**: `ultralytics==8.4.161`, `lap>=0.5.12`, `shapely>=2.0.0` y, para ffmpeg en local, `imageio-ffmpeg`. Colab ya trae torch, opencv, pandas y matplotlib. Si `shapely` también viene de serie 🟡, instalarlo no cuesta nada.
- **Telemetría** (relevante para el aviso de privacidad): `SETTINGS["sync"]=True` por defecto. Ultralytics envía eventos de uso anónimos (modo, tarea, nombre del modelo, versiones, CPU; **no imágenes**, según `utils/events.py`). `events.enabled` se calcula **al importar**, así que `settings.update({"sync": False})` no lo apaga en la sesión en curso (verificado). Para apagarlo: `from ultralytics.utils.events import events; events.enabled = False`. Opcional, pero coherente con «del notebook solo salen agregados».

---

## 5. Velocidad en CPU (M3 Pro; con ruido, ver aviso arriba)

**Hilos:** `select_device()` hace `torch.set_num_threads(NUM_THREADS)` en la **primera** inferencia de cada modelo o solution, con `NUM_THREADS = min(8, max(1, os.cpu_count()-1))`. Por eso un `torch.set_num_threads(2)` previo **se pierde**: hay que fijarlo después de la primera llamada. **En Colab gratuito (2 vCPU) ultralytics usa 1 hilo por defecto.** El proxy correcto de Colab es 1 hilo, no 2.

Por fotograma, con entrada 1280×720 (mediana):

| Configuración | predict | RegionCounter (bytetrack) | ObjectCounter (bytetrack) | RegionCounter (botsort) |
|---|---|---|---|---|
| 1 hilo, imgsz 640 (pared / CPU) | 45,2 / 44,5 ms | 43,6 / 43,5 | 43,5 / 43,4 | 47,2 / 47,2 |
| 1 hilo, imgsz 480 | 24,5 / 24,5 ms | 26,1 / 26,0 | 26,1 / 26,0 | 29,6 / 29,6 |
| 8 hilos, imgsz 640 | 30-35 ms | 29-39 ms | ~30-34 ms | 37-45 ms |
| 8 hilos, imgsz 480 | 18-21 ms | 22-24 ms | — | — |

- La red nano apenas escala con hilos a este tamaño: con 8 hilos va más o menos igual que con 1 cuando la máquina está cargada. La medida con 2 hilos salió contaminada (32,8 ms para predict y 40,1 para RegionCounter, casi lo mismo que con 8).
- **Primera llamada** de cada solution: entre 1,2 y 2,3 s (preparar el predictor, fusionar capas, iniciar el tracker). Hay que contarla en el presupuesto de tiempo.
- **imgsz 480 pierde personas pequeñas**: 15 frente a 20 en el mismo fotograma con conf 0,35. Recomiendo **640 por defecto**, con 480 como plan B.
- Coste de E/S: `cap.read()` + resize de 1080p a 720p: unos 3,4 ms; `cap.grab()` en los fotogramas saltados: 0,8 ms; `VideoWriter mp4v` a 1280×720: 3,5 ms por fotograma; recodificar 10 s con libx264 veryfast: 0,7 s.
- **Flujo completo medido** (`snippets_nb2.py`: 2.3 + 2.5 con permanencia + 2.6, con escritura y recodificación de los dos vídeos, 10 s, salto 2): 24,5 s con 8 hilos (pasada de zonas 11,1 s y de línea 9,0 s) y 31,7 s en la segunda ejecución, con la pasada de zonas a 1 hilo (11,8 s).
- **Estimación en Colab 🟡**: un vCPU Xeon a 2,0-2,2 GHz rinde por hilo aproximadamente 2,5-3,5 veces menos que un núcleo P del M3 (no lo he medido). Eso da unos 110-150 ms por llamada: 125 llamadas por pasada son 15-20 s, dos pasadas entre 30 y 40 s, y con arranque, E/S y recodificación el total queda en **60-90 s**. Cabe en los 2 minutos, pero el margen es corto. **Mide en 2.3** el tiempo de una inferencia e imprime una estimación («≈ N s para 2.5 y 2.6»). Si pasa de 100 s, sugiere salto 3 o imgsz 480.

---

## 6. Vídeo: escritura, recodificación y reproducción

- ffmpeg de `imageio_ffmpeg.get_ffmpeg_exe()` (local): **v7.1 con `--enable-libx264`**, encoder `libx264` presente (verificado con `-encoders`).
- En Colab, `shutil.which("ffmpeg")`. Según `e_colab.md`, Colab trae ffmpeg 6.1.1 con libx264. [colabtools#5815](https://github.com/googlecolab/colabtools/issues/5815), de febrero de 2026, mostraba 4.4.2 🟡. El comando de abajo solo usa opciones que existen desde hace años, así que vale para 4.4, 6.1 y 7.1.
- **Comando recomendado** (verificado):
```python
cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", entrada, "-an",
       "-vf", "scale=-2:'min(720,trunc(ih/2)*2)'",
       "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
       "-movflags", "+faststart", salida]
subprocess.run(cmd, check=True, capture_output=True, text=True)
```
  Dimensiones pares verificadas: 1283×721 → 1282×720, 1081×1921 (vertical) → 406×720, 641×481 → 640×480, 1280×720 → 1280×720. `scale=-2:'min(720,ih)'`, la variante de `e_colab.md`, **falla con una altura impar menor de 720**, porque libx264 con yuv420p exige dimensiones pares. Usa `trunc(ih/2)*2`.
- Resultado verificado (releído con cv2 y con `ffmpeg -i`): `h264 (High) (avc1)`, `yuv420p`, 1280×720, 12,5 fps (FPS/salto, así que la duración real se conserva), 10,00 s, `moov` antes de `mdat` (faststart OK). **Tamaño: 1,55-1,65 MB** para 10 s anotados. Base64 añade un 33 %: unos 2,2 MB en la salida.
- Mostrar: `IPython.display.Video(ruta, embed=True, width=640, html_attributes="controls muted loop")` y el `<video src="data:video/mp4;base64,…">` hecho a mano generan el mismo HTML. Verificado el `_repr_html_` con IPython 9.17. `e_colab.md` prefiere la variante HTML por el texto de reserva en castellano, y me parece bien.
- Recuerda: `os.remove` del `.mp4` temporal `mp4v` tras recodificar.

---

## 7. Licencia

- Metadatos instalados: `License: AGPL-3.0`, clasificador `GNU Affero General Public License v3 or later (AGPLv3+)` y `dist-info/licenses/LICENSE` con el texto completo de la AGPL. Cada fichero fuente lleva la cabecera `# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license`.
- Los pesos YOLO26 están bajo AGPL-3.0 o licencia Enterprise: «available … under AGPL-3.0 and Enterprise licenses» ([docs YOLO26](https://docs.ultralytics.com/models/yolo26/)).
- Aviso de Ultralytics ([ultralytics.com/license](https://www.ultralytics.com/license)): la AGPL-3.0 obliga a publicar el código del proyecto que la use. La licencia Enterprise es para uso propietario o comercial cerrado. El uso académico y docente encaja con la AGPL si el trabajo se mantiene abierto.
- Texto propuesto para NB2 (llano, sin asesorar): «Ultralytics YOLO se distribuye con licencia AGPL-3.0: puedes usarlo para aprender y experimentar, pero si lo integras en un producto tienes que publicar tu código o comprar su licencia comercial.»

---

## 8. Snippets mínimos verificados para NB2 §2.3–2.7

Versión completa ejecutada: `verificacion/c/snippets_nb2.py`. Termina sin errores y comprueba con `assert` que el recuento por umbral coincide con volver a predecir. Resumen de lo esencial:

```python
# --- comunes (2.1/2.2) ---
PERSONA, RUTA_PESOS = 0, "pesos/yolo26n.pt"
def fotogramas(ruta, segundos, salto, ancho_max=1280):
    cap = cv2.VideoCapture(ruta); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    for i in range(int(segundos * fps)):
        if i % salto:
            if not cap.grab(): break
            continue
        ok, f = cap.read()
        if not ok: break
        h, w = f.shape[:2]
        if w > ancho_max:
            f = cv2.resize(f, (ancho_max, round(h * ancho_max / w / 2) * 2), interpolation=cv2.INTER_AREA)
        yield i / fps, f
    cap.release()

# --- 2.3 un fotograma: UNA inferencia con conf=0.05 y curva recuento-umbral ---
modelo = YOLO(RUTA_PESOS)
res = modelo.predict(primer, classes=[PERSONA], conf=0.05, verbose=False)[0]
confs = res.boxes.conf.cpu().numpy()
umbrales = np.round(np.arange(0.1, 0.91, 0.05), 2)
recuento = [(confs >= u).sum() for u in umbrales]   # == predecir con conf=u (verificado con assert)
vistas = res[res.boxes.conf >= confianza_minima]    # Results admite máscara booleana
vistas.names = {**vistas.names, PERSONA: "persona"} # etiqueta en castellano (ASCII)
img_bgr = vistas.plot(conf=True, line_width=2)

# --- 2.4 zonas en % -> píxeles ENTEROS ---
def zona_a_pixeles(texto, nombre):
    try:
        pts = [tuple(float(v) for v in p.split(",")) for p in texto.split(";") if p.strip()]
        assert all(len(p) == 2 for p in pts)
    except (ValueError, AssertionError):
        raise ValueError(f"La zona «{nombre}» no se entiende. Escribe pares x,y en % separados por «;», p. ej. 0,0; 50,0; 50,100; 0,100")
    if any(not 0 <= v <= 100 for p in pts for v in p):
        raise ValueError(f"La zona «{nombre}» tiene valores fuera de 0-100 %. Corrígelos.")
    return [(int(round(x / 100 * (ANCHO - 1))), int(round(y / 100 * (ALTO - 1)))) for x, y in pts]
# zonas: dict {nombre: pts} con 3+ puntos; linea: exactamente 2 puntos

# --- 2.5 aforo por zonas + 2.7 permanencia en la MISMA pasada ---
rc = solutions.RegionCounter(model=RUTA_PESOS, region=zonas, classes=[PERSONA], conf=confianza_minima,
                             tracker="bytetrack.yaml", show_conf=False, line_width=2, verbose=False)
rc.names = {**rc.names, PERSONA: "persona"}
serie, fotos_en_zona = [], {}
vw = cv2.VideoWriter("tmp_zonas.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps / salto, (ANCHO, ALTO))
for t, f in fotogramas(ruta, segundos, salto):
    r = rc(f)                                           # dibuja sobre f
    serie.append({"t_s": round(t, 2), **r.region_counts})   # ocupación de ESTE fotograma
    for caja, id_ in zip(rc.boxes, rc.track_ids):
        x0, y0, x1, y1 = map(float, caja)
        centro = rc.Point(((x0 + x1) / 2, (y0 + y1) / 2))
        for z in rc.counting_regions:
            if z["prepared_polygon"].contains(centro):
                fotos_en_zona[(z["name"], id_)] = fotos_en_zona.get((z["name"], id_), 0) + 1
    for n, pts in zonas.items():                        # RegionCounter no rotula nombres
        cv2.putText(r.plot_im, n, (pts[0][0] + 8, pts[0][1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    vw.write(r.plot_im)
vw.release(); a_h264("tmp_zonas.mp4", "zonas.mp4")
df = pd.DataFrame(serie); df[list(zonas)].agg(["max", "mean"])
permanencia_s = {k: n * salto / fps for k, n in fotos_en_zona.items()}   # segundos por (zona, ID)

# --- 2.6 entradas/salidas por línea, con rótulo en castellano ---
oc = solutions.ObjectCounter(model=RUTA_PESOS, region=linea, classes=[PERSONA], conf=confianza_minima,
                             tracker="bytetrack.yaml", show_conf=False, show_in=False, show_out=False,
                             line_width=2, verbose=False)   # show_in/out=False: sin el rótulo inglés «IN/OUT»
oc.names = {**oc.names, PERSONA: "persona"}
for t, f in fotogramas(ruta, segundos, salto):
    r = oc(f)
    oc.annotator.display_analytics(r.plot_im, {"Entradas": r.in_count, "Salidas": r.out_count},
                                   (104, 31, 17), (255, 255, 255), oc.margin)
    ...  # serie con r.in_count / r.out_count (acumulados) y vw.write(r.plot_im)
```
En el vídeo se ven cajas con «N persona», el nombre de cada zona y los rótulos «Entradas: 7» / «Salidas: 8» (captura en `verificacion/c/mosaico.jpg`).

---

## 9. Recomendaciones para el builder de NB2

1. Preparación: `pip install -q "ultralytics==8.4.161" "lap>=0.5.12" "shapely>=2.0.0" imageio-ffmpeg`, variables `YOLO_VERBOSE`/`YOLO_AUTOINSTALL` antes del import, `LOGGER.setLevel(ERROR)`, descarga explícita a `pesos/yolo26n.pt` e impresión de versiones. Fijar la versión exacta (en vez de `>=`) está justificado: la API de solutions cambia entre versiones menores (`half`→`quantize`, `nms`, `forget_tracks`).
2. Crea la solution **dentro** de la celda: al reejecutar, los contadores y el tracker empiezan de cero (cada solution carga su propio `YOLO`).
3. `tracker="bytetrack.yaml"`, `imgsz=640`, `conf=confianza_minima` (0,35 por defecto) y `classes=[0]`.
4. Frames reducidos a 1280 de ancho **antes** de las solutions: las zonas en % se convierten con ese `ANCHO`/`ALTO`. Mejor limitar el **lado mayor** a 1280 para que un vídeo vertical no pase entero.
5. En 2.6, explica el sentido en llano: «Con una línea horizontal, cuenta como entrada quien cruza hacia abajo; con una vertical, quien cruza hacia la derecha». Ofrece `invertir_sentido`. Explica también que cada ID se cuenta una vez y que un cambio de ID duplica.
6. En 2.7, cuenta solo los IDs con al menos N fotogramas en la zona (p. ej. ≥1 s) y avisa de que los IDs vivos al principio o al final del clip dan permanencias truncadas.
7. En 2.3, redacta con cuidado lo del «sin NMS» (sección 1) y usa la curva de una sola inferencia.
8. Si quieres, apaga la telemetría con `events.enabled = False` y explícalo en el aviso de privacidad.
9. Para `requirements-verificacion.txt`, añade `lap>=0.5.12` y `shapely>=2.0.0`: **el venv compartido no los tiene**.

## Fuentes
- Fuente instalado: `ultralytics/solutions/{config,solutions,region_counter,object_counter}.py`, `utils/{downloads,checks,events,__init__,torch_utils}.py`, `trackers/utils/matching.py`, `cfg/trackers/*.yaml`, `cfg/default.yaml`, `engine/predictor.py`, `nn/modules/head.py`.
- https://docs.ultralytics.com/models/yolo26/ (nombre de pesos, ficha, cabezas uno-a-uno y uno-a-muchos, `nms=False`, licencia)
- https://docs.ultralytics.com/guides/region-counting/ (formato de región, ocupación por fotograma)
- https://docs.ultralytics.com/guides/object-counting/ (línea de 2 puntos o polígono; no define el sentido)
- https://www.ultralytics.com/license (AGPL-3.0 frente a Enterprise)
- https://github.com/googlecolab/colabtools/issues/5815 (ffmpeg en Colab, febrero de 2026)

## Pendientes de validar
- 🟡 Tiempo real en Colab CPU: estimado, no medido.
- 🟡 Aviso de `check_imshow` en Colab: deducido del fuente (`IS_COLAB`), no observado.
- 🟡 Versión exacta de ffmpeg en Colab hoy (6.1.1 según `e_colab.md` o 4.4.2 según el issue de febrero). El comando funciona con ambas.
- 🟡 Si Colab trae `shapely` y `lap` preinstalados: da igual, se instalan explícitamente.
