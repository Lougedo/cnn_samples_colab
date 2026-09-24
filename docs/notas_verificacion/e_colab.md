# Frente e) Colab: formularios, metadatos, vídeo, curvas en vivo y entorno

Fecha: 24-sep-2026. Material de prueba en `verificacion/e/` (gitignored): `build_test_nb.py` (notebook de prueba con nbformat), `prueba_colab.ipynb`, `celda_instalacion.py`, `bench_hilos.py`, listados de paquetes de Colab (`bi_*.txt`) y resoluciones con uv (`resolved_*.txt`).
Entorno local: `.venv` con Python 3.12.7, IPython 9.17.1, nbclient 0.11.0, nbformat 5.11.1, ipykernel 7.3.0, matplotlib 3.11.2 y matplotlib-inline 0.2.2.
🟡 marca lo que no he podido ejecutar ni encontrar en fuente primaria. En la práctica: todo lo que solo se ve en la interfaz de Colab.

---

## 0. Lo que el constructor tiene que saber sí o sí

1. **Colab ya no va con Python 3.12: desde el 25-ago-2026 usa Python 3.13** ([colabtools#6081](https://github.com/googlecolab/colabtools/issues/6081)). Desde el 9-sep-2026 corre sobre Ubuntu 24.04 ([#6087](https://github.com/googlecolab/colabtools/issues/6087)). El punto 10 del BRIEF («emula 3.11–3.12») está desfasado. Nuestra `.venv` usa 3.12.7. Lo aceptamos para la verificación y lo documentamos en `DECISIONS.md` (ver §6.3).
2. **TensorFlow sigue preinstalado** en los runtimes de CPU y GPU, pero no en los de TPU. La versión es la 2.20.0 o la 2.21.0 (se actualizó a 2.21 el 23-sep y el despliegue tarda 1-2 días). Keras es la **3.13.2**, no la 3.15.1 que tenemos en local. numpy es la **2.1.3** y pandas la **2.2.3** (en local tenemos 3.0.6).
3. `pip install ultralytics medmnist supervision visualkeras` en el Colab actual **no cambia ningún paquete preinstalado**. Solo añade 11 paquetes pequeños. Lo he comprobado resolviendo con uv contra el `pip-freeze` oficial y no en Colab real, así que 🟡 hasta probarlo allí.
4. `cellView: "form"` va en `cell.metadata`. Es lo que oculta el código. `# @title` solo pone el título. El JSON está verificado con nbformat y coincide con el de los notebooks de Google.
5. Para las curvas en vivo, el patrón `display(fig, display_id=True)` + `handle.update(fig)` + **`plt.close(fig)` al final** funciona en nbclient. Sin el `plt.close` la figura sale duplicada (verificado).
6. Para el vídeo inline: H.264 embebido en base64. Un clip real de 10 s a 720p con CRF 28 ocupa entre 0,35 y 1 MB, así que no hay problema de tamaño.
7. **No usar `cv2.VideoWriter(..., 'avc1')`**. En el Mac funciona porque tira de AVFoundation, pero en Colab/Linux la rueda pip de OpenCV no trae libx264 (🟡). Hay que escribir en `mp4v` y recodificar con ffmpeg.

---

## 1. Sintaxis de formularios (tal como la escribe Colab)

Fuente: copia literal del notebook oficial «Forms» ([ipyform/example/colab_offical.ipynb](https://github.com/phihung/ipyform/blob/main/example/colab_offical.ipynb), que es el mismo contenido de https://colab.research.google.com/notebooks/forms.ipynb). La página de Colab es una SPA y no se puede descargar el .ipynb directamente. Se complementa con los notebooks de Google en [colabtools/notebooks](https://github.com/googlecolab/colabtools/tree/main/notebooks).

```python
# @title 1.3 Diseña tu red { display-mode: "form" }
# @markdown Mueve los controles y ejecuta la celda (▶).
bloques_convolucionales = 2  # @param {type:"slider", min:1, max:4, step:1}
filtros_primer_bloque = 16  # @param [8, 16, 32] {type:"raw"}
usar_pooling = True  # @param {type:"boolean"}
dropout = 0.2  # @param {type:"slider", min:0, max:0.6, step:0.1}
nombre_experimento = "grupo_1"  # @param {type:"string"}
confianza_minima = 0.35  # @param {type:"number"}
epocas = 8  # @param {type:"integer"}
fuente_video = "demo: personas caminando"  # @param ["demo: personas caminando", "subir mi vídeo"]
zonas = "0,0; 50,0; 50,100; 0,100"  # @param {type:"string", placeholder:"x1,y1; x2,y2; ..."}
```

Tipos que aparecen en el notebook oficial:
- `{type:"string"}`, `{type:"string", placeholder:"..."}`
- `{type:"number"}` (float), `{type:"integer"}`
- `{type:"slider", min:…, max:…, step:…}`
- `{type:"boolean"}`, `{type:"date"}`, `{type:"raw"}`
- desplegable `["a","b"]`, desplegable editable `["a","b"] {allow-input: true}` y desplegable raw `[1, "False"] {type:"raw"}`
- en el título, `{ display-mode: "form" }` y `{ run: "auto" }` (este último vuelve a ejecutar la celda al cambiar un control, pero solo después de haberla ejecutado una vez a mano)

Reglas prácticas:
- **Asignación y `# @param` en la misma línea**, con una sola variable simple (`x = valor  # @param …`). Colab reescribe el literal de esa línea cuando el alumno mueve el control. El notebook oficial solo usa esta forma. El parser de ipyform exige `ast.Assign` de una sola línea y con un solo objetivo. No uses valores multilínea, anotaciones de tipo (`x: int = …`) ni desempaquetados. 🟡 que Colab rechace esas variantes: no lo he probado, simplemente no las uses.
- **Desplegables numéricos: `{type:"raw"}`** (`[8, 16, 32] {type:"raw"}`). Sin `raw`, las opciones del ejemplo oficial son cadenas y lo que se inserta es un string. 🟡 cómo trata Colab una lista numérica sin `raw`. En el código conviene convertir de todos modos: `int(filtros_primer_bloque)`.
- **Nada después de la configuración.** No pongas `# @param {...}  # comentario` ni texto tras `]` o `}`. En Python es un comentario y no molesta (verificado), pero el parser de Colab es cerrado y al reescribir la línea puede comerse ese texto. 🟡 Las explicaciones van en líneas `# @markdown` encima del control.
- `# @title` en la **primera línea** de la celda. En los notebooks de Google siempre es así. 🟡 si funciona en otra posición.
- Tanto `#@title` como `# @title` son válidos: Google usa las dos formas en sus propios notebooks (`Classify_an_image_using_Gemini.ipynb` usa `#@title`, y `Getting_started_with_google_colab_ai.ipynb` y el Forms oficial usan `# @title`). Colab, cuando edita, escribe `# @param` con espacio. **Recomendado: `# @title` y `# @param` con espacio.**
- En los títulos no metas `{` ni `}`: Colab los interpreta como opciones (`display-mode`, `run`).
- **Acentos y eñes en títulos, en `@markdown` y en opciones de desplegable:** el notebook es UTF-8 y localmente se ejecuta sin problemas (verificado en `prueba_colab.ipynb`, celda 1). 🟡 no he podido ver el renderizado en Colab. Es muy probable que funcione porque la interfaz de Colab está localizada al español, pero entra en la lista de comprobaciones manuales.
- Slider con decimales: redondea en el código (`dropout = round(float(dropout), 2)`). 🟡 que Colab inserte valores tipo `0.30000000000000004`.
- Valores fuera del rango del slider escritos en el código (p. ej. `epocas = 1` con `min:3`): Python los ejecuta sin quejarse (verificado). 🟡 cómo los pinta Colab en el slider. Esto afecta al «modo rápido» de verificación (ver §4.3).
- Fuera de Colab, `# @param` es un comentario sin más: todas las celdas de formulario se ejecutaron en nbclient con los valores por defecto (verificado). Los tipos resultantes son `int`, `int` (raw), `str`, `bool`, `float`, `str`, `float`, `int`, `str` y `str`.

## 2. `cellView` y metadatos del notebook

JSON verificado con nbformat 5.11.1 (`nbf.validate` pasa):

```json
{"cell_type": "code", "id": "nb1-03", "metadata": {"cellView": "form"}, "outputs": [], "execution_count": null,
 "source": ["# @title 1.3 Diseña tu red\n", "..."]}
```

- `cellView: "form"` hace que la celda muestre solo el título, los controles y la salida. El alumno puede ver el código con «Mostrar código». Sin `cellView`, una celda con `# @title` muestra el título y además el código: los notebooks `Getting_started_with_google_colab_ai` y `How_to_Use_Slideshow_Mode` de Google tienen celdas con `# @title` y sin `cellView`. Los notebooks «de formulario» de Google (Gemini, Sheets…) llevan siempre `"cellView": "form"` + `#@title`. `{ display-mode: "form" }` en el título hace lo mismo desde el propio código. Es opcional, pero conviene añadirlo por si alguna herramienta borra metadatos. 🟡 cuál manda si se contradicen.
- Una celda `cellView: form` debe llevar siempre `# @title`, porque si no el alumno solo ve una celda vacía plegada. 🟡
- Metadatos de notebook recomendados (son los que usan los notebooks de Google):
  ```python
  nb.metadata["colab"] = {"provenance": [], "toc_visible": True}   # toc_visible: abre el índice lateral 🟡
  nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
  nb.metadata["language_info"] = {"name": "python"}
  ```
  **No pongas `accelerator: "GPU"` ni `colab.gpuType`**: harían que Colab pidiera GPU al abrir, y queremos arrancar en CPU. El kernel `python3` también existe en la `.venv` local (`jupyter kernelspec list`), así que nbclient lo encuentra.
- nbformat crea notebooks v4.5 con `id` en la raíz de la celda, mientras que Colab guarda v4.0 con `metadata.id`. 🟡 que Colab abra v4.5 sin problemas: es el formato por defecto de JupyterLab desde 2021, así que el riesgo es bajo.
- **IDs deterministas:** nbformat genera ids aleatorios en cada build, lo que ensucia los diffs. Asigna `cell.id = f"nb1-{i:02d}"`. El patrón válido es `[A-Za-z0-9_-]{1,64}`: un id con espacio lo rechaza `validate` (verificado).
- Guardar sin salidas: las celdas nuevas de nbformat ya salen con `outputs: []` y `execution_count: null`.

## 3. Vídeo inline en Colab

**Recomendación:** H.264 embebido en base64. Hay dos formas equivalentes:

```python
from IPython.display import Video, HTML, display
display(Video(ruta_h264, embed=True, width=640, html_attributes="controls muted loop"))
# o bien, con el texto de reserva en castellano:
b64 = base64.b64encode(open(ruta_h264, "rb").read()).decode()
display(HTML(f'<video controls muted loop playsinline width="640" src="data:video/mp4;base64,{b64}">'
             'Tu navegador no puede reproducir este vídeo.</video>'))
```

- En Colab hay **IPython 7.34.0**. He revisado el código de su `Video`: existen `embed`, `width` y `html_attributes`, y el tipo MIME se adivina por la extensión `.mp4`. El texto de reserva está en inglés («Your browser does not support the video tag.») y solo se ve si falla la reproducción. **Prefiero la variante `HTML`**: el texto de reserva queda en castellano y no dependes de la versión de IPython. `mediapy`, de Google y pensada para Colab, hace exactamente lo mismo ([mediapy/__init__.py](https://github.com/google/mediapy/blob/main/mediapy/__init__.py), `<source src="data:video/mp4;base64,…">`).
- `Video(ruta)` sin `embed=True` **no funciona en Colab**: el navegador no puede acceder al disco de la VM. Hay que embeber siempre.
- Ambas variantes funcionan en nbclient: la salida es `text/html` (verificado).
- Recodificación verificada con el ffmpeg de imageio-ffmpeg (v7.1, incluye libx264):
  ```python
  FFMPEG = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()   # Colab trae los dos
  [FFMPEG, "-y", "-loglevel", "error", "-i", entrada, "-vf", "scale=-2:'min(720,ih)'",
   "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
   "-movflags", "+faststart", "-an", salida]
  ```
  `scale=-2` fuerza un ancho par, que yuv420p necesita. `+faststart` permite que empiece a reproducirse antes de cargar el fichero entero.
- Tamaños medidos con 10 s de clip real (máquina cargada, tiempos orientativos):

  | Clip | Origen | 720p CRF 28 | 720p CRF 28, 12,5 fps | 480p CRF 28, 12,5 fps | 720p CRF 23 |
  |---|---|---|---|---|---|
  | people-walking | 1080p 25 fps | 0,96 MB (1,1 s) | 0,90 MB | 0,51 MB | 1,90 MB |
  | grocery-store | **4K** 30 fps | 0,35 MB (4,6 s) | 0,33 MB | 0,13 MB | 0,99 MB |

  El ruido sintético (incomprimible) llega a unos 7,8 MB en 10 s: es el peor caso y no se da con vídeo real. Un vídeo anotado con cajas pesará algo más que el original (🟡 de 1 a 3 MB).
- **Límites:** base64 añade un 33 %. Con dos vídeos de 7,8 MB, el notebook ejecutado de prueba ocupaba 20 MB. Colab no publica límites de tamaño de salida ni de notebook ([FAQ](https://research.google.com/colaboratory/faq.html): «Colab does not publish these limits»). Hay informes de desconexión del runtime al embeber vídeos de decenas de MB (🟡 [SO 67591072](https://stackoverflow.com/questions/67591072/displaying-large-video-files-in-google-colab)). **Regla para el builder:** 720p máximo y CRF 28. Si el mp4 supera 20 MB, no lo embebas: muestra un aviso en castellano y ofrece la descarga.
- Decodificar un 4K cuesta: unos 4,6 s por cada 10 s de clip, solo para recodificar, en el M3 Pro. En NB2, reescala los fotogramas al leerlos si el origen pasa de 1080p.
- **`cv2.VideoWriter`:** en el Mac se abren `avc1`, `H264` y `mp4v` (build con AVFoundation, verificado). En Linux las ruedas pip de OpenCV no llevan encoder H.264 (🟡, informe habitual). Usa siempre `mp4v` y recodifica con ffmpeg.

## 4. Curvas en vivo por época

### 4.1 Patrón recomendado: `display_id` + `update` + `plt.close`

```python
class CurvasEnVivo(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.h = {}
        self.fig, self.axs = plt.subplots(1, 2, figsize=(11, 4))
        self.graf = display(self.fig, display_id=True)
        self.txt = display(Markdown("Entrenando…"), display_id=True)
    def on_epoch_end(self, epoch, logs=None):
        for k, v in (logs or {}).items(): self.h.setdefault(k, []).append(float(v))
        # ... ax.clear(), redibujar pérdida/accuracy (entrenamiento y validación) ...
        self.fig.tight_layout(); self.graf.update(self.fig)
        self.txt.update(Markdown(f"Época {epoch + 1} · {seg:.1f} s"))
    def on_train_end(self, logs=None):
        plt.close(self.fig)   # imprescindible: si no, el backend inline pinta otra copia al final de la celda
```

- Verificado en nbclient con un bucle simple y con `model.fit(..., verbose=0, callbacks=[CurvasEnVivo()])`. La celda termina con **una sola** imagen y un solo texto, ambos en su estado final. **Sin `plt.close`, sale la figura duplicada** (verificado: dos `display_data` con `image/png`).
- Existe en IPython 7.34 (Colab): `display(..., display_id=True)` devuelve un `DisplayHandle` con `.update()` (revisado en el código de 7.34).
- En Colab: [colabtools#2020](https://github.com/googlecolab/colabtools/issues/2020). Colab atribuyó el fallo de `display_id` a Plotly (que no pone `display_id`), no a Colab. El que abrió el issue, además, usaba `.display()` en vez de `.update()`. Las librerías que intercambian PNG con `display_id` (p. ej. [liveplot](https://github.com/ARENA-education/liveplot)) dicen funcionar igual en Jupyter y en Colab. 🟡 no lo he visto en Colab con mis propios ojos. Entra en la lista de comprobación manual.
- Usa `verbose=0` en `fit`: la barra de progreso de Keras llena la salida de Colab, y Colab trunca las salidas de texto muy largas (🟡).
- Coste de redibujar una figura de 2 paneles y pasarla a PNG: **unos 90 ms por época en local** (con carga de otros agentes). En Colab calcula de 0,2 a 0,3 s (🟡). Con 8 épocas son 2-3 s del presupuesto de 90 s.

### 4.2 Alternativa: `clear_output(wait=True)`
También funciona en nbclient (verificado: termina con una imagen y el último `print`). El inconveniente es que borra **todo** lo que ha salido en la celda, así que la ficha, los avisos y el tiempo por época hay que repintarlos en cada época. Además, en Colab puede parpadear o hacer saltar el scroll (🟡). **Me quedo con el patrón A.**

### 4.3 Modo rápido en la verificación
Colab cambia el valor de un control **reescribiendo la línea del `@param`**. Por eso `scripts/verificar.py` puede simular a un alumno haciendo lo mismo antes de ejecutar:
```python
re.sub(r'^(epocas\s*=\s*)[^#]+(#\s*@param)', r'\g<1>1  \2', src, flags=re.M)
```
Verificado en nbclient: da `epocas 1`. Es más fiel que usar papermill, que inyecta una celda nueva y solo admite una celda con la etiqueta `parameters`.

## 5. Ficheros y detección de Colab

```python
import os, sys
EN_COLAB = "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules
try:
    from google.colab import files as colab_files
except ImportError:
    colab_files = None
```
- Ultralytics 8.4.161 detecta Colab con `"COLAB_RELEASE_TAG" in os.environ or "COLAB_BACKEND_VERSION" in os.environ` (`ultralytics/utils/__init__.py:763`). El `try/except ImportError` se ha ejecutado en local sin errores.
- `files.upload(target_dir='')` ([colabtools/files.py](https://github.com/googlecolab/colabtools/blob/main/google/colab/files.py)) devuelve `{nombre_local: bytes}`. Si el nombre ya existe, lo renombra a `video (1).mp4`, así que **usa la clave del dict y no el nombre original**. Imprime en inglés «Saving X to Y». Se bloquea hasta que el alumno elige un fichero. Si cancela, el dict llega vacío 🟡: contémplalo con un mensaje en castellano. La transferencia va por trozos base64 a través de `eval_js`, así que es lenta con ficheros grandes: pon un tope de 50-100 MB 🟡 y recomienda clips cortos. Da problemas en Firefox ([#3479](https://github.com/googlecolab/colabtools/issues/3479)). Alternativa: arrastrar el fichero al panel «Archivos» y escribir la ruta en un campo de texto.
- `files.download(ruta)`: lanza `FileNotFoundError` si el fichero no existe. Solo funciona en la **pestaña del navegador que ejecutó la celda** (`google.colab.kernel.accessAllowed`). Hay problemas históricos en Safari y Firefox ([#1909](https://github.com/googlecolab/colabtools/issues/1909), [#468](https://github.com/googlecolab/colabtools/issues/468)). Recomienda **Chrome** y como plan B el panel «Archivos» → ⋮ → Descargar. Fuera de Colab, imprime la ruta absoluta (verificado).
- Mensaje heredado en inglés: si el notebook se abre con salidas guardadas, el widget de subida muestra «Upload widget is only available when…». No aplica si guardamos sin salidas.

## 6. Entorno Colab actual y celda de instalación

### 6.1 Versiones (fuente primaria: [googlecolab/backend-info](https://github.com/googlecolab/backend-info), `pip-freeze.txt`/`os-info.txt`/`apt-list.txt`, commits del 23-sep-2026)

| | Colab por defecto (CPU) | Runtime fijado «2026.07» | Nuestra `.venv` |
|---|---|---|---|
| SO | Ubuntu 24.04.5 | Ubuntu 22.04.5 | macOS arm64 |
| Python | **3.13.15** | 3.12.13 | 3.12.7 |
| tensorflow | 2.20.0 → **2.21.0** (se está desplegando) | 2.20.0 | 2.21.0 |
| keras | **3.13.2** | 3.13.2 | 3.15.1 |
| numpy | **2.1.3** | 2.0.2 | 2.5.3 |
| torch / torchvision | 2.11.0+cpu / 0.26.0 | 2.11.0+cpu | 2.14.0 |
| opencv-python(-headless) | 5.0.0.93 (+ contrib 4.14) | 5.0.0.93 | 5.0.0.93 |
| pandas | **2.2.3** | 2.2.2 | **3.0.6** |
| matplotlib | 3.10.0 | 3.10.0 | 3.11.2 |
| scikit-learn | 1.6.1 | 1.6.1 | 1.9.1 |
| IPython / ipykernel | **7.34.0** / 6.17.1 | 7.34.0 | 9.17.1 / 7.3.0 |
| imageio-ffmpeg | 0.6.0 | — | 0.6.0 |
| ffmpeg de sistema | **sí** (6.1.1, con libx264) | — | no |
| ultralytics / supervision / medmnist / visualkeras | **no preinstalados** | no | sí |

- En el runtime de TPU no hay TensorFlow ([runtime-version-faq](https://research.google.com/colaboratory/runtime-version-faq.html)). En el de GPU sí (tensorflow 2.21.0 y torch 2.11.0+cu130).
- **Plan B por si 3.13 da guerra:** la paleta de comandos tiene «Change runtime version» → 2026.07 (Python 3.12). No persiste entre sesiones ([#6081](https://github.com/googlecolab/colabtools/issues/6081)).
- CPU gratuita: **2 vCPU** (1 núcleo con hyperthreading), Intel Xeon @ 2,20 GHz y unos 12,7 GB de RAM. 🟡 Colab no lo publica ([FAQ](https://research.google.com/colaboratory/faq.html)); los datos vienen de fuentes de la comunidad ([colabtools#1937](https://github.com/googlecolab/colabtools/issues/1937), [Saturn Cloud](https://saturncloud.io/blog/whats-the-hardware-spec-for-google-colaboratory/)). **Imprime `os.cpu_count()` y el modelo de CPU al inicio de cada notebook** para que el profesor vea qué le ha tocado.

### 6.2 Riesgos de `pip install` en Colab
- Resolución con `uv pip compile` (Python 3.13, `x86_64-manylinux_2_28`) de `ultralytics medmnist supervision visualkeras`, **restringida a las versiones exactas del pip-freeze de Colab** (sin `nvidia-*`, que es ruido de la rueda de torch de PyPI): **resuelve sin tocar ningún paquete preinstalado.** Se instalarían 11 paquetes nuevos: `ultralytics 8.4.161`, `ultralytics-thop 2.1.6`, `ultralytics-platform 0.1.57`, `supervision 0.30.5`, `av 18.1.0`, `pydeprecate 0.11.0`, `medmnist 3.0.2`, `fire 0.7.1`, `visualkeras 0.2.0`, `aggdraw 1.4.1` (hay rueda cp313 manylinux) y `nvidia-ml-py`. Con el runtime 2026.07 (Python 3.12) sale lo mismo. **No hace falta reiniciar por numpy.** 🟡 hasta probarlo en Colab real.
- Ultralytics 8.4.161 excluye numpy 2.0–2.3.4 **solo en macOS** (`platform_system == "Darwin"`). En Colab no afecta, pero impide montar en el Mac una venv que emule Colab con numpy 2.1.3 y ultralytics a la vez (uv lo confirma: no hay solución). La pila de NB1 con las versiones de Colab sí resuelve en macOS con Python 3.13 (`resolved_emul_nb1.txt`).
- `tensorflow-cpu 2.21.0` tiene ruedas cp313 **solo para Linux y Windows**, no para macOS. En Linux x86_64 la rueda de `tensorflow` pesa 573 MB. Como el fallback solo se activaría en TPU, basta con `pip install tensorflow`.
- La `.venv` local **no tiene pip** (la gestiona uv). La celda de instalación no puede llamar a pip sin más: primero comprueba con `find_spec`.

### 6.3 Celda de instalación robusta (probada en local: sin nada que instalar y en modo simulación)
Ver `verificacion/e/celda_instalacion.py`. Esquema:
```python
faltan = [pip for mod, pip in {"medmnist": "medmnist", "visualkeras": "visualkeras",
                                 "sklearn": "scikit-learn"}.items() if importlib.util.find_spec(mod) is None]
if importlib.util.find_spec("tensorflow") is None: faltan.append("tensorflow")
if faltan:
    fijos = [f"{p}=={version(p)}" for p in ("numpy", "tensorflow", "keras", "torch", "opencv-python",
                                            "pandas", "matplotlib") if instalado(p)]
    # escribir fijos en un fichero temporal y pasarlo con -c: pip NO puede actualizar numpy y compañía;
    # si algo fuese incompatible, pip falla sin romper el entorno
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *faltan, "-c", restricciones], ...)
    importlib.invalidate_caches()   # importar lo recién instalado sin reiniciar
```
- Usa `sys.executable -m pip` y no `!pip`: garantiza que se instala en el mismo intérprete y no mete magia de IPython en el código generado.
- Si pip falla, lanza un `RuntimeError` en castellano con la acción a tomar y las últimas líneas de stderr.
- **Decisión para `DECISIONS.md`:** la verificación local corre con Python 3.12.7 y Keras 3.15, y Colab usa 3.13 y Keras 3.13.2. Hay que escribir código compatible con **pandas 2.2 y 3.0** (sin asignaciones encadenadas ni `inplace`), **numpy 2.1** (nada de APIs posteriores), **scikit-learn 1.6**, **matplotlib 3.10** (sin estilos `petroff6/8`) y **Keras 3.13**. 🟡 Opcional pero muy recomendable: una segunda venv con las versiones de Colab (`uv venv -p 3.13` + `req_colab_emul_nb1.txt`) para pasar NB1 una vez. No la he creado para no descargar unos 300 MB mientras otros agentes trabajan.

## 7. Abrir en Colab

- Badge oficial ([colab-github-demo.ipynb](https://github.com/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb); el SVG responde 200):
  ```markdown
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<USUARIO>/<REPO>/blob/main/notebooks/01_laboratorio_cnn_radiografias.ipynb)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<USUARIO>/<REPO>/blob/main/notebooks/02_conteo_personas_cafeteria.ipynb)
  ```
  El formato es `…/github/<org>/<repo>/blob/<rama>/<ruta>`: no olvides `blob`. Solo funciona si el repo es público; para uno privado hay que autorizar GitHub en Colab. Es un notebook ajeno a Google, así que la primera ejecución muestra el aviso «no lo ha creado Google» con el botón «Ejecutar de todos modos» (🟡 texto exacto): hay que decirlo en el guion.
- Desde Drive: subir el `.ipynb` a Drive → doble clic o «Abrir con → Google Colaboratory». Otra opción es, en Colab, **Archivo → Subir cuaderno**. 🟡 La interfaz de Colab cambió en julio de 2026: el diálogo clásico de «Abrir cuaderno» está ahora en «Recientes → Ver más» ([#6069](https://github.com/googlecolab/colabtools/issues/6069)). **Recomendación:** repartir un **enlace directo** (el badge o el enlace de compartir de Drive, que tiene la forma `colab.research.google.com/drive/<ID>`) en lugar de describir menús. Los nombres de los menús en español entran en la comprobación manual.
- Aviso para clases presenciales: Colab ha bloqueado IPs de colegios por tráfico concurrente ([#5996](https://github.com/googlecolab/colabtools/issues/5996)). Nuestras clases son online, cada alumno desde su casa, así que el riesgo es bajo.

## 8. Matplotlib para proyector

matplotlib-inline 0.2.2 (la misma versión en Colab y en local) **no sobrescribe rcParams** desde la v0.1.4 (lo he comprobado en su `config.py`) y guarda con `bbox_inches="tight"`. Pon el estilo después de `import matplotlib.pyplot as plt`, en la celda de preparación:
```python
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "legend.fontsize": 12, "figure.dpi": 100, "savefig.dpi": 100,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]  # categorías (entrenamiento/validación, Normal/Neumonía)
```
- Mapas continuos: `viridis` (feature maps, rejillas) y `cividis` (matriz de confusión). Para la superposición de Grad-CAM, `viridis` o `inferno` con `alpha≈0.4`; **nunca `jet`**. Los dos mapas están en matplotlib 3.10 y 3.11.
- No distingas las series solo por color: usa también marcador y tipo de línea (`"o-"` para entrenamiento, `"s--"` para validación). El estilo `tableau-colorblind10` existe en las dos versiones, pero con los colores Okabe–Ito explícitos no dependes de la versión.
- Tamaños de figura de 10-12 × 4-5 pulgadas a 100 dpi dan unos 1.100 px de ancho, que llenan la celda de Colab. No uses `retina`: duplica el PNG y no aporta nada en un proyector.
- Las fuentes DejaVu (por defecto) tienen tildes, ñ, ¿ y ¡ (verificado en las figuras de prueba sin avisos de glifos).

## 9. Estimar el tiempo en Colab desde el M3 Pro

`bench_hilos.py` mide la red por defecto del BRIEF (2 bloques, 16 → 32 filtros, pooling, densa de 64, 28×28, 4.708 imágenes de entrenamiento y 524 de validación con datos aleatorios del tamaño real, batch 64). **Ojo: medido con una carga de 9-11 procesos de otros agentes, así que los tiempos son ruidosos.**

| Hilos TF | s/época |
|---|---|
| todos (11) | 0,89 |
| 2 | 1,55 |
| 1 | 1,61 |

La CPU de Colab (2 vCPU = 1 núcleo físico) se parece a nuestro caso de «1-2 hilos». A eso hay que multiplicarle lo lento que es un núcleo de Xeon a 2,2 GHz frente a un núcleo P del M3, que es de 2 a 3 veces más lento (🟡 estimación por benchmarks públicos de un solo hilo, no medida). **Estimación:** de 3,5 a 5 s por época en Colab. Con 8 épocas son 30-40 s, más la carga, la compilación y la evaluación. Entra en 90 s con margen (🟡).
- Receta para el builder: mide en local con `tf.config.threading.set_intra_op_parallelism_threads(2)` + `inter_op 1` (hay que llamarlo antes de crear tensores) y multiplica por 2,5. Para torch/YOLO en NB2 usa `torch.set_num_threads(2)`.

## 10. Lista de comprobación manual en Colab (para `INFORME_FINAL.md`)

- [ ] Las celdas de formulario muestran solo el título y los controles, con tildes y eñes bien pintadas.
- [ ] Los desplegables `raw` devuelven `int` y los sliders con decimales no dejan colas raras.
- [ ] Las curvas se actualizan en la misma figura y al final no sale una copia duplicada.
- [ ] El vídeo inline se reproduce en Chrome (y a ser posible también en Safari).
- [ ] `files.upload` funciona con un clip corto y `files.download` descarga el CSV/JSON.
- [ ] La celda de instalación termina sin pedir reiniciar la sesión (Python 3.13) y las versiones impresas coinciden con §6.1.
- [ ] El índice lateral (`toc_visible`) se abre solo.
- [ ] El aviso «notebook no creado por Google» aparece al abrir desde GitHub y se entiende.

## Fuentes

- Forms (notebook oficial): https://colab.research.google.com/notebooks/forms.ipynb · copia literal: https://github.com/phihung/ipyform/blob/main/example/colab_offical.ipynb · parser de referencia (no es de Google): https://github.com/phihung/ipyform/blob/main/src/ipyform/parser.py
- Notebooks de Google con `cellView`: https://github.com/googlecolab/colabtools/tree/main/notebooks
- Badge y URLs de GitHub: https://github.com/googlecolab/colabtools/blob/main/notebooks/colab-github-demo.ipynb
- Paquetes del runtime: https://github.com/googlecolab/backend-info (pip-freeze.txt, os-info.txt, apt-list.txt) · https://research.google.com/colaboratory/runtime-version-faq.html
- Python 3.13: https://github.com/googlecolab/colabtools/issues/6081 · Ubuntu 24.04: https://github.com/googlecolab/colabtools/issues/6087
- FAQ de recursos: https://research.google.com/colaboratory/faq.html
- files.py: https://github.com/googlecolab/colabtools/blob/main/google/colab/files.py · issues #3479, #1909, #468, #826
- display_id en Colab: https://github.com/googlecolab/colabtools/issues/2020 · https://github.com/ARENA-education/liveplot
- Vídeo H.264 en base64 (Google): https://github.com/google/mediapy
- Cambios de interfaz en julio de 2026: https://github.com/googlecolab/colabtools/issues/6069 · bloqueo de IPs en aulas: https://github.com/googlecolab/colabtools/issues/5996
- Hardware (comunidad, 🟡): https://github.com/googlecolab/colabtools/issues/1937 · https://saturncloud.io/blog/whats-the-hardware-spec-for-google-colaboratory/
