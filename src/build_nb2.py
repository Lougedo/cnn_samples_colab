"""Genera notebooks/02_conteo_personas_cafeteria.ipynb con nbformat.

Uso (desde la raíz del repo):  .venv/bin/python src/build_nb2.py
Determinista: dos ejecuciones seguidas producen el mismo archivo byte a byte.
"""
import textwrap
from pathlib import Path

import nbformat

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "notebooks" / "02_conteo_personas_cafeteria.ipynb"

celdas = []


def md(texto):
    celdas.append(nbformat.v4.new_markdown_cell(textwrap.dedent(texto).strip()))


def codigo(titulo, fuente):
    cabecera = f'# @title {titulo} {{ display-mode: "form" }}\n'
    celda = nbformat.v4.new_code_cell(cabecera + textwrap.dedent(fuente).strip())
    celda.metadata = {"cellView": "form"}
    celdas.append(celda)


# Todas las celdas salvo la 2.1 empiezan igual: formulario, comprobación de que la 2.1 se ha
# ejecutado y el cuerpo dentro de `with celda():`, que convierte los errores previstos (Parar)
# en un mensaje en castellano sin traza.
GUARDA = (
    'if "celda" not in globals():\n'
    '    print("⚠️ Primero ejecuta la celda 2.1 · Preparación (la primera del cuaderno).")\n'
    "else:\n"
    "    with celda():\n"
)


def paso(titulo, formulario, cuerpo):
    formulario = textwrap.dedent(formulario).strip()
    cuerpo = textwrap.indent(textwrap.dedent(cuerpo).strip(), " " * 8)
    codigo(titulo, (formulario + "\n\n" if formulario else "") + GUARDA + cuerpo)


# ---------------------------------------------------------------------------------------------
# 2.0 Portada
# ---------------------------------------------------------------------------------------------
md(r'''
# Contar personas en vídeo con una CNN preentrenada

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<USUARIO>/<REPO>/blob/main/notebooks/02_conteo_personas_cafeteria.ipynb)

**Qué vamos a hacer.** Aplicar a un vídeo un detector de personas que ya viene entrenado (YOLO26 nano). Sin entrenar nada, vamos a detectar personas en cada fotograma, seguirlas de un fotograma al siguiente, contar cuántas hay en cada zona y cuántas cruzan una línea, y exportar solo los totales para que n8n pueda usarlos.

**Cómo se usa.** Ejecuta las celdas en orden con el botón ▶. Algunas tienen controles (desplegables, deslizadores, cajas de texto): cámbialos y vuelve a ejecutar esa celda. No hace falta tocar código.

> **Contar no es identificar, pero grabar sí es tratar datos.** La imagen de una persona reconocible es un dato personal (RGPD, art. 4), y grabarla o analizarla es un tratamiento aunque al final solo queramos un número. El Comité Europeo de Protección de Datos considera el conteo simple menos intrusivo que la biometría, pero sigue haciendo falta una base legal, informar y guardar lo mínimo. En sitios como las mesas de un bar, la expectativa de no ser grabado pesa más.
>
> - No subas vídeos con personas reconocibles si no tienes base legal para usarlos.
> - Los archivos que exporta el cuaderno (CSV y JSON) solo llevan agregados: personas por zona en cada instante, entradas y salidas. Sin fotogramas ni identificadores.
> - Los vídeos anotados de 2.5 y 2.6 sí muestran a las personas. Se guardan en `datos/tmp/` y dentro del propio cuaderno: si lo guardas o lo compartes con las salidas, van con él.
> - En la demo del andén de metro se ven caras de pasajeros cerca de la cámara. La usamos solo para la demostración en clase.
>
> Esto es divulgación, no asesoramiento jurídico.

**Licencias**
- **Ultralytics YOLO** tiene licencia AGPL-3.0. Puedes usarlo para aprender y experimentar. Si algún día lo metes en un producto, lo ofreces como servicio o distribuyes una versión modificada, la AGPL te obliga a publicar tu código. Para uso comercial cerrado existe una licencia Enterprise de pago.
- **Vídeos de demostración**: los distribuye Roboflow con la librería `supervision` (licencia MIT, que cubre el código y no los vídeos). Roboflow no publica su origen; el de «personas caminando» coincide con un clip de Pexels (autor: Coverr). Los usamos solo para la demostración en clase, con la licencia de su fuente, y no se redistribuyen.
- Ultralytics envía por defecto estadísticas de uso anónimas. Este cuaderno las desactiva.

**Índice**
1. 2.1 Preparación
2. 2.2 Elige el vídeo
3. 2.3 Qué ve el detector en un fotograma
4. 2.4 Zonas y línea de puerta, sin dibujar
5. 2.5 Aforo por zonas
6. 2.6 Entradas y salidas por una línea
7. 2.7 Avanzado (opcional): tiempo de permanencia
8. 2.8 Exportar para automatizar (y enviar a n8n)
9. 2.9 Cierre
''')

# ---------------------------------------------------------------------------------------------
# 2.1 Preparación
# ---------------------------------------------------------------------------------------------
md(r'''
## 2.1 Preparación

Instala lo que falte, descarga el detector (5 MB) y muestra qué ordenador te ha tocado. La primera vez tarda 1-2 minutos.
''')

codigo("2.1 Preparación (la primera vez tarda 1-2 minutos)", r'''
import base64, contextlib, html, importlib, importlib.util, io, itertools, json, logging, os, platform
import random, shutil, subprocess, sys, tempfile, time, unicodedata, urllib.error, urllib.request, warnings
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from IPython.display import HTML, Image, display

EN_COLAB = "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules
try:
    from google.colab import files as colab_files
except ImportError:
    colab_files = None

if "ESTADO" not in globals():
    ESTADO = {}  # lo que cada celda deja preparado para las siguientes
    HILOS_ENTORNO = os.environ.get("OMP_NUM_THREADS")  # se lee antes de importar ultralytics, que lo cambia

CARPETA_VIDEOS, CARPETA_TMP, CARPETA_SALIDAS = Path("datos/videos"), Path("datos/tmp"), Path("salidas/nb2")
RUTA_PESOS = Path("pesos/yolo26n.pt")
PERSONA = 0  # «persona» es la clase 0 del detector
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]  # colores aptos para daltonismo
PASOS = {"modelo": "2.1 · Preparación", "video": "2.2 · Elige el vídeo", "confianza": "2.3 · Qué ve el detector",
         "zonas": "2.4 · Zonas y línea de puerta", "aforo": "2.5 · Aforo por zonas", "resumen": "2.8 · Exportar CSV y JSON"}


# ---------- Mensajes en castellano (se leen también con el tema oscuro) ----------
def _caja(texto, color, icono):
    display(HTML(f'<div style="border-left:6px solid {color};background:{color}22;padding:8px 12px;'
                 f'margin:6px 0;font-size:15px;line-height:1.5">{icono} {texto}</div>'))

def error(texto): _caja(texto, "#D55E00", "❌")
def aviso(texto): _caja(texto, "#E69F00", "⚠️")
def ok(texto): _caja(texto, "#009E73", "✅")
def info(texto): _caja(texto, "#0072B2", "ℹ️")


class Parar(Exception):
    """Problema previsto: se explica en castellano y la celda termina sin traza."""


@contextlib.contextmanager
def celda():
    try:
        yield
    except Parar as e:
        error(str(e))


def requiere(*claves):
    for clave in claves:
        if clave not in ESTADO:
            raise Parar(f"Falta el paso anterior: ejecuta sin errores la celda <b>{PASOS[clave]}</b> "
                        "y después vuelve a esta.")


# Qué deja de valer cuando se vuelve a ejecutar la celda que produce cada clave
DEPENDEN = {"video": ("primer_fotograma", "zonas", "aforo", "linea", "permanencia", "resumen"),
            "confianza": ("aforo", "linea", "permanencia", "resumen"),
            "zonas": ("aforo", "linea", "permanencia", "resumen"),
            "aforo": ("permanencia", "resumen"), "linea": ("resumen",), "permanencia": ("resumen",), "resumen": ()}


def reiniciar(clave):
    """Borra el resultado de esta celda y lo que depende de él: si la celda falla, nadie usa datos viejos."""
    for c in (clave, *DEPENDEN[clave]):
        ESTADO.pop(c, None)


def es(x, dec=1):
    """Número con formato español: 1.234,5"""
    return f"{x:,.{dec}f}".replace(",", " ").replace(".", ",").replace(" ", ".")


def tabla(filas, cabecera=None):
    estilo = "padding:4px 14px;text-align:left"
    cab = "<tr>" + "".join(f"<th style='{estilo}'>{c}</th>" for c in cabecera) + "</tr>" if cabecera else ""
    cuerpo = "".join("<tr>" + "".join(f"<td style='{estilo}'>{c}</td>" for c in fila) + "</tr>" for fila in filas)
    display(HTML(f"<table style='font-size:15px;border-collapse:collapse'>{cab}{cuerpo}</table>"))


def mostrar(fig):
    """Muestra una figura como imagen y la cierra (así no sale repetida)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    display(Image(data=buf.getvalue()))


def a_ascii(texto):
    """Dentro del vídeo OpenCV solo sabe escribir letras sin tilde."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()


def hex_a_bgr(color):
    return tuple(int(color[i:i + 2], 16) for i in (5, 3, 1))


# ---------- Vídeo ----------
def reducir(f, lado_max=1280):
    """Lado largo ≤ 1280 px y dimensiones pares: las zonas en % valen igual y todo va más rápido."""
    alto, ancho = f.shape[:2]
    escala = min(1.0, lado_max / max(alto, ancho))
    nuevo = (round(ancho * escala / 2) * 2, round(alto * escala / 2) * 2)
    return f if nuevo == (ancho, alto) else cv2.resize(f, nuevo, interpolation=cv2.INTER_AREA)


def fotogramas(v):
    """Lee el tramo elegido desde el archivo, sin guardarlo en memoria: (segundos desde el inicio, fotograma)."""
    cap = cv2.VideoCapture(v["ruta"])
    if v["inicio_f"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, v["inicio_f"])
    try:
        for i in range(v["tramo"]):
            if i % v["salto"]:
                if not cap.grab():  # fotograma saltado: se descodifica pero no se analiza
                    break
                continue
            leido, f = cap.read()
            if not leido:
                break
            yield i / v["fps"], reducir(f)
    finally:
        cap.release()


def rotulo(img, texto, x, y, escala=1.1, fondo=(40, 40, 40)):
    """Texto blanco sobre un recuadro oscuro; (x, y) es la esquina superior izquierda."""
    texto = a_ascii(texto)
    grosor = 2 if escala >= 0.8 else 1
    (an, al), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grosor)
    x = int(min(max(x, 0), img.shape[1] - an - 12))
    y = int(min(max(y, 0), img.shape[0] - al - base - 12))
    cv2.rectangle(img, (x, y), (x + an + 12, y + al + base + 12), fondo, -1)
    cv2.putText(img, texto, (x + 6, y + al + 6), cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), grosor, cv2.LINE_AA)


def crear_solucion(clase, region, **extra):
    """Una solución nueva en cada ejecución: contadores y seguimiento empiezan de cero."""
    sol = clase(model=str(RUTA_PESOS), region=region, classes=[PERSONA], conf=ESTADO["confianza"],
                tracker="bytetrack.yaml", imgsz=640, show_conf=False, line_width=3, verbose=False, **extra)
    sol.names = {**sol.names, PERSONA: "persona"}
    sol.adjust_box_label = lambda cls, conf, track_id=None: f"ID {track_id}"  # etiqueta de cada caja
    return sol


def a_h264(entrada, salida):
    """Convierte el vídeo de OpenCV a H.264, el formato que reproducen los navegadores."""
    if not FFMPEG or not Path(entrada).exists():
        return False
    r = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(entrada), "-an",
                        "-vf", "scale=-2:'min(720,trunc(ih/2)*2)'", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "28", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(salida)],
                       capture_output=True, text=True)
    Path(entrada).unlink(missing_ok=True)
    return r.returncode == 0


def mostrar_video(ruta, ancho):
    mb = Path(ruta).stat().st_size / 1e6
    if mb > 20:
        aviso(f"El vídeo anotado ocupa {es(mb)} MB y no lo muestro aquí para no bloquear el navegador. "
              f"Está en <code>{ruta}</code> (panel Archivos, a la izquierda).")
        return
    b64 = base64.b64encode(Path(ruta).read_bytes()).decode()
    display(HTML(f'<video controls muted loop playsinline width="{ancho}" style="max-width:100%" src="data:video/mp4;base64,{b64}">'
                 'Tu navegador no puede reproducir este vídeo.</video>'))


def pasada(sol, v, nombre, por_fotograma):
    """Recorre el tramo una vez con una solución de Ultralytics y prepara el vídeo anotado."""
    CARPETA_TMP.mkdir(parents=True, exist_ok=True)
    tmp, final = CARPETA_TMP / f"{nombre}_opencv.mp4", CARPETA_TMP / f"{nombre}.mp4"
    escritor = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), v["fps"] / v["salto"], (v["ancho"], v["alto"]))
    linea = display(HTML("Preparando el detector…"), display_id=True)
    t0, n = time.perf_counter(), 0
    for t, f in fotogramas(v):
        r = sol(f)  # detecta, sigue y dibuja sobre el fotograma
        por_fotograma(t, r)
        escritor.write(r.plot_im)
        n += 1
        if n % 10 == 0:
            linea.update(HTML(f"<span style='font-size:15px'>Fotograma {n} de {v['n_proc']} · "
                              f"{es(time.perf_counter() - t0)} s</span>"))
    escritor.release()
    if n == 0:
        raise Parar("No se ha podido leer ningún fotograma del tramo. Vuelve a ejecutar la celda 2.2.")
    seg_bucle = time.perf_counter() - t0
    linea.update(HTML(f"<span style='font-size:15px'>Procesados {n} fotogramas en {es(seg_bucle)} s "
                      f"({es(n / seg_bucle)} fotogramas por segundo).</span>"))
    video = final if a_h264(tmp, final) else None
    return n, seg_bucle, time.perf_counter() - t0, video


# ---------- Instalación, importaciones y pesos del detector ----------
with celda():
    PAQUETES = {"ultralytics": "ultralytics==8.4.161", "supervision": "supervision==0.30.5",
                "lap": "lap>=0.5.12", "shapely": "shapely>=2.0.0"}
    faltan = [pip for modulo, pip in PAQUETES.items() if importlib.util.find_spec(modulo) is None]
    if shutil.which("ffmpeg") is None and importlib.util.find_spec("imageio_ffmpeg") is None:
        faltan.append("imageio-ffmpeg")
    if faltan:
        info("Instalando " + ", ".join(faltan) + ". La primera vez tarda 1-2 minutos.")
        fijos = []  # lo que ya trae el entorno no se toca: así no hay que reiniciar la sesión
        for p in ("numpy", "torch", "torchvision", "opencv-python", "opencv-python-headless", "pandas",
                  "matplotlib", "tensorflow", "keras"):
            try:
                fijos.append(f"{p}=={version(p)}")
            except PackageNotFoundError:
                pass
        restricciones = Path(tempfile.gettempdir()) / "restricciones_nb2.txt"
        restricciones.write_text("\n".join(fijos))
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *faltan, "-c", str(restricciones)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise Parar("No se pudo instalar " + ", ".join(faltan) + ". Vuelve a ejecutar esta celda. Si falla otra vez, "
                        "ve a <i>Entorno de ejecución → Desconectar y eliminar entorno</i> y empieza de nuevo."
                        f"<pre style='font-size:12px;white-space:pre-wrap'>{html.escape(r.stderr[-800:])}</pre>")
        importlib.invalidate_caches()

    os.environ["YOLO_VERBOSE"] = "False"      # sin mensajes por cada fotograma
    os.environ["YOLO_AUTOINSTALL"] = "False"  # nada se instala a mitad de clase
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import ultralytics
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    from ultralytics import YOLO, solutions
    from ultralytics.utils import LOGGER, torch_utils
    from ultralytics.utils.events import events

    LOGGER.setLevel(logging.ERROR)
    events.enabled = False  # sin estadísticas de uso hacia Ultralytics
    # Fuera de Colab, tqdm avisa de que no puede dibujar su barra de progreso gráfica: inofensivo
    warnings.filterwarnings("ignore", message="IProgress not found")
    if HILOS_ENTORNO and not EN_COLAB:  # solo en local: permite medir con los hilos de Colab gratuito
        torch_utils.NUM_THREADS = int(HILOS_ENTORNO)

    FFMPEG = shutil.which("ffmpeg")
    if FFMPEG is None:
        try:
            import imageio_ffmpeg
            FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            FFMPEG = None

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    COMA = FuncFormatter(lambda x, _: f"{x:g}".replace(".", ","))  # decimales con coma en los ejes
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "legend.fontsize": 12, "figure.dpi": 100, "savefig.dpi": 100,
                         "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})

    if not RUTA_PESOS.exists():
        info("Descargando el detector YOLO26 nano (5 MB)…")
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            modelo = YOLO(str(RUTA_PESOS))  # descarga a pesos/yolo26n.pt si no está
    except Exception as e:
        RUTA_PESOS.unlink(missing_ok=True)
        raise Parar("No se pudieron descargar o abrir los pesos del detector (<code>pesos/yolo26n.pt</code>). "
                    "Comprueba la conexión a internet y vuelve a ejecutar esta celda."
                    f"<br><small>Detalle técnico: {html.escape(type(e).__name__)}</small>")

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no hay: todo irá en CPU"
    version_ffmpeg = (subprocess.run([FFMPEG, "-version"], capture_output=True, text=True).stdout.split() + ["?"] * 3)[2] \
        if FFMPEG else "no encontrado: no se podrán ver los vídeos anotados"
    tabla([["Python", platform.python_version()], ["ultralytics", ultralytics.__version__],
           ["supervision", version("supervision")], ["torch", torch.__version__], ["OpenCV", cv2.__version__],
           ["numpy", np.__version__], ["pandas", pd.__version__], ["ffmpeg", version_ffmpeg],
           ["Tarjeta gráfica (GPU)", gpu],
           ["Procesador (CPU)", f"{torch_utils.get_cpu_info()} · {os.cpu_count()} núcleos lógicos"],
           ["Hilos de CPU para el detector", torch_utils.NUM_THREADS], ["Semilla", 42]],
          cabecera=["", "Versión / dato"])
    ESTADO["modelo"] = modelo
    ok("Preparación terminada. Sigue con la celda 2.2.")
''')

# ---------------------------------------------------------------------------------------------
# 2.2 Elegir el vídeo
# ---------------------------------------------------------------------------------------------
md(r'''
## 2.2 Elige el vídeo

Elige una demo o sube tu vídeo, cuántos segundos procesar y cada cuántos fotogramas mira el detector. Un **fotograma** es cada una de las imágenes fijas que forman el vídeo (unas 25-30 por segundo). Con «saltar fotogramas = 2» el detector mira uno de cada dos: va casi el doble de rápido y los recuentos suelen cambiar poco.
''')

paso("2.2 Elige el vídeo", r'''
# @markdown Elige el vídeo y cuánto procesar. Luego pulsa ▶.
fuente_video = "Demo 1: personas caminando (interior, vista cenital)"  # @param ["Demo 1: personas caminando (interior, vista cenital)", "Demo 2: andén de metro (interior)", "Subir mi vídeo"]
segundos_a_procesar = 10  # @param {type:"slider", min:5, max:30, step:1}
saltar_fotogramas = 2  # @param {type:"slider", min:1, max:5, step:1}
''', r'''
requiere("modelo")
reiniciar("video")

# Demos de supervision. Empiezan donde hay algo que contar; «puerta» es la línea recomendada (en %).
DEMOS = {
    "Demo 1: personas caminando (interior, vista cenital)":
        {"fichero": "people-walking.mp4", "mb": 8, "inicio_s": 0, "puerta": "0,50; 100,50", "caras": False},
    "Demo 2: andén de metro (interior)":
        {"fichero": "subway.mp4", "mb": 128, "inicio_s": 22, "puerta": "62,15; 22,100", "caras": True},
}
segundos, salto = float(segundos_a_procesar), int(saltar_fotogramas)
if segundos <= 0 or salto < 1:
    raise Parar("Los segundos a procesar y el salto de fotogramas tienen que ser mayores que 0.")
CARPETA_VIDEOS.mkdir(parents=True, exist_ok=True)

if fuente_video in DEMOS:
    demo = DEMOS[fuente_video]
    ruta = CARPETA_VIDEOS / demo["fichero"]
    if not ruta.exists():
        info(f"Descargando «{demo['fichero']}» ({demo['mb']} MB). Solo la primera vez.")
        try:
            from supervision.assets import download_assets
            logging.getLogger("supervision.assets.downloader").setLevel(logging.WARNING)  # tras importarlo
            with contextlib.redirect_stderr(io.StringIO()):
                download_assets(demo["fichero"], directory=str(CARPETA_VIDEOS))
        except Exception:
            try:  # plan B: el mismo archivo, directamente del servidor de Roboflow
                parcial = CARPETA_VIDEOS / (demo["fichero"] + ".parcial")
                urllib.request.urlretrieve(
                    f"https://media.roboflow.com/supervision/video-examples/{demo['fichero']}", parcial)
                parcial.replace(ruta)
            except Exception:
                raise Parar("No he podido descargar el vídeo de demostración. Comprueba la conexión a internet y "
                            "vuelve a ejecutar la celda, o elige «Subir mi vídeo».")
    inicio_s, puerta, caras = demo["inicio_s"], demo["puerta"], demo["caras"]
elif fuente_video == "Subir mi vídeo":
    ruta = CARPETA_VIDEOS / "mi_video.mp4"
    if ruta.exists():
        info("Uso el vídeo que ya subiste (<code>datos/videos/mi_video.mp4</code>). Para cambiarlo, bórralo "
             "en el panel Archivos y vuelve a ejecutar esta celda.")
    elif colab_files is None:
        raise Parar("Fuera de Colab no hay botón de subida. Copia tu vídeo con este nombre exacto: "
                    f"<code>{html.escape(str(ruta.resolve()))}</code> y vuelve a ejecutar esta celda.")
    else:
        info("Pulsa el botón de subida que aparece debajo y elige un vídeo corto (10-30 s, MP4, mejor de menos "
             "de 50 MB). Sin personas reconocibles, salvo que tengas base legal para usarlo.")
        subidos = colab_files.upload()
        if not subidos:
            raise Parar("No se ha subido ningún archivo. Vuelve a ejecutar la celda y elige un vídeo.")
        shutil.move(next(iter(subidos)), ruta)
        del subidos
    inicio_s, puerta, caras = 0, "0,50; 100,50", False
else:
    raise Parar(f"No reconozco la opción «{html.escape(str(fuente_video))}». Elige una del desplegable.")

cap = cv2.VideoCapture(str(ruta))
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ancho0, alto0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
if not ancho0 or not alto0:
    raise Parar(f"No puedo abrir <code>{html.escape(str(ruta))}</code>. Si es una demo, bórralo en el panel Archivos "
                "y vuelve a ejecutar la celda para descargarlo de nuevo; si es tu vídeo, prueba con un MP4 (H.264).")
if not 1 <= fps <= 240:  # algunos vídeos no guardan bien los fps
    aviso("El vídeo no indica bien sus fotogramas por segundo. Supongo 25.")
    fps = 25.0
if total <= 0:
    total = int(fps * (inicio_s + segundos))

inicio_f = int(round(inicio_s * fps))
pedidos = int(round(segundos * fps))
tramo = min(pedidos, total - inicio_f)
if tramo <= 0:
    raise Parar("El vídeo es más corto que el punto de inicio. Elige otro vídeo.")
if tramo < pedidos:
    aviso(f"Desde el segundo {inicio_s} el vídeo solo tiene {es(tramo / fps)} s: proceso ese tramo entero "
          f"en lugar de {es(segundos, 0)} s.")

v = {"etiqueta": fuente_video, "fichero": ruta.name, "ruta": str(ruta), "fps": fps, "total": total,
     "inicio_s": inicio_s, "inicio_f": inicio_f, "tramo": tramo, "salto": salto,
     "n_proc": (tramo + salto - 1) // salto, "segundos": tramo / fps, "puerta": puerta}
primero = next(fotogramas(v), None)
if primero is None:
    raise Parar(f"No puedo leer los fotogramas de «{ruta.name}». Prueba con un MP4 (H.264).")
primer = primero[1]
v["alto"], v["ancho"] = primer.shape[:2]
ESTADO["video"], ESTADO["primer_fotograma"] = v, primer

tabla([["Vídeo", f"{ruta.name} ({html.escape(fuente_video)})"],
       ["Resolución original", f"{ancho0}×{alto0} px"],
       ["Resolución de proceso", f"{v['ancho']}×{v['alto']} px (lado largo de 1280 px como máximo)"],
       ["Fotogramas por segundo (fps)", es(fps, 2)],
       ["Duración total", f"{es(total / fps)} s"],
       ["Tramo que se procesa", f"del segundo {es(inicio_s, 0)} al {es(inicio_s + v['segundos'])} ({es(v['segundos'])} s)"],
       ["Fotogramas que mirará el detector", f"{v['n_proc']} (uno de cada {salto})"],
       ["fps efectivos tras saltar", es(fps / salto)]])

vertical = v["alto"] > v["ancho"]
fig, ax = plt.subplots(figsize=(5.4, 9.6) if vertical else (10.5, 6))
ax.imshow(primer[..., ::-1])
ax.set_title(f"Primer fotograma del tramo (segundo {es(inicio_s, 0)})")
ax.axis("off")
mostrar(fig)
if caras:
    aviso("En este vídeo se ven caras de pasajeros cerca de la cámara. Lo usamos solo para la demostración. "
          "El CSV y el JSON de 2.8 solo llevan números, pero los vídeos anotados de 2.5 y 2.6 muestran esas caras: "
          "quedan en <code>datos/tmp/</code> y dentro del cuaderno. Borra esas salidas antes de guardarlo o compartirlo.")
ok("Vídeo listo. Sigue con la celda 2.3.")
''')

md(r'''
> **Nota para el profesor: un clip de cafetería propio.** Busca en [Pexels](https://www.pexels.com/search/videos/coffee%20shop/) («coffee shop», «cafe people», «overhead people»); por ejemplo, [este clip de una cafetería llena](https://www.pexels.com/video/a-coffee-shop-restaurant-full-of-customers-3135924/). Elige cámara fija y alta, cuerpos enteros, formato horizontal y 10-30 s, y descárgalo en 1280×720 o 1920×1080 (no 4K). La licencia de Pexels permite usarlo gratis y sin atribución, pero Pexels no garantiza que las personas grabadas hayan dado su consentimiento: licencia libre no equivale a consentimiento. Para usarlo, elige «Subir mi vídeo» en 2.2: en Colab aparece un botón de subida (si la sesión se reinicia, hay que volver a subirlo); fuera de Colab, copia el archivo en `datos/videos/mi_video.mp4`. Pruébalo antes de clase: con la cámara a la altura de los ojos hay más oclusiones y cambios de ID que en las demos. Si vas a usar la demo 2 (128 MB), ejecútala una vez antes de empezar para que ya esté descargada en la sesión.
''')

# ---------------------------------------------------------------------------------------------
# 2.3 Qué ve el detector
# ---------------------------------------------------------------------------------------------
md(r'''
## 2.3 Qué ve el detector en un fotograma

YOLO26 nano es un detector preentrenado que reconoce 80 tipos de objeto; «persona» es el tipo 0 y es el único que usamos. Por dentro es mayoritariamente convolucional, como la red del Notebook 1, pero con unos 2,6 millones de parámetros y un par de bloques de atención. La primera parte (el *backbone*) saca mapas de activación de la imagen; las **cabezas de detección** convierten esos mapas en cajas, cada una con una **confianza** entre 0 y 1.

Mueve la confianza mínima y vuelve a ejecutar: solo cuentan las cajas que la superan.
''')

paso("2.3 Qué ve el detector", r'''
# @markdown Solo se cuentan las cajas con una confianza igual o mayor que este umbral.
confianza_minima = 0.35  # @param {type:"slider", min:0.1, max:0.9, step:0.05}
''', r'''
requiere("modelo", "video")
reiniciar("confianza")
from matplotlib.patches import Rectangle

conf = round(float(confianza_minima), 2)
if not 0 < conf < 1:
    raise Parar("La confianza mínima tiene que estar entre 0 y 1 (por ejemplo, 0.35).")
ESTADO["confianza"] = conf
v, primer, modelo = ESTADO["video"], ESTADO["primer_fotograma"], ESTADO["modelo"]

# Una sola predicción con umbral muy bajo; de ella salen las cajas y la curva para cualquier umbral.
modelo.predict(primer, classes=[PERSONA], conf=0.05, verbose=False)  # la primera llamada prepara el modelo
tiempos = []
for _ in range(3):  # se cronometra 3 veces y se toma la más rápida (la medida más estable)
    t0 = time.perf_counter()
    res = modelo.predict(primer, classes=[PERSONA], conf=0.05, verbose=False)[0]
    tiempos.append(time.perf_counter() - t0)
t_inferencia = min(tiempos)
cajas = res.boxes.xyxy.cpu().numpy()
confs = res.boxes.conf.cpu().numpy()
vistas = confs >= conf

vertical = v["alto"] > v["ancho"]
fig, ax = plt.subplots(figsize=(5.4, 9.6) if vertical else (10.5, 6))
ax.imshow(primer[..., ::-1])
for (x0, y0, x1, y1), c in zip(cajas[vistas], confs[vistas]):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#E69F00", linewidth=2))
    ax.text(x0, y0 - 3, es(c, 2), color="black", fontsize=12, bbox={"facecolor": "#E69F00", "edgecolor": "none", "pad": 1})
ax.set_title(f"{int(vistas.sum())} personas con confianza ≥ {es(conf, 2)}")
ax.axis("off")
mostrar(fig)

umbrales = np.round(np.arange(0.1, 0.91, 0.05), 2)
recuento = [int((confs >= u).sum()) for u in umbrales]
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(umbrales, recuento, "o-", color="#0072B2", linewidth=2.2)
ax.axvline(conf, color="#D55E00", linestyle="--", linewidth=2, label=f"tu umbral: {es(conf, 2)}")
ax.set(xlabel="Confianza mínima (umbral)", ylabel="Personas detectadas",
       title="Personas detectadas en este fotograma según el umbral")
ax.xaxis.set_major_formatter(COMA)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_ylim(bottom=0)
ax.legend()
mostrar(fig)

bajo, alto_ = int((confs >= 0.1).sum()), int((confs >= 0.7).sum())
info(f"Con {es(conf, 2)} el detector ve <b>{int(vistas.sum())}</b> personas. Con un umbral bajo (0,10) vería "
     f"{bajo}: más cajas, y algunas pueden ser falsas (falsos positivos). Con uno alto (0,70) vería {alto_}: "
     "menos errores, pero se pierden personas reales (falsos negativos).")
if conf < 0.25:
    aviso("Por debajo de 0,25 el seguimiento de 2.5 y 2.6 no abre IDs nuevos para las cajas más dudosas: "
          "allí contará menos personas de las que ves aquí.")

# Duración estimada de 2.5 + 2.6: inferencia medida aquí + lectura del vídeo + arranque y conversión.
lector = fotogramas(v)
next(lector)  # el primero incluye el salto al inicio del tramo: no cuenta
t0 = time.perf_counter()
leidos = sum(1 for _ in itertools.islice(lector, 6))
t_lectura = (time.perf_counter() - t0) / max(leidos, 1)
lector.close()
estimacion = 2 * (v["n_proc"] * (t_inferencia * 1.1 + t_lectura + 0.01) + 2.5)
texto = (f"Una detección tarda {es(t_inferencia * 1000, 0)} ms en este ordenador. Estimación para las celdas "
         f"2.5 y 2.6 con estos ajustes: unos <b>{es(estimacion, 0)} s</b> en total.")
if estimacion > 100:
    aviso(texto + " Es mucho para clase: sube <b>saltar_fotogramas</b> a 3 o baja los segundos en la celda 2.2, "
          "y vuelve a ejecutar 2.2 y 2.3.")
else:
    info(texto)
''')

# ---------------------------------------------------------------------------------------------
# 2.4 Zonas y línea
# ---------------------------------------------------------------------------------------------
md(r'''
## 2.4 Zonas y línea de puerta, sin dibujar

Las zonas se escriben como puntos `x,y` en **porcentaje** de la imagen: `0,0` es la esquina de arriba a la izquierda y `100,100` la de abajo a la derecha. Separa los puntos con `;` y escríbelos en orden, recorriendo el borde. Una zona necesita al menos 3 puntos; la línea de puerta, exactamente 2. Puedes renombrar las zonas (por ejemplo, «Barra» y «Mesas»).
''')

paso("2.4 Zonas y línea de puerta", r'''
# @markdown Zonas en % de la imagen (x,y; x,y; …). Por defecto, mitad izquierda y mitad derecha.
nombre_zona_1 = "Izquierda"  # @param {type:"string"}
zona_1 = "0,0; 50,0; 50,100; 0,100"  # @param {type:"string"}
nombre_zona_2 = "Derecha"  # @param {type:"string"}
zona_2 = "50,0; 100,0; 100,100; 50,100"  # @param {type:"string"}
# @markdown Línea de puerta: dos puntos, o «auto» para usar la recomendada para cada demo.
linea_puerta = "auto"  # @param {type:"string"}
invertir_sentido = False  # @param {type:"boolean"}
''', r'''
requiere("video")
from matplotlib.patches import Polygon as PoligonoDibujo
from shapely.geometry import LineString, Polygon

reiniciar("zonas")
v, primer = ESTADO["video"], ESTADO["primer_fotograma"]
EJEMPLO_ZONA, EJEMPLO_LINEA = "0,0; 50,0; 50,100; 0,100", "0,50; 100,50"


def leer_puntos(texto, que, ejemplo):
    """Convierte «x,y; x,y; …» (en %) en una lista de puntos, con errores en castellano."""
    puntos = []
    for trozo in [t.strip() for t in str(texto).replace("(", "").replace(")", "").split(";") if t.strip()]:
        try:
            x, y = (float(n) for n in trozo.split(","))
        except ValueError:
            raise Parar(f"No entiendo {que}: «{html.escape(str(texto))}». El punto «{html.escape(trozo)}» debe tener "
                        f"dos números, x,y (decimales con punto: 12.5). Ejemplo correcto: <code>{ejemplo}</code>")
        if not (0 <= x <= 100 and 0 <= y <= 100):
            raise Parar(f"En {que}, el punto «{html.escape(trozo)}» se sale de la imagen: x e y van de 0 a 100 (%). "
                        f"Ejemplo correcto: <code>{ejemplo}</code>")
        puntos.append((x, y))
    return puntos


zonas_pct = {}
for i, (nombre, texto) in enumerate([(nombre_zona_1, zona_1), (nombre_zona_2, zona_2)], start=1):
    nombre = str(nombre).strip() or f"Zona {i}"
    que = f"la zona {i} («{html.escape(nombre)}»)"
    Que = "L" + que[1:]  # al principio de frase
    puntos = leer_puntos(texto, que, EJEMPLO_ZONA)
    if len(puntos) < 3:
        raise Parar(f"{Que} tiene {len(puntos)} punto(s) y necesita al menos 3. "
                    f"Ejemplo correcto: <code>{EJEMPLO_ZONA}</code>")
    poligono = Polygon(puntos)
    if not poligono.is_valid or poligono.area < 1:
        raise Parar(f"{Que} no forma una zona válida: sus lados se cruzan o no tiene superficie. "
                    f"Escribe los puntos en orden, recorriendo el borde. Ejemplo correcto: <code>{EJEMPLO_ZONA}</code>")
    if nombre.lower() in (n.lower() for n in zonas_pct):
        raise Parar("Las dos zonas se llaman igual. Cambia uno de los nombres.")
    zonas_pct[nombre] = puntos

auto = str(linea_puerta).strip().lower() == "auto"
texto_linea = v["puerta"] if auto else str(linea_puerta)
linea_pct = leer_puntos(texto_linea, "la línea de puerta", EJEMPLO_LINEA)
if len(linea_pct) != 2:
    raise Parar(f"La línea de puerta necesita exactamente 2 puntos y tiene {len(linea_pct)}. "
                f"Ejemplo correcto: <code>{EJEMPLO_LINEA}</code> (o escribe <code>auto</code>).")

# De % a píxeles de la resolución de proceso (enteros: Ultralytics no admite decimales aquí)
W, H = v["ancho"], v["alto"]
a_px = lambda puntos: [(int(round(x / 100 * (W - 1))), int(round(y / 100 * (H - 1)))) for x, y in puntos]
linea_px = a_px(linea_pct)
if linea_px[0] == linea_px[1]:
    raise Parar(f"Los dos puntos de la línea de puerta («{html.escape(texto_linea)}») caen en el mismo sitio: "
                f"sepáralos. Ejemplo correcto: <code>{EJEMPLO_LINEA}</code> (o escribe <code>auto</code>).")
# Mismo criterio que ObjectCounter: línea más alta que ancha = vertical (cuenta el paso hacia la derecha);
# si no, horizontal (cuenta el paso hacia abajo).
vertical = abs(linea_px[0][0] - linea_px[1][0]) < abs(linea_px[0][1] - linea_px[1][1])
sentido = (1, 0) if vertical else (0, 1)
if invertir_sentido:
    sentido = (-sentido[0], -sentido[1])
entrada_txt = {(1, 0): "hacia la derecha", (-1, 0): "hacia la izquierda", (0, 1): "hacia abajo", (0, -1): "hacia arriba"}[sentido]
ESTADO["zonas"] = {"nombres": list(zonas_pct), "pct": zonas_pct, "px": {n: a_px(p) for n, p in zonas_pct.items()},
                   "linea_texto": texto_linea, "linea_px": linea_px, "sentido": sentido,
                   "invertir": bool(invertir_sentido), "entrada_txt": entrada_txt}

vertical_img = H > W
fig, ax = plt.subplots(figsize=(5.4, 9.6) if vertical_img else (10.5, 6.3))
ax.imshow(primer[..., ::-1], extent=[0, 100, 100, 0])
ax.set_aspect(H / W)
ax.set_xticks(range(0, 101, 20))
ax.set_yticks(range(0, 101, 20))
ax.set_xticks(range(0, 101, 10), minor=True)
ax.set_yticks(range(0, 101, 10), minor=True)
ax.grid(which="both", color="white", alpha=0.6, linewidth=0.8)
ax.set(xlabel="x (% del ancho)", ylabel="y (% del alto)", title="Zonas y línea de puerta (coordenadas en %)")
for (nombre, puntos), color in zip(zonas_pct.items(), OKABE_ITO):
    ax.add_patch(PoligonoDibujo(puntos, closed=True, facecolor=color, alpha=0.4, edgecolor=color, linewidth=3))
    libre = Polygon(puntos).difference(LineString(linea_pct).buffer(8))  # el nombre, fuera de la línea
    c = (Polygon(puntos) if libre.is_empty else libre).representative_point()
    ax.text(c.x, c.y, nombre, ha="center", va="center", fontsize=15, weight="bold",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85})
(x0, y0), (x1, y1) = linea_pct
ax.plot([x0, x1], [y0, y1], color="#D55E00", linewidth=4)
mx, my = (x0 + x1) / 2, (y0 + y1) / 2
punta = (mx + 12 * sentido[0], my + 12 * sentido[1])
ax.annotate("", xy=punta, xytext=(mx, my), arrowprops={"arrowstyle": "-|>", "color": "#D55E00", "lw": 3, "mutation_scale": 25})
ax.text(punta[0] + 2 * sentido[0], punta[1] + 2 * sentido[1] + (0 if sentido[1] else -2), "entrada", color="black",
        fontsize=13, ha="left" if sentido[0] >= 0 else "right", va="center",
        bbox={"boxstyle": "round", "facecolor": "#D55E00", "alpha": 0.85, "edgecolor": "none"})
mostrar(fig)

tabla([[html.escape(n), html.escape(zona_texto)] for n, zona_texto in
       zip(zonas_pct, [str(zona_1), str(zona_2)])] +
      [["Línea de puerta", f"{html.escape(texto_linea)}" + (" (automática para este vídeo)" if auto else "")]],
      cabecera=["", "Puntos en %"])
info(f"Cuenta como <b>entrada</b> quien cruza la línea {entrada_txt}, y como salida quien la cruza al revés"
     + (" (sentido invertido con la casilla)" if invertir_sentido else "") +
     ". Solo cuenta el tramo dibujado de la línea, no su prolongación.")
ok("Zonas listas. Sigue con la celda 2.5.")
''')

# ---------------------------------------------------------------------------------------------
# 2.5 Aforo por zonas
# ---------------------------------------------------------------------------------------------
md(r'''
## 2.5 Aforo por zonas

El vídeo se procesa fotograma a fotograma: el detector encuentra a las personas, el **seguimiento** (*tracking*) les asigna un número de identificación (**ID**) y se cuenta cuántas hay dentro de cada zona. Una persona está en una zona si el centro de su caja cae dentro.
''')

paso("2.5 Aforo por zonas", "", r'''
requiere("modelo", "video", "confianza", "zonas")
reiniciar("aforo")
v, zn = ESTADO["video"], ESTADO["zonas"]

rc = crear_solucion(solutions.RegionCounter, zn["px"])
for region, color in zip(rc.counting_regions, OKABE_ITO):  # mismos colores que la vista previa
    region["region_color"] = hex_a_bgr(color)

serie, fotos_en_zona, dentro_por_fotograma = [], {}, []


def por_fotograma(t, r):
    serie.append({"t_s": round(t, 2), **r.region_counts})  # ocupación de este fotograma
    # Para 2.7: fotogramas que pasa cada ID en cada zona, con el mismo criterio que el contador
    dentro = set()
    for caja, id_ in zip(rc.boxes, rc.track_ids):
        x0, y0, x1, y1 = (float(c) for c in caja)
        centro = rc.Point(((x0 + x1) / 2, (y0 + y1) / 2))
        for region in rc.counting_regions:
            if region["prepared_polygon"].contains(centro):
                dentro.add((region["name"], id_))
    for clave in dentro:
        fotos_en_zona[clave] = fotos_en_zona.get(clave, 0) + 1
    dentro_por_fotograma.append(dentro)
    for nombre, puntos in zn["px"].items():  # RegionCounter no escribe el nombre de la zona
        rotulo(r.plot_im, f"{nombre}: {r.region_counts.get(nombre, 0)}",
               min(p[0] for p in puntos) + 10, min(p[1] for p in puntos) + 10)


n, seg_bucle, seg_total, video = pasada(rc, v, "aforo", por_fotograma)
if video:
    mostrar_video(video, 360 if v["alto"] > v["ancho"] else 854)
else:
    aviso("No se ha podido preparar el vídeo anotado para verlo aquí (falta ffmpeg o ha fallado la conversión). "
          "Los recuentos de abajo son válidos.")

df = pd.DataFrame(serie)
fig, ax = plt.subplots(figsize=(11, 4.2))
for i, (zona, color) in enumerate(zip(zn["nombres"], OKABE_ITO)):
    ax.plot(df["t_s"], df[zona], color=color, linestyle=["-", "--"][i % 2], marker=["o", "s"][i % 2],
            markevery=max(1, len(df) // 12), linewidth=2.2, label=zona)
ax.set(xlabel="Segundos desde el inicio del tramo", ylabel="Personas en la zona",
       title="Personas en cada zona, fotograma a fotograma")
ax.xaxis.set_major_formatter(COMA)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_ylim(bottom=0)
ax.legend()
mostrar(fig)

resumen = {z: {"maximo": int(df[z].max()), "media": round(float(df[z].mean()), 2)} for z in zn["nombres"]}
tabla([[html.escape(z), r_["maximo"], es(r_["media"])] for z, r_ in resumen.items()],
      cabecera=["Zona", "Máximo de personas", "Media de personas"])
info(f"Cada punto de la gráfica es la ocupación de ese instante, no un acumulado: quien se queda quieto cuenta "
     f"en todos los fotogramas en los que está. Tiempo de proceso: {es(seg_total)} s para {n} fotogramas "
     f"({es(n / seg_bucle)} por segundo), incluida la conversión del vídeo.")
if all(r_["maximo"] == 0 for r_ in resumen.values()):
    aviso("No se ha contado a nadie en ninguna zona. Revisa las zonas en 2.4 o baja la confianza en 2.3.")

ESTADO["aforo"] = {"zonas": zn["nombres"], "serie": serie, "resumen": resumen, "fotos_en_zona": fotos_en_zona,
                   # (zona, ID) dentro de su zona en el primer o el último fotograma: estancia recortada
                   "en_bordes": dentro_por_fotograma[0] | dentro_por_fotograma[-1],
                   "fotogramas": n, "segundos": round(seg_total, 1), "confianza": ESTADO["confianza"]}
''')

# ---------------------------------------------------------------------------------------------
# 2.6 Entradas y salidas
# ---------------------------------------------------------------------------------------------
md(r'''
## 2.6 Entradas y salidas por una línea

Para saber si alguien **cruza** la línea hay que seguirlo de un fotograma al siguiente. El seguimiento (aquí, el algoritmo ByteTrack) mantiene el ID de cada persona mientras la ve; así se sabe que la caja de ahora es la misma persona que la de antes y hacia dónde se ha movido. Sin IDs persistentes no se podría distinguir a quien cruza de quien simplemente aparece al otro lado.

Errores típicos:
- **Oclusiones**: si alguien queda tapado por otra persona, el seguimiento puede perderlo y no contar su cruce.
- **Cambio de ID**: si al reaparecer recibe un ID nuevo, la misma persona puede contar dos veces.
- **Doble conteo**: aquí cada ID cuenta una sola vez; quien cruza y vuelve no suma de nuevo, salvo que haya cambiado de ID.
''')

paso("2.6 Entradas y salidas por la línea", "", r'''
requiere("modelo", "video", "confianza", "zonas")
reiniciar("linea")
v, zn = ESTADO["video"], ESTADO["zonas"]

oc = crear_solucion(solutions.ObjectCounter, zn["linea_px"], show_in=False, show_out=False)
(x0, y0), (x1, y1) = zn["linea_px"]
medio = ((x0 + x1) // 2, (y0 + y1) // 2)
punta = (medio[0] + 70 * zn["sentido"][0], medio[1] + 70 * zn["sentido"][1])


def entradas_salidas():
    return (oc.out_count, oc.in_count) if zn["invertir"] else (oc.in_count, oc.out_count)


def por_fotograma(t, r):
    entradas, salidas = entradas_salidas()
    rotulo(r.plot_im, f"Entradas: {entradas}", 10, 10)
    rotulo(r.plot_im, f"Salidas: {salidas}", 10, 60)
    cv2.arrowedLine(r.plot_im, medio, punta, hex_a_bgr("#D55E00"), 4, tipLength=0.35)
    rotulo(r.plot_im, "entrada", punta[0] + 8, punta[1] - 12, escala=0.7, fondo=hex_a_bgr("#D55E00"))


n, seg_bucle, seg_total, video = pasada(oc, v, "linea", por_fotograma)
if video:
    mostrar_video(video, 360 if v["alto"] > v["ancho"] else 854)
else:
    aviso("No se ha podido preparar el vídeo anotado para verlo aquí (falta ffmpeg o ha fallado la conversión). "
          "El recuento de abajo es válido.")

entradas, salidas = entradas_salidas()
ok(f"<b>Entradas: {entradas} · Salidas: {salidas}</b> en {es(v['segundos'])} s de vídeo. Cuenta como entrada "
   f"quien cruza la línea {zn['entrada_txt']}; cada ID cuenta una sola vez.")
info(f"Tiempo de proceso: {es(seg_total)} s para {n} fotogramas ({es(n / seg_bucle)} por segundo), "
     "incluida la conversión del vídeo.")
if entradas + salidas == 0:
    aviso("Nadie ha cruzado la línea en este tramo. Colócala en 2.4 donde la gente camine atravesándola, "
          "o procesa más segundos en 2.2.")
ESTADO["linea"] = {"entradas": entradas, "salidas": salidas, "segundos": round(seg_total, 1)}
''')

# ---------------------------------------------------------------------------------------------
# 2.7 Permanencia
# ---------------------------------------------------------------------------------------------
md(r'''
## 2.7 Avanzado (opcional): tiempo de permanencia

Con los IDs que guardó la celda 2.5 medimos cuántos segundos pasa cada persona dentro de cada zona, sin volver a procesar el vídeo. Solo cuentan las estancias de 1 segundo o más.
''')

paso("2.7 Tiempo de permanencia (opcional)", "", r'''
requiere("video", "aforo")
reiniciar("permanencia")
v, a = ESTADO["video"], ESTADO["aforo"]
seg_por_fotograma = v["salto"] / v["fps"]

fig, ejes = plt.subplots(1, len(a["zonas"]), figsize=(11, 4), sharey=True, squeeze=False)
filas, medias, cortadas_total = [], {}, 0
for eje, zona, color in zip(ejes[0], a["zonas"], OKABE_ITO):
    estancias = {id_: n * seg_por_fotograma for (z, id_), n in a["fotos_en_zona"].items()
                 if z == zona and n * seg_por_fotograma >= 1}
    tiempos = list(estancias.values())
    cortadas = sum(1 for id_ in estancias if (zona, id_) in a["en_bordes"])
    cortadas_total += cortadas
    if tiempos:
        eje.hist(tiempos, bins=np.arange(1, max(tiempos) + 2), color=color, edgecolor="white")
        medias[zona] = round(float(np.mean(tiempos)), 1)
        filas.append([html.escape(zona), len(tiempos), es(float(np.median(tiempos))), es(medias[zona]),
                      es(max(tiempos)), cortadas])
    else:
        eje.text(0.5, 0.5, "Nadie ha estado 1 s o más", ha="center", va="center", transform=eje.transAxes)
        medias[zona] = None
        filas.append([html.escape(zona), 0, "–", "–", "–", 0])
    eje.set(title=zona, xlabel="Segundos dentro de la zona")
    eje.xaxis.set_major_formatter(COMA)
    eje.yaxis.set_major_locator(MaxNLocator(integer=True))
ejes[0][0].set_ylabel("Número de IDs (≈ personas)")
mostrar(fig)

tabla(filas, cabecera=["Zona", "IDs con 1 s o más", "Mediana (s)", "Media (s)", "Máximo (s)",
                       "Ya estaban al empezar o seguían al acabar"])
aviso(f"Dos cosas acortan estas cifras. {cortadas_total} estancia(s) ya estaban en curso al empezar el tramo o seguían "
      "al terminarlo, así que su duración real es mayor. Y si el seguimiento cambia el ID de alguien, su estancia se "
      "parte en dos más cortas.")
ESTADO["permanencia"] = medias
''')

# ---------------------------------------------------------------------------------------------
# 2.8 Exportar y webhook
# ---------------------------------------------------------------------------------------------
md(r'''
## 2.8 Exportar para automatizar

Guardamos dos archivos pequeños en `salidas/nb2/`: un CSV con las personas de cada zona en cada instante y un JSON con el resumen. Ninguno contiene imágenes ni IDs.
''')

paso("2.8 Exportar CSV y JSON", "", r'''
requiere("video", "aforo")
reiniciar("resumen")
v, a = ESTADO["video"], ESTADO["aforo"]
linea, permanencia = ESTADO.get("linea"), ESTADO.get("permanencia")
CARPETA_SALIDAS.mkdir(parents=True, exist_ok=True)

ruta_csv, ruta_json = CARPETA_SALIDAS / "aforo_por_zonas.csv", CARPETA_SALIDAS / "resumen.json"
df = pd.DataFrame(a["serie"])[["t_s", *a["zonas"]]]
df.to_csv(ruta_csv, sep=";", decimal=",", index=False, encoding="utf-8-sig")
resumen = {
    "video": v["fichero"],
    "inicio_s": v["inicio_s"],
    "resolucion_proceso": f"{v['ancho']}x{v['alto']}",
    "fps_video": round(v["fps"], 2),
    "saltar_fotogramas": v["salto"],
    "segundos_procesados": round(v["segundos"], 2),
    "fotogramas_procesados": a["fotogramas"],
    "confianza_minima": a["confianza"],
    "zonas": a["resumen"],
    "entradas": linea["entradas"] if linea else None,
    "salidas": linea["salidas"] if linea else None,
    "permanencia_media_s": permanencia,
    "tiempos_proceso_s": {"aforo": a["segundos"], "linea": linea["segundos"] if linea else None},
    "generado_en": datetime.now().isoformat(timespec="seconds"),
    "nota": "Solo agregados: sin imágenes ni identificadores.",
}
ruta_json.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
ESTADO["resumen"] = resumen

ok(f"Guardados <code>{ruta_csv}</code> ({len(df)} filas) y <code>{ruta_json}</code>.")
tabla([[es(fila["t_s"], 2)] + [fila[z] for z in a["zonas"]] for fila in a["serie"][:5]],
      cabecera=["t_s (primeras filas del CSV)", *[html.escape(z) for z in a["zonas"]]])
if not linea:
    aviso("No has ejecutado la celda 2.6: entradas y salidas quedan vacías en el JSON.")
if EN_COLAB and colab_files is not None:
    try:
        for ruta in (ruta_csv, ruta_json):
            colab_files.download(str(ruta))
    except Exception:
        aviso("No se han podido descargar solos. Descárgalos desde el panel Archivos (carpeta salidas/nb2, ⋮ → Descargar).")
else:
    info(f"Fuera de Colab los archivos quedan en <code>{html.escape(str(CARPETA_SALIDAS.resolve()))}</code>.")
info("Con muy pocas personas o intervalos muy cortos, incluso un recuento puede ayudar a saber quién era "
     "(por ejemplo, «a las 9:02 entró 1 persona» junto con el cuadrante de turnos). Un agregado solo deja de ser "
     "dato personal si de verdad no permite señalar a nadie.")
''')

paso("2.8 (opcional) Enviar el resumen a n8n", r'''
# @markdown Pega la URL del nodo Webhook de n8n. Si lo dejas vacío, no se envía nada.
webhook_n8n = ""  # @param {type:"string"}
''', r'''
requiere("resumen")
resumen = ESTADO["resumen"]
display(HTML("<p style='font-size:15px'>Esto es lo que recibiría n8n (en el nodo Webhook llega dentro de "
             "<code>body</code>):</p><pre style='font-size:13px'>"
             + html.escape(json.dumps(resumen, ensure_ascii=False, indent=2)) + "</pre>"))
url = str(webhook_n8n).strip()
if not url:
    info("No se envía nada: el campo <b>webhook_n8n</b> está vacío.")
elif not url.startswith(("http://", "https://")):
    raise Parar(f"La dirección «{html.escape(url)}» no parece una URL: tiene que empezar por https:// (o http://). "
                "Cópiala entera desde el nodo Webhook de n8n.")
else:
    peticion = urllib.request.Request(url, data=json.dumps(resumen, ensure_ascii=False).encode("utf-8"),
                                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(peticion, timeout=15) as respuesta:
            codigo = respuesta.status
    except urllib.error.HTTPError as e:
        consejo = (" n8n no encuentra ese webhook. Si usas la URL de prueba (contiene «webhook-test»), n8n solo la "
                   "escucha mientras el nodo Webhook espera un evento de prueba; con la de producción, el flujo tiene "
                   "que estar activado." if e.code == 404 else " Revisa la configuración del nodo Webhook (método POST).")
        raise Parar(f"n8n ha respondido con el código {e.code}.{consejo}")
    except (urllib.error.URLError, OSError, ValueError):
        raise Parar("No he podido conectar con esa dirección. Comprueba que la URL está completa y que n8n está "
                    "en marcha, y vuelve a ejecutar la celda.")
    ok(f"Enviado. n8n ha respondido con el código {codigo}.")
''')

md(r'''
**Un flujo de alerta de aforo en n8n** (solo la idea; no lo construimos aquí):
1. **Webhook** (método POST): recibe el JSON de arriba. Los datos llegan dentro de `body`.
2. **IF**: comprueba si el máximo de una zona supera el aforo permitido, por ejemplo `{{ $json.body.zonas.Izquierda.maximo }}` mayor que 12.
3. Si lo supera: **aviso** por correo o por Telegram con la zona, el máximo y la hora (`generado_en`).
4. En todos los casos: **registro** de una fila en una hoja de cálculo para ver la evolución por días.
''')

# ---------------------------------------------------------------------------------------------
# 2.9 Cierre
# ---------------------------------------------------------------------------------------------
md(r'''
## 2.9 Cierre

**Límites de este sistema**
- **Oclusiones y multitudes**: cuando unas personas tapan a otras, el detector suele perder a algunas y el recuento se queda corto.
- **Ángulo de cámara y luz**: suele funcionar mejor con la cámara alta y fija y buena luz; a la altura de los ojos hay más oclusiones.
- **Cambios de ID**: parten estancias y pueden duplicar entradas.
- **Perspectiva**: las zonas se dibujan sobre la imagen, no sobre el suelo, y se cuenta por el centro de la caja, no por los pies.
- **Contar no es identificar**: el sistema no sabe quién es nadie, pero el vídeo sí contiene datos personales.

**POC ≠ producción.** Esto es un prototipo: llevarlo a un local real exige medir el error con vídeo propio, decidir dónde se procesa y qué se guarda, y cumplir la normativa de protección de datos.

**Otros sectores** (ideas para el debate, no casos medidos)
- **Retail**: ocupación de la zona de cajas para abrir otra cuando se forma cola.
- **Transporte**: ocupación de andenes para ajustar frecuencias o avisar de aglomeraciones.
- **Eventos**: aforo por zonas y flujo de personas en los accesos.
- **Industria**: aviso cuando alguien entra en la zona de trabajo de una máquina, como apoyo y nunca en lugar de los sistemas de seguridad certificados.

**Preguntas sobre privacidad**
1. Si el sistema solo guarda números, ¿hace falta avisar a los clientes de que hay una cámara? ¿Por qué?
2. ¿Quién debería poder ver el vídeo original, y durante cuánto tiempo?
3. «A las 9:02 entró 1 persona» es un agregado. ¿Podría servir para saber quién era? ¿Cómo lo evitarías?
''')

# ---------------------------------------------------------------------------------------------
for i, c in enumerate(celdas):
    c.id = f"nb2-{i:02d}"
nb = nbformat.v4.new_notebook(cells=celdas, metadata={
    "colab": {"provenance": [], "toc_visible": True},
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
})
nbformat.validate(nb)
DESTINO.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, DESTINO)
print(f"Escrito {DESTINO.relative_to(RAIZ)} ({len(celdas)} celdas)")
