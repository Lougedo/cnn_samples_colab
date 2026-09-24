"""Genera notebooks/01_laboratorio_cnn_radiografias.ipynb con nbformat.

Uso (desde la raíz del repo):  .venv/bin/python src/build_nb1.py
Es determinista: dos ejecuciones producen el mismo fichero byte a byte.
"""
from pathlib import Path
import textwrap

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "notebooks" / "01_laboratorio_cnn_radiografias.ipynb"

# Red del reto C (calibrada en docs/notas_verificacion/g_construccion_nb1.md): con la red por defecto
# 20 épocas no llegan a sobreajustar; sin pooling, con dropout 0, sí (3 de 3 semillas).
RETO_C = {"pooling": False, "epocas": 20}

# ponytail: SEGUNDOS_POR_GMAC (celda 1.1) es una cota fija ~6× lo medido en local, no una medida de Colab; si en
# Colab 1.3 bloquea por tiempo alguna solución que propone, subirla (ver g_construccion_nb1.md, correcciones).

# Valores por defecto de 1.3 (los mismos que DEFECTO en la celda 1.1): se repiten en 1.3 y en 1.9 porque los
# formularios de Colab no tienen botón de restablecer.
RED_DEFECTO = ("`bloques_convolucionales` 2 · `filtros_primer_bloque` 16 · `duplicar_filtros_en_cada_bloque` ☑ · "
               "`tamano_kernel` 3 · `usar_pooling` ☑ · `usar_batchnorm` ☐ · `dropout` 0,2 · `neuronas_capa_densa` 64 · "
               "`compensar_desbalanceo` ☐ · `epocas` 8.")

CELDAS = []


def md(texto):
    CELDAS.append(new_markdown_cell(textwrap.dedent(texto).strip().replace("__RED_DEFECTO__", RED_DEFECTO)))


def codigo(titulo, fuente):
    celda = new_code_cell(f'# @title {titulo} {{ display-mode: "form" }}\n' + textwrap.dedent(fuente).strip())
    celda.metadata = {"cellView": "form"}
    CELDAS.append(celda)


FALTA_PREPARACION = '''print("⚠️ Primero ejecuta la celda 1.1 · Preparación (arriba del todo).")'''

# ---------------------------------------------------------------------------------------------
# 1.0 Portada
# ---------------------------------------------------------------------------------------------
md(r'''
# Cuaderno 1 · Una CNN por dentro con radiografías de tórax

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lougedo/cnn_samples_colab/blob/main/notebooks/01_laboratorio_cnn_radiografias.ipynb)

Vas a diseñar una red neuronal convolucional (CNN) sin escribir código y a entrenarla para distinguir
radiografías de tórax normales de radiografías con neumonía. Después verás qué ha aprendido cada capa
y en qué zonas de la imagen se fija para decidir. Cada uno prueba un cambio de arquitectura y compara
sus resultados con los del resto de la clase en una hoja común.

> ⚠️ **No apto para uso clínico.** PneumoniaMNIST es un conjunto educativo: radiografías de tórax de niños
> reducidas a 28×28 o 64×64 píxeles. Sus autores advierten que a esa resolución se pierde información
> necesaria para diagnosticar. Nada de lo que hagamos aquí sirve para diagnosticar a nadie. Son
> radiografías reales, publicadas por sus autores con licencia abierta (CC BY 4.0).

**Cómo se usa.** El código está oculto: cada celda se ve como un título con un botón ▶ y, debajo, sus controles.
Algo así:

```
 ▶  1.3 Diseña tu red
    bloques_convolucionales:  ──●──────────  2
    usar_pooling:             ☑
    nombre_experimento:       [ INICIALES_RETO ]
```

Cambia los controles (deslizador, casilla, desplegable o texto) y pulsa ▶: el resultado aparece debajo.
Si al pulsar ▶ por primera vez Colab te avisa de que este cuaderno no lo ha creado Google, es normal:
pulsa **Ejecutar de todos modos**. Tarda unos segundos en conectarse.
Ejecuta 1.1 y 1.2. Si en clase te han dado un reto, salta a 1.9 para apuntar el reto y tu hipótesis y vuelve a 1.3;
si no, sigue en orden, de arriba abajo, hasta la 1.9. Si a una celda le falta un paso anterior, te dirá cuál.

**Índice**
1.1 Preparación · 1.2 Los datos · 1.3 Diseña tu red · 1.4 Entrena · 1.5 Evalúa en test ·
1.6 Qué ve cada capa · 1.7 Dónde mira el modelo (Grad-CAM) · 1.8 Registro de experimentos ·
1.9 Retos guiados · 1.10 Plan B (solo el profesor) · 1.11 Cierre

<small>Datos: PneumoniaMNIST (MedMNIST v2; Yang et al., *Scientific Data* 10, 41, 2023,
doi:10.1038/s41597-022-01721-8), derivado de Kermany et al. (*Cell* 172(5), 2018). Licencia
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cambios: solo se pasa el brillo de cada píxel de
la escala 0-255 a la escala 0-1; no se recorta ni se retoca ninguna imagen.</small>
''')

# ---------------------------------------------------------------------------------------------
# 1.1 Preparación
# ---------------------------------------------------------------------------------------------
md(r'''
## 1.1 Preparación
Instala lo que falte (la primera vez tarda algo más), carga las librerías y fija la semilla
aleatoria en 42 para que, en el mismo entorno, repetir un entrenamiento dé lo mismo.
''')

codigo("1.1 Preparación", r'''
import os, sys, io, time, html, platform, importlib, importlib.util, subprocess, tempfile
os.environ["KERAS_BACKEND"] = "tensorflow"   # Keras 3 con TensorFlow por debajo
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"     # oculta los mensajes internos de TensorFlow
from IPython.display import display, HTML, Image

EN_COLAB = "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules


# --- Mensajes para el alumno (se leen bien también con el tema oscuro de Colab) ---
def _recuadro(color, icono, texto):
    display(HTML(
        f'<div style="border-left:6px solid {color}; background:{color}22; color:inherit; padding:10px 14px; '
        f'margin:8px 0; border-radius:4px; font-size:15px; line-height:1.5">{icono}&nbsp; {texto}</div>'))


def error(texto): _recuadro("#D55E00", "⛔", texto)
def aviso(texto): _recuadro("#E69F00", "⚠️", texto)
def ok(texto): _recuadro("#009E73", "✅", texto)
def info(texto): _recuadro("#0072B2", "ℹ️", texto)
def nota(texto): display(HTML(f'<div style="font-size:15px; line-height:1.5; margin:6px 0">{texto}</div>'))


def tabla_html(df):
    display(HTML('<div style="font-size:14px; overflow-x:auto">' + df.to_html(index=False, border=0) + "</div>"))


# --- Instalación: solo lo que falte (en Colab, normalmente solo medmnist) ---
def _instalar_lo_que_falte():
    faltan = [paquete for modulo, paquete in (("medmnist", "medmnist==3.0.2"), ("sklearn", "scikit-learn"),
                                                ("tensorflow", "tensorflow"))
              if importlib.util.find_spec(modulo) is None]
    if not faltan:
        return True
    print("Instalando " + ", ".join(faltan) + "… (solo la primera vez)")
    from importlib.metadata import version, PackageNotFoundError
    fijos = []   # congela lo que ya trae el entorno: pip no podrá actualizarlo (así no hay que reiniciar)
    for paquete in ("numpy", "tensorflow", "keras", "torch", "opencv-python", "pandas", "matplotlib"):
        try:
            fijos.append(f"{paquete}=={version(paquete)}")
        except PackageNotFoundError:
            pass
    if "tensorflow" in faltan:
        fijos = [f for f in fijos if not f.startswith("keras==")]
    restricciones = os.path.join(tempfile.gettempdir(), "restricciones_nb1.txt")
    with open(restricciones, "w") as f:
        f.write("\n".join(fijos) + "\n")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *faltan, "-c", restricciones],
                       capture_output=True, text=True)
    importlib.invalidate_caches()
    if r.returncode != 0:
        detalle = html.escape((r.stderr or r.stdout).strip()[-600:])
        error("No he podido instalar " + ", ".join(faltan) + ". Comprueba la conexión a internet y vuelve a "
              "ejecutar esta celda. Si sigue fallando, en el menú <b>Entorno de ejecución</b> elige "
              "<b>Desconectar y eliminar entorno de ejecución</b> y empieza de nuevo."
              f"<br><small>Detalle técnico: {detalle}</small>")
        return False
    return True


_listo = _instalar_lo_que_falte()
if _listo:
    try:
        import logging
        import numpy as np
        import pandas as pd
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch, Rectangle
        from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter
        import sklearn
        from sklearn.metrics import confusion_matrix, roc_auc_score
        import tensorflow as tf
        import keras
        from keras import layers
        import medmnist
    except ImportError as e:
        _listo = False
        error(f"Falta la librería <b>{html.escape(str(e.name))}</b>. Vuelve a ejecutar esta celda; si sigue "
              "fallando, en el menú <b>Entorno de ejecución</b> elige <b>Desconectar y eliminar entorno de "
              "ejecución</b> y empieza de nuevo.")

# --- Constantes del laboratorio ---
SEMILLA, LOTE = 42, 128
CLASES = ("Normal", "Neumonía")                      # 0 = Normal, 1 = Neumonía
AZUL, NARANJA, VERDE, ROSA, CELESTE, BERMELLON = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"
DEFECTO = dict(bloques=2, filtros=16, duplicar=True, kernel=3, pooling=True, batchnorm=False,
               dropout=0.2, densa=64, compensar=False, epocas=8)
MAX_PARAMETROS = 20_000_000                          # por encima, ni se estima: tardaría del orden de una hora
AVISO_SEGUNDOS, MAX_SEGUNDOS = 90, 600               # aviso si pasa de 90 s; bloqueo si pasa de 10 min
# Cota de tiempo sin construir la red (solo para proponer soluciones en 1.3): segundos por cada 10⁹ multiplicaciones
# × imagen al entrenar. ~6 veces lo medido en local (0,02-0,06) para cubrir la CPU de Colab gratuito.
SEGUNDOS_POR_GMAC = 0.25
UMBRAL_COLAPSO, UMBRAL_SUBIDA, UMBRAL_SEPARACION, UMBRAL_INFRA = 0.97, 0.10, 1.5, 0.90   # diagnóstico (1.4)
CARPETA_SALIDAS = os.path.join("salidas", "nb1")
RUTA_CSV = os.path.join(CARPETA_SALIDAS, "experimentos.csv")
COLUMNAS = ["id", "experimento", "reto", "hipotesis", "resolucion", "bloques", "filtros", "duplicar", "kernel",
            "pooling", "batchnorm", "dropout", "densa", "compensar", "epocas", "parametros",
            "tiempo_entrenamiento_s", "acc_validacion", "accuracy", "sensibilidad", "especificidad", "auc",
            "diagnostico"]
PASOS = {"datos": "1.2 · Los datos", "config": "1.3 · Diseña tu red", "modelo": "1.4 · Entrena la red"}
DONDE = "en la máquina de Colab" if EN_COLAB else "en este ordenador"   # los tiempos se miden donde corre la red


# --- Formato español de números ---
def num(x, decimales=0):
    """52.113 · 0,87"""
    return f"{x:,.{decimales}f}".replace(",", "·").replace(".", ",").replace("·", ".")


def pct(x, decimales=1):
    return num(100 * x, decimales) + " %"


def duracion(segundos):
    return f"{num(segundos)} s" if segundos < 120 else f"{num(segundos / 60)} minutos"


def respuesta(p):
    """Clase que dice la red y la probabilidad que le da a esa respuesta («> 99 %» mejor que un «100 %» redondeado)."""
    pred = int(p >= 0.5)
    q = p if pred else 1 - p
    return CLASES[pred], "> 99 %" if q >= 0.995 else pct(q, 0)


def _millones(v):
    n = f"{v / 1e6:g}".replace(".", ",")
    return f"{n} millón" if v == 1e6 else f"{n} millones"


FORMATO_PCT = FuncFormatter(lambda v, _: num(100 * v, 1).rstrip("0").rstrip(",") + " %") if _listo else None
FORMATO_DEC = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")) if _listo else None
FORMATO_PARAM = FuncFormatter(lambda v, _: _millones(v) if v >= 1e6 else f"{v / 1e3:g} mil".replace(".", ",")
                              if v >= 1e3 else f"{v:g}".replace(".", ",")) if _listo else None


def png(fig):
    """Figura → imagen PNG (se ve igual en Colab y en Jupyter)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    return Image(data=buf.getvalue(), format="png")


def mostrar(fig):
    display(png(fig))
    plt.close(fig)


# --- Comprobaciones de pasos previos ---
def requiere(*claves):
    for clave in claves:
        if clave == "config" and ESTADO["config_invalida"]:
            error("La red de la celda 1.3 no se puede entrenar (mira el mensaje de esa celda). "
                  "Cambia la configuración y vuelve a ejecutar 1.3.")
            return False
        if ESTADO.get(clave) is None:
            error(f"Primero ejecuta la celda <b>{PASOS[clave]}</b>.")
            return False
    return True


def modelo_listo():
    """Hay una red entrenada que sirve con los datos cargados. Dice de qué experimento es y si 1.3 ha cambiado."""
    if not requiere("datos"):
        return False
    if ESTADO["modelo"] is None:
        if ESTADO["config_invalida"]:
            error("Todavía no hay ninguna red entrenada y la configuración de la celda 1.3 no es válida: corrígela "
                  "como dice el mensaje de 1.3 y ejecuta 1.3 y 1.4.")
        else:
            error(f"Primero ejecuta la celda <b>{PASOS['modelo']}</b>.")
        return False
    entreno = ESTADO["entreno"]
    r = entreno["config"]["resolucion"]
    if r != ESTADO["resolucion"]:
        error(f"Tu red se entrenó con imágenes de {r}×{r} px, pero en 1.2 has cargado las de "
              f"{ESTADO['resolucion']}×{ESTADO['resolucion']} px. Vuelve a ejecutar 1.3 y 1.4, "
              f"o pon la resolución a {r} en la celda 1.2.")
        return False
    nota(f"Resultados del experimento n.º {entreno['id']} («{html.escape(entreno['nombre'])}»).")
    if ESTADO["config_invalida"]:
        aviso(f"La configuración actual de 1.3 no se puede entrenar: lo que ves es la red n.º {entreno['id']}, la "
              "última que entrenaste. Corrige 1.3 y ejecuta 1.3 y 1.4 para ver la nueva.")
    elif ESTADO["red_actual"] and ESTADO["red_actual"][0] != diseno(entreno["config"])[1]:
        aviso(f"Has cambiado la red en 1.3 pero aún no la has entrenado: lo que ves es la red n.º {entreno['id']}. "
              "Ejecuta 1.4 para ver la nueva.")
    return True


# --- La red ---
def diseno(c):
    """(arquitectura, diseño completo). El diseño añade lo que no cambia la red pero sí el entrenamiento."""
    arquitectura = tuple(c[x] for x in ("resolucion", "bloques", "filtros", "duplicar", "kernel", "pooling",
                                        "batchnorm", "dropout", "densa"))
    return arquitectura, arquitectura + (c["compensar"], c["epocas"])


def tamanos(res, bloques, kernel, pooling):
    """Lado de la imagen tras cada bloque. Devuelve (trayectoria, bloque donde se queda en 0×0 o None)."""
    lado, trayectoria = res, []
    for i in range(1, bloques + 1):
        lado = lado - kernel + 1          # convolución sin relleno: pierde kernel-1 píxeles de lado
        if lado >= 1 and pooling:
            lado //= 2                    # pooling 2×2: la mitad de lado
        if lado < 1:
            return trayectoria, i
        trayectoria.append(lado)
    return trayectoria, None


def construir_modelo(c):
    """API funcional con capas con nombre (necesario para ver mapas de activación y Grad-CAM)."""
    x = entrada = keras.Input((c["resolucion"], c["resolucion"], 1), name="entrada")
    filtros = c["filtros"]
    for i in range(1, c["bloques"] + 1):
        x = layers.Conv2D(filtros, c["kernel"], padding="valid", name=f"conv_{i}")(x)
        if c["batchnorm"]:
            x = layers.BatchNormalization(momentum=0.9, name=f"bn_{i}")(x)  # 0,99 (por defecto) colapsa aquí
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        if c["pooling"]:
            x = layers.MaxPooling2D(2, name=f"pool_{i}")(x)
        if c["duplicar"]:
            filtros *= 2
    x = layers.Flatten(name="aplanar")(x)
    if c["densa"]:
        x = layers.Dense(c["densa"], activation="relu", name="densa")(x)
    if c["dropout"] > 0:
        x = layers.Dropout(c["dropout"], name="dropout")(x)
    x = layers.Dense(1, name="logit")(x)
    salida = layers.Activation("sigmoid", name="probabilidad")(x)
    modelo = keras.Model(entrada, salida, name="mi_red")
    modelo.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return modelo


def pesos_clase(y):
    """Pesos «balanced»: cada clase pesa lo mismo en total (Normal ≈ 1,94; Neumonía ≈ 0,67)."""
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    return {0: len(y) / (2 * n0), 1: len(y) / (2 * n1)}


def probabilidades(modelo, x, lote=256):
    """Probabilidad de neumonía de cada imagen."""
    return np.concatenate([modelo(x[i:i + lote], training=False).numpy().ravel() for i in range(0, len(x), lote)])


def medir_lotes(modelo):
    """Cronometra 1 lote de calentamiento y 5 lotes normales (mediana: aguanta un pico de carga del ordenador)."""
    xtr, ytr = ESTADO["datos"]["train"]
    inicio = time.perf_counter()
    modelo.train_on_batch(xtr[:LOTE], ytr[:LOTE])
    calentamiento, tiempos = time.perf_counter() - inicio, []
    for i in range(1, 6):
        inicio = time.perf_counter()
        modelo.train_on_batch(xtr[i * LOTE:(i + 1) * LOTE], ytr[i * LOTE:(i + 1) * LOTE])
        tiempos.append(time.perf_counter() - inicio)
    return calentamiento, float(np.median(tiempos))


def segundos_estimados(medida, epocas):
    """Extrapola la medida a todas las épocas (+ validación) y multiplica por 1,2 porque suele quedarse corta."""
    calentamiento, por_lote = medida
    pasos = -(-len(ESTADO["datos"]["train"][1]) // LOTE) + len(ESTADO["datos"]["val"][1]) / LOTE / 3
    return 1.2 * (calentamiento + epocas * pasos * por_lote)


def entrenar(cfg, callbacks=()):
    """Construye la red desde cero con la semilla fija y la entrena. Devuelve (modelo, historial, segundos)."""
    (xtr, ytr), (xva, yva) = ESTADO["datos"]["train"], ESTADO["datos"]["val"]
    keras.utils.set_random_seed(SEMILLA)
    modelo = construir_modelo(cfg)
    inicio = time.perf_counter()
    h = modelo.fit(xtr, ytr, validation_data=(xva, yva), epochs=cfg["epocas"], batch_size=LOTE, verbose=0,
                   class_weight=pesos_clase(ytr) if cfg["compensar"] else None, callbacks=list(callbacks))
    return modelo, {k: [float(v) for v in vs] for k, vs in h.history.items()}, time.perf_counter() - inicio


REGLAS = (f"<b>Reglas del diagnóstico:</b> <i>colapso</i> si en validación dice casi siempre lo mismo (en el "
          f"{pct(UMBRAL_COLAPSO, 0)} de los casos o más) · <i>sobreajuste</i> si la pérdida de validación acaba un "
          f"{pct(UMBRAL_SUBIDA, 0)} o más por encima de su mínimo (y ese mínimo quedó 3 épocas o más antes del "
          f"final) o si acaba siendo {num(UMBRAL_SEPARACION, 1)} veces la de entrenamiento o más (la de "
          "entrenamiento es la media de cada época, con el dropout activo: con dropout, las curvas parecen algo más "
          f"juntas) · <i>infraajuste</i> si el acierto no llega al {pct(UMBRAL_INFRA, 0)} ni en entrenamiento ni en "
          f"validación · si no, <i>razonable</i>.")


def diagnosticar(modelo, h):
    """Diagnóstico en llano con reglas simples. Devuelve (etiqueta, explicación, cifras de este entrenamiento)."""
    p = probabilidades(modelo, ESTADO["datos"]["val"][0])
    parte_neumonia = float((p >= 0.5).mean())
    misma, mayoritaria = max(parte_neumonia, 1 - parte_neumonia), CLASES[int(parte_neumonia >= 0.5)]
    vl = h["val_loss"]
    n, ep_min = len(vl), int(np.argmin(vl)) + 1
    subida = vl[-1] / vl[ep_min - 1] - 1
    separacion = vl[-1] / h["loss"][-1]
    ta, va = h["accuracy"][-1], h["val_accuracy"][-1]
    trayectoria = (f"siguió bajando hasta la última época ({n})" if ep_min == n else
                   f"tocó su mínimo en la época {ep_min} de {n} y acabó un {pct(subida)} por encima")
    cifras = (f"En este entrenamiento: la pérdida de validación {trayectoria}; al final es "
              f"{num(separacion, 2)} veces la de entrenamiento; acierto "
              f"final del {pct(ta)} en entrenamiento y del {pct(va)} en validación; en validación dice "
              f"«{mayoritaria}» en el {pct(misma)} de los casos.")
    if misma >= UMBRAL_COLAPSO:
        return ("colapso", f"La red responde «{mayoritaria}» casi siempre: se ha quedado en la respuesta fácil y no "
                           "distingue las radiografías. Prueba otra configuración y vuelve a entrenar.", cifras)
    if (subida >= UMBRAL_SUBIDA and ep_min <= n - 3) or separacion >= UMBRAL_SEPARACION:
        return ("sobreajuste", "La red empieza a memorizar las imágenes de entrenamiento: su pérdida sigue "
                               "bajando, pero la de validación se estanca o sube y las dos curvas se separan. Más "
                               "dropout o menos épocas suelen ayudar.", cifras)
    if max(ta, va) < UMBRAL_INFRA:
        return ("infraajuste", "La red aún no ha aprendido lo suficiente: no llega al "
                               f"{pct(UMBRAL_INFRA, 0)} de acierto ni con las imágenes con las que entrena. "
                               "Más épocas o una red con más capacidad (más filtros o neuronas) suelen ayudar.",
                cifras)
    if ep_min == n:
        return ("razonable (aún mejoraba)", "Entrenamiento y validación van parejos y la pérdida de validación "
                                            "seguía bajando al final: con más épocas quizá mejore algo.", cifras)
    if subida >= UMBRAL_SUBIDA:
        return ("razonable", "Entrenamiento y validación van parejos, aunque la pérdida de validación subió en las "
                             "últimas épocas: puede ser ruido o el principio de un sobreajuste.", cifras)
    return ("razonable", "Entrenamiento y validación van parejos y la pérdida de validación apenas ha subido "
                         f"desde su mínimo (menos de un {pct(UMBRAL_SUBIDA, 0)}).", cifras)


def evaluar(modelo):
    """Métricas de test con umbral 0,5 (clase positiva = Neumonía). None si la red da valores no válidos."""
    xte, yte = ESTADO["datos"]["test"]
    p = probabilidades(modelo, xte)
    if not np.isfinite(p).all():
        return None
    y, pred = yte.astype(int), (p >= 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    vn, fp, fn, vp = (int(v) for v in cm.ravel())
    return dict(p=p, pred=pred, cm=cm, accuracy=(vp + vn) / len(y), sensibilidad=vp / (vp + fn),
                especificidad=vn / (vn + fp), auc=float(roc_auc_score(y, p)))


# --- Registro de experimentos ---
def _si(v):
    return v is True or v == "sí"


def registrar(entreno, metricas=None):
    """Añade (o actualiza, si ya existe ese id) la fila del experimento y guarda el CSV.
    Las cifras se guardan sin redondear: se redondean una sola vez, al mostrarlas o al escribir el CSV."""
    c = entreno["config"]
    fila = dict(id=entreno["id"], experimento=entreno["nombre"], reto=entreno["reto"],
                hipotesis=entreno["hipotesis"], resolucion=c["resolucion"], bloques=c["bloques"],
                filtros=c["filtros"], duplicar="sí" if c["duplicar"] else "no", kernel=c["kernel"],
                pooling="sí" if c["pooling"] else "no", batchnorm="sí" if c["batchnorm"] else "no",
                dropout=c["dropout"], densa=c["densa"], compensar="sí" if c["compensar"] else "no",
                epocas=c["epocas"], parametros=entreno["parametros"],
                tiempo_entrenamiento_s=float(entreno["segundos"]),
                acc_validacion=float(entreno["historial"]["val_accuracy"][-1]),
                accuracy=None, sensibilidad=None, especificidad=None, auc=None,
                diagnostico=entreno["diagnostico"])
    if metricas:
        fila.update({k: float(metricas[k]) for k in ("accuracy", "sensibilidad", "especificidad", "auc")})
    for i, f in enumerate(ESTADO["experimentos"]):
        if f["id"] == fila["id"]:
            ESTADO["experimentos"][i] = fila
            break
    else:
        ESTADO["experimentos"].append(fila)
    return guardar_csv()


def guardar_csv():
    """CSV con punto y coma y coma decimal: se abre bien en Excel en español."""
    flotantes = ("dropout", "tiempo_entrenamiento_s", "acc_validacion", "accuracy", "sensibilidad",
                 "especificidad", "auc")
    try:
        os.makedirs(CARPETA_SALIDAS, exist_ok=True)
        df = pd.DataFrame(ESTADO["experimentos"], columns=COLUMNAS).astype({c: "float64" for c in flotantes})
        df = df.round({c: 1 if c == "tiempo_entrenamiento_s" else 4 for c in flotantes})
        df.to_csv(RUTA_CSV, sep=";", decimal=",", encoding="utf-8-sig", index=False)
        return True
    except OSError as e:
        aviso(f"No he podido guardar <code>{RUTA_CSV}</code> ({html.escape(str(e))}). Si lo tienes abierto en "
              "Excel, ciérralo y vuelve a ejecutar la celda.")
        return False


def cambios(c):
    """Qué cambia respecto a la red por defecto, en palabras (vale para filas del registro y configuraciones)."""
    d, partes = DEFECTO, []
    if c["resolucion"] != 28:
        partes.append(f"{c['resolucion']} px")
    if c["bloques"] != d["bloques"]:
        partes.append(f"{c['bloques']} bloque" + ("s" if c["bloques"] > 1 else ""))
    if c["filtros"] != d["filtros"]:
        partes.append(f"{c['filtros']} filtros")
    if not _si(c["duplicar"]):
        partes.append("sin duplicar filtros")
    if c["kernel"] != d["kernel"]:
        partes.append(f"kernel {c['kernel']}×{c['kernel']}")
    if not _si(c["pooling"]):
        partes.append("sin pooling")
    if _si(c["batchnorm"]):
        partes.append("con normalización por lotes")
    if round(c["dropout"], 1) != d["dropout"]:
        partes.append(f"dropout {c['dropout']:g}".replace(".", ","))
    if c["densa"] != d["densa"]:
        partes.append("sin capa densa" if c["densa"] == 0 else f"densa de {c['densa']}")
    if _si(c["compensar"]):
        partes.append("compensa el desbalanceo")
    if c["epocas"] != d["epocas"]:
        partes.append(f"{c['epocas']} épocas")
    return " · ".join(partes) or "ninguno (red por defecto)"


def _etiquetar(ax, xs, ys, ids):
    """Rótulos «#id» a la derecha de cada punto, sin pisarse. Llamar después de fijar ejes y de tight_layout.
    Los puntos superpuestos comparten rótulo («#5, #7»); si dos rótulos chocan, se separan en vertical y una
    línea fina une cada uno con su punto."""
    y0, y1 = ax.get_ylim()
    alto = ax.bbox.height * 72 / ax.figure.dpi                 # alto del panel en puntos tipográficos
    def a_pt(y):
        return (y - y0) / (y1 - y0) * alto
    orden = sorted(range(len(xs)), key=lambda i: xs[i])
    grupos = [[orden[0]]]
    for i in orden[1:]:                                         # vecinos en x: a menos de 0,3 décadas
        if np.log10(xs[i] / xs[grupos[-1][-1]]) < 0.3:
            grupos[-1].append(i)
        else:
            grupos.append([i])
    for grupo in grupos:
        rotulos = []                                            # de arriba abajo
        for i in sorted(grupo, key=lambda i: -ys[i]):
            r = rotulos[-1] if rotulos else None
            if r and abs(np.log10(xs[i] / r["x"])) < 0.01 and a_pt(r["y"]) - a_pt(ys[i]) < 2:
                r["ids"].append(ids[i])                         # marcadores superpuestos: un solo rótulo
            else:
                rotulos.append({"x": xs[i], "y": ys[i], "ids": [ids[i]]})
        pos = [a_pt(r["y"]) for r in rotulos]
        for j in range(1, len(pos)):                            # al menos 16 pt entre rótulos
            pos[j] = min(pos[j], pos[j - 1] - 16)
        if pos[-1] < 8:                                         # sin salirse por abajo
            pos[-1] = 8
            for j in range(len(pos) - 2, -1, -1):
                pos[j] = max(pos[j], pos[j + 1] + 16)
        for r, p in zip(rotulos, pos):
            dy = p - a_pt(r["y"])
            ax.annotate(", ".join(f"#{n}" for n in sorted(r["ids"])), (r["x"], r["y"]), xytext=(12, dy),
                        textcoords="offset points", fontsize=12, va="center",
                        arrowprops=dict(arrowstyle="-", color="0.3", lw=1.1, shrinkA=1, shrinkB=5)
                        if abs(dy) > 4 else None)


def mostrar_experimentos(filas):
    """Tabla del registro y gráfico de parámetros frente a sensibilidad y especificidad."""
    def cifra(v, f):
        return "—" if v is None else f(v)
    tabla_html(pd.DataFrame([{
        "#": f["id"], "Experimento": f["experimento"], "Reto": f["reto"] or "—",
        "Cambios respecto a la red por defecto": cambios(f), "Parámetros": num(f["parametros"]),
        "Tiempo (s)": num(f["tiempo_entrenamiento_s"], 1), "Acierto (validación)": pct(f["acc_validacion"]),
        "Acierto (test)": cifra(f["accuracy"], pct), "Sensibilidad": cifra(f["sensibilidad"], pct),
        "Especificidad": cifra(f["especificidad"], pct), "AUC": cifra(f["auc"], lambda v: num(v, 3)),
        "Diagnóstico": f["diagnostico"]} for f in filas]))
    con_test = [f for f in filas if f["accuracy"] is not None]
    if not con_test:
        aviso("Para el gráfico hace falta evaluar en test: ejecuta la celda 1.5 y vuelve aquí.")
        return
    xs = [f["parametros"] for f in con_test]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, clave, titulo in ((axs[0], "sensibilidad", "Sensibilidad (neumonías detectadas)"),
                              (axs[1], "especificidad", "Especificidad (normales bien descartadas)")):
        ys = [f[clave] for f in con_test]
        ax.scatter(xs, ys, s=80, color=AZUL, edgecolor="black", zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(min(xs) / 2, max(xs) * 6)
        ax.set_ylim(min(ys) - 0.08, 1.015)
        ax.set_title(titulo)
        ax.set_xlabel("Parámetros de la red\n(cada marca vale 10 veces más que la anterior)")
        ax.xaxis.set_major_formatter(FORMATO_PARAM)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FORMATO_PCT)
    axs[0].axhline(1.0, color=BERMELLON, ls="--", lw=2, label="Siempre «Neumonía»: 100 %")
    axs[0].legend(loc="lower left")
    axs[1].text(0.02, 0.03, "Siempre «Neumonía»: 0 % (muy por debajo, fuera del gráfico)", transform=axs[1].transAxes,
                color=BERMELLON, fontsize=12)
    fig.tight_layout()
    for ax, clave in ((axs[0], "sensibilidad"), (axs[1], "especificidad")):
        _etiquetar(ax, xs, [f[clave] for f in con_test], [f["id"] for f in con_test])
    mostrar(fig)
    nota("Cada punto es un experimento; el número es su # en la tabla de arriba (si varios puntos caen casi en el "
         "mismo sitio, comparten rótulo). Más a la derecha, más parámetros; más arriba, mejor en esa métrica. "
         "Responder siempre «Neumonía» da un 100 % de sensibilidad y un 0 % de especificidad: hacen falta las dos.")
    nota("Ojo: cada fila es un solo entrenamiento con la semilla 42. Con otra semilla las cifras cambian (en nuestras "
         "pruebas, la especificidad de la red por defecto se movió más de 15 puntos solo por eso), así que una "
         "diferencia de pocos puntos puede ser azar: no saques conclusiones de ella. Fíjate solo en las diferencias "
         "grandes.")


# --- Arranque ---
if _listo:
    tf.get_logger().setLevel(logging.ERROR)
    try:
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)
    except ImportError:
        pass
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "legend.fontsize": 12, "figure.dpi": 100, "savefig.dpi": 100,
                         "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False})
    keras.utils.set_random_seed(SEMILLA)
    ya_habia = "ESTADO" in globals()
    if not ya_habia:   # re-ejecutar esta celda no borra el registro de experimentos
        ESTADO = {"datos": None, "resolucion": None, "config": None, "config_invalida": False,
                  "red_actual": None, "red_anterior": None, "modelo": None, "entreno": None,
                  "contador": 0, "experimentos": [], "reto": "", "hipotesis": "", "aleatoria": 0}

    nota('<span style="font-size:14px">Versiones instaladas (por si algo falla; no hace falta que las mires).</span>')
    tabla_html(pd.DataFrame([{"Python": platform.python_version(), "TensorFlow": tf.__version__,
                              "Keras": keras.__version__, "NumPy": np.__version__, "pandas": pd.__version__,
                              "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                              "medmnist": medmnist.__version__}]))
    gpu = tf.config.list_physical_devices("GPU")
    ok(f"Todo listo. Tarjeta gráfica (GPU): {'sí, se usará' if gpu else 'no, y no hace falta'} · Semilla: {SEMILLA}.")
    if ya_habia and ESTADO["experimentos"]:
        nota(f"Se conserva el registro de esta sesión ({len(ESTADO['experimentos'])} experimentos).")
''')

# ---------------------------------------------------------------------------------------------
# 1.2 Los datos
# ---------------------------------------------------------------------------------------------
md(r'''
## 1.2 Los datos
PneumoniaMNIST: 5.856 radiografías de tórax de niños, etiquetadas como «Normal» o «Neumonía».
Elige la resolución: a 28×28 px la red entrena en segundos; a 64×64 px se ve más detalle, pero tarda
varias veces más.
''')

codigo("1.2 Los datos", r'''
resolucion = 28  # @param [28, 64] {type:"raw"}

import hashlib, shutil, urllib.request

RAIZ_DATOS = os.path.join("datos", "medmnist")
MD5 = {28: "28209eda62fecd6e6a2d98b1501bb15f", 64: "8f4eceb4ccffa70c672198ea285246c6"}
URLS = {28: ["https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1",
             # Espejo NO oficial (copia idéntica, se comprueba con MD5), solo para 28 px:
             "https://huggingface.co/datasets/albertvillanova/medmnist-v2/resolve/"
             "f6dd981c7400b3e3738bde12ff948b7cc8f0d623/data/pneumoniamnist.npz"],
        64: ["https://zenodo.org/records/10519652/files/pneumoniamnist_64.npz?download=1"]}


def _md5(ruta):
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _asegurar_npz(res):
    """Descarga el archivo si falta o está dañado. Devuelve (ruta, None) o (None, mensaje de error)."""
    os.makedirs(RAIZ_DATOS, exist_ok=True)
    ruta = os.path.join(RAIZ_DATOS, "pneumoniamnist.npz" if res == 28 else f"pneumoniamnist_{res}.npz")
    if os.path.exists(ruta) and _md5(ruta) == MD5[res]:
        return ruta, None
    fallos = []
    for url in URLS[res]:
        parcial = ruta + ".parcial"
        try:
            # timeout: un servidor que no responde no deja la celda colgada. Zenodo rechaza «Mozilla/5.0».
            peticion = urllib.request.Request(url, headers={"User-Agent": "laboratorio-cnn-unir/1.0"})
            with urllib.request.urlopen(peticion, timeout=60) as r, open(parcial, "wb") as f:
                shutil.copyfileobj(r, f)
            if _md5(parcial) != MD5[res]:
                raise ValueError("el archivo llegó incompleto (MD5 distinto)")
            os.replace(parcial, ruta)
            return ruta, None
        except Exception as e:
            fallos.append(f"{url.split('/')[2]}: {type(e).__name__}: {e}")
            if os.path.exists(parcial):
                os.remove(parcial)
    consejo = ("Vuelve a la resolución 28, que tiene un servidor de reserva, o espera un minuto y reintenta."
               if res != 28 else "Espera un minuto y vuelve a ejecutar la celda.")
    return None, (f"No he podido descargar PneumoniaMNIST a {res}×{res} px. {consejo}<br>Si sigue fallando, "
                  f"descarga «{os.path.basename(ruta)}» desde "
                  '<a href="https://doi.org/10.5281/zenodo.10519652" target="_blank">Zenodo</a> y súbelo a la '
                  f"carpeta <code>{RAIZ_DATOS}</code> desde el panel de archivos (a la izquierda)."
                  f"<br><small>Detalle técnico: {html.escape(' | '.join(fallos))}</small>")


def _cargar_datos(res):
    try:
        res = int(res)
    except (TypeError, ValueError):
        res = None
    if res not in (28, 64):
        error("La resolución tiene que ser 28 o 64. Elígela en el desplegable y vuelve a ejecutar la celda.")
        return
    if ESTADO["datos"] is None or ESTADO["resolucion"] != res:
        estado = display(HTML("Comprobando los datos (si hay que descargarlos, tarda unos segundos)…"),
                         display_id=True)
        ruta, fallo = _asegurar_npz(res)
        if fallo:
            estado.update(HTML(""))
            error(fallo)
            return
        try:
            datos = {}
            for parte in ("train", "val", "test"):   # medmnist solo lee el archivo: nunca descarga
                ds = medmnist.PneumoniaMNIST(split=parte, size=res, root=RAIZ_DATOS, download=False)
                datos[parte] = (ds.imgs.astype("float32")[..., None] / 255.0, ds.labels.ravel().astype("float32"))
        except Exception as e:
            estado.update(HTML(""))
            error(f"No he podido leer <code>{ruta}</code>. Bórralo desde el panel de archivos y vuelve a ejecutar "
                  f"la celda para descargarlo de nuevo.<br><small>Detalle técnico: {html.escape(str(e))}</small>")
            return
        estado.update(HTML(""))
        ESTADO["datos"], ESTADO["resolucion"] = datos, res
    (xtr, ytr), (xva, yva), (xte, yte) = (ESTADO["datos"][p] for p in ("train", "val", "test"))
    ok(f"Datos listos a {res}×{res} px: {num(len(ytr))} radiografías de entrenamiento, {num(len(yva))} de "
       f"validación y {num(len(yte))} de test.")
    nota("<b>Entrenamiento</b>: las imágenes con las que aprende la red. <b>Validación</b>: imágenes que la red no "
         "usa para aprender; sirven para comprobar, mientras entrena, si aprende de verdad o solo memoriza. "
         "<b>Test</b>: el examen final, con imágenes que la red no ve mientras aprende.")
    if ESTADO["config"] and ESTADO["config"]["resolucion"] != res:
        aviso(f"Has cambiado la resolución: vuelve a ejecutar 1.3 y 1.4 para diseñar y entrenar la red con "
              f"imágenes de {res}×{res} px.")

    # Rejilla 4×4: 8 normales y 8 con neumonía, elegidas con la semilla
    rng = np.random.default_rng(SEMILLA)
    indices = np.concatenate([rng.choice(np.flatnonzero(ytr == c), 8, replace=False) for c in (0, 1)])
    fig, axs = plt.subplots(4, 4, figsize=(8.5, 9.4))
    for ax, i in zip(axs.ravel(), indices):
        ax.imshow(xtr[i, ..., 0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(CLASES[int(ytr[i])], fontsize=14)
        ax.axis("off")
    fig.suptitle(f"Ejemplos de entrenamiento ({res}×{res} px): arriba normales, abajo neumonías", fontsize=15)
    fig.tight_layout()
    mostrar(fig)

    # Reparto de clases
    partes = [("Entrenamiento", ytr), ("Validación", yva), ("Test", yte)]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    posiciones, ancho = np.arange(3), 0.38
    for k, (clase, color, trama) in enumerate(((0, AZUL, ""), (1, NARANJA, "//"))):
        cuentas = [int((y == clase).sum()) for _, y in partes]
        barras = ax.bar(posiciones + (k - 0.5) * ancho, cuentas, ancho, color=color, hatch=trama,
                        edgecolor="black", label=CLASES[clase])
        for barra, (_, y), n in zip(barras, partes, cuentas):
            ax.annotate(pct(n / len(y)), (barra.get_x() + barra.get_width() / 2, n), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=12)
    ax.set_xticks(posiciones, [nombre for nombre, _ in partes])
    ax.set_ylabel("Radiografías")
    ax.set_ylim(0, len(ytr) * 0.85)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: num(v)))
    ax.set_title("Normales y neumonías en cada conjunto")
    ax.legend()
    fig.tight_layout()
    mostrar(fig)
    p_tr, p_te = float(ytr.mean()), float(yte.mean())
    nota(f"En entrenamiento hay unas {num(p_tr / (1 - p_tr), 1)} neumonías por cada radiografía normal "
         f"({pct(p_tr)} de neumonías): los datos están <b>desbalanceados</b>. En test hay menos neumonías "
         f"({pct(p_te)}).")
    info(f"<b>Referencia tonta:</b> responder siempre «Neumonía» acierta el <b>{pct(p_te)}</b> en test: detecta "
         "todas las neumonías, pero no reconoce ninguna radiografía normal. Es el listón mínimo que tu red debe "
         "superar.")


if "ESTADO" not in globals():
    __FALTA__
else:
    _cargar_datos(resolucion)
''')

# ---------------------------------------------------------------------------------------------
# 1.3 Diseña tu red
# ---------------------------------------------------------------------------------------------
md(r'''
## 1.3 Diseña tu red
Mueve los controles y ejecuta la celda: verás la ficha de tu red y cuánto tardará en entrenar.
Como en CNN Explainer, las convoluciones no rellenan los bordes: una convolución 3×3 quita 2 píxeles
de lado (28 → 26).

| Control | Qué cambia |
|---|---|
| `bloques_convolucionales` | Cuántos bloques se apilan. Cada bloque: convolución + ReLU (deja a cero los valores negativos) + pooling si está activado. |
| `filtros_primer_bloque` | Filtros del primer bloque. Un filtro es un pequeño detector de patrones que recorre la imagen. |
| `duplicar_filtros_en_cada_bloque` | Si cada bloque tiene el doble de filtros que el anterior. |
| `tamano_kernel` | Lado de cada filtro (el «kernel» de CNN Explainer): 3 (3×3 píxeles) o 5 (5×5). |
| `usar_pooling` | Pooling máx. 2×2: se queda con el valor más alto de cada cuadro de 2×2 y deja la imagen a la mitad de lado. |
| `usar_batchnorm` | Normalización por lotes: dentro de cada bloque, vuelve a poner los números en una escala parecida antes de pasar a la ReLU. Suele hacer el entrenamiento más estable. |
| `dropout` | Parte de las neuronas que se apagan al azar mientras la red entrena, para que memorice menos (0,2 = el 20 %; en el deslizador sale 0.2). |
| `neuronas_capa_densa` | Capa densa: neuronas que miran a la vez todo lo que sale de los bloques y lo combinan para decidir. Aquí eliges cuántas (0 = ninguna). |
| `compensar_desbalanceo` | Al entrenar, equivocarse con una radiografía normal cuenta unas 3 veces más que equivocarse con una neumonía, porque hay muchas menos normales. |
| `epocas` | Época = una pasada completa por las 4.708 imágenes de entrenamiento. |
| `nombre_experimento` | Tus iniciales y tu reto (p. ej. «ALM_A»): así se ve en la hoja común. |

**Red por defecto:** __RED_DEFECTO__

Un **parámetro** es un número que la red ajusta al entrenar (los pesos de filtros y neuronas).
''')

codigo("1.3 Diseña tu red", r'''
bloques_convolucionales = 2  # @param {type:"slider", min:1, max:4, step:1}
filtros_primer_bloque = 16  # @param [8, 16, 32] {type:"raw"}
duplicar_filtros_en_cada_bloque = True  # @param {type:"boolean"}
tamano_kernel = 3  # @param [3, 5] {type:"raw"}
usar_pooling = True  # @param {type:"boolean"}
usar_batchnorm = False  # @param {type:"boolean"}
dropout = 0.2  # @param {type:"slider", min:0, max:0.6, step:0.1}
neuronas_capa_densa = 64  # @param [0, 32, 64, 128] {type:"raw"}
compensar_desbalanceo = False  # @param {type:"boolean"}
epocas = 8  # @param {type:"slider", min:3, max:20, step:1}
nombre_experimento = "INICIALES_RETO"  # @param {type:"string"}


def _tipo_capa(capa):
    t = capa.__class__.__name__
    if t == "InputLayer":
        return "Entrada (imagen en gris)"
    if t == "Conv2D":
        return f"Convolución {capa.kernel_size[0]}×{capa.kernel_size[1]} · {capa.filters} filtros"
    if t == "BatchNormalization":
        return "Normalización por lotes"
    if t == "MaxPooling2D":
        return "Pooling máx. 2×2"
    if t == "Flatten":
        return "Aplanar"
    if t == "Dropout":
        return f"Dropout {pct(capa.rate, 0)}"
    if t == "Dense":
        return "Densa · 1 neurona (puntuación)" if capa.units == 1 else f"Densa · {capa.units} neuronas"
    return "Salida sigmoide (probabilidad de neumonía)" if capa.name == "probabilidad" else "ReLU"


def _forma(capa):
    dims = tuple(capa.output.shape[1:])
    return " × ".join(str(d) for d in dims) if len(dims) > 1 else num(dims[0])


def _diagrama(modelo, res):
    """Diagrama 2D: una barra por capa, de arriba (entrada) abajo (salida); ancho = lado de la imagen."""
    colores = {"Entrada": "#999999", "Convolución + ReLU": AZUL, "Pooling máx. 2×2": NARANJA,
               "Aplanar": VERDE, "Densa": ROSA, "Dropout": CELESTE, "Salida": BERMELLON}
    con_bn = any(c.__class__.__name__ == "BatchNormalization" for c in modelo.layers)
    filas = []
    for capa in modelo.layers:
        t, d = capa.__class__.__name__, tuple(capa.output.shape[1:])
        if t == "InputLayer":
            filas.append(("Entrada", f"Entrada · {d[0]}×{d[1]} px, 1 canal (gris)", d))
        elif t == "Conv2D":
            k = capa.kernel_size[0]
            filas.append(("Convolución + ReLU", f"Convolución {k}×{k}{' + normalización' if con_bn else ''} + ReLU"
                                                f" · {capa.filters} filtros · {d[0]}×{d[1]}", d))
        elif t == "MaxPooling2D":
            filas.append(("Pooling máx. 2×2", f"Pooling máx. 2×2 · {d[0]}×{d[1]}", d))
        elif t == "Flatten":
            filas.append(("Aplanar", f"Aplanar · {num(d[0])} valores en fila", d))
        elif t == "Dense" and capa.name == "densa":
            filas.append(("Densa", f"Densa · {capa.units} neuronas", d))
        elif t == "Dropout":
            filas.append(("Dropout", f"Dropout {pct(capa.rate, 0)} (solo al entrenar)", d))
        elif capa.name == "logit":
            filas.append(("Salida", "Salida · 1 neurona + sigmoide → probabilidad de neumonía", d))
    n = len(filas)
    fig, ax = plt.subplots(figsize=(12, 0.55 * n + 1.2))
    for j, (tipo, texto, d) in enumerate(filas):
        y = n - 1 - j
        ancho = 0.9 * d[0] / res if len(d) == 3 else 0.05
        ax.add_patch(Rectangle((0.5 - ancho / 2, y - 0.3), ancho, 0.6, facecolor=colores[tipo], edgecolor="black"))
        ax.text(1.05, y, texto, va="center", fontsize=13)
    ax.set_xlim(0, 3.3)
    ax.set_ylim(-0.6, n - 0.4)
    ax.axis("off")
    tipos = list(dict.fromkeys(t for t, _, _ in filas))
    ax.legend(handles=[Patch(facecolor=colores[t], edgecolor="black", label=t) for t in tipos],
              loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False)
    ax.set_title("Tu red, de la entrada (arriba) a la salida (abajo) · ancho de la barra = lado de la imagen",
                 loc="left", fontsize=14)
    return fig


def _segundos_cota(c):
    """Tiempo aproximado sin construir la red: multiplicaciones por imagen × imágenes × épocas.
    Por lo alto (CPU de Colab), para no proponer una solución que luego se bloquee por tiempo."""
    lado, canales, filtros, macs = c["resolucion"], 1, c["filtros"], 0
    for _ in range(c["bloques"]):
        lado -= c["kernel"] - 1
        macs += lado * lado * c["kernel"] ** 2 * canales * filtros
        if c["pooling"]:
            lado //= 2
        canales, filtros = filtros, filtros * (2 if c["duplicar"] else 1)
    macs += lado * lado * canales * max(c["densa"], 1) + c["densa"]
    imagenes = len(ESTADO["datos"]["train"][1]) + len(ESTADO["datos"]["val"][1]) / 3
    return 1.2 * c["epocas"] * imagenes * macs * SEGUNDOS_POR_GMAC / 1e9


def _viable(c):
    # Límite conocido: la cota está pensada para Colab; en un ordenador rápido puede callar alguna solución válida
    return tamanos(c["resolucion"], c["bloques"], c["kernel"], c["pooling"])[1] is None and \
        _segundos_cota(c) <= MAX_SEGUNDOS and construir_modelo(c).count_params() <= MAX_PARAMETROS


def _soluciones(c):
    """Solo las soluciones que funcionan de verdad con el resto de la configuración."""
    sol = []
    caben = [b for b in range(1, c["bloques"]) if _viable({**c, "bloques": b})]
    if caben:
        sol.append(f"usa {max(caben)} bloque{'s' if max(caben) > 1 else ''} como máximo")
    if c["kernel"] == 5 and _viable({**c, "kernel": 3}):
        sol.append("usa kernel 3")
    if c["resolucion"] == 28 and _viable({**c, "resolucion": 64}):
        sol.append("sube la resolución a 64 en la celda 1.2")
    if c["pooling"] and _viable({**c, "pooling": False}):
        sol.append("desactiva el pooling (la red crecerá mucho y tardará más)")
    if not sol:
        return "quita bloques."
    return (", ".join(sol[:-1]) + " o " + sol[-1] if len(sol) > 1 else sol[0]) + "."


def _disenar(cfg):
    ESTADO["config_invalida"] = True   # hasta que la red pase todas las comprobaciones
    cfg["resolucion"] = ESTADO["resolucion"]
    if cfg["bloques"] < 1 or cfg["epocas"] < 1 or cfg["filtros"] < 1 or cfg["densa"] < 0 or \
            cfg["kernel"] < 1 or not 0 <= cfg["dropout"] < 1:
        error("Algún valor del formulario no es válido: bloques, filtros, kernel y épocas tienen que ser 1 o más, "
              "y el dropout, un número entre 0 y 0,9.")
        return
    res, b, k = cfg["resolucion"], cfg["bloques"], cfg["kernel"]
    trayectoria, falla = tamanos(res, b, k, cfg["pooling"])
    if falla:
        recorrido = " → ".join(str(s) for s in [res] + trayectoria) + " → 0"
        error(f"Con {b} bloques, kernel {k}×{k} {'y pooling' if cfg['pooling'] else 'sin pooling'}, la imagen de "
              f"{res}×{res} se queda en 0×0 píxeles en el bloque {falla} (lado tras cada bloque: {recorrido}), así "
              f"que la red no se puede construir. Soluciones: {_soluciones(cfg)}")
        return

    modelo = construir_modelo(cfg)
    parametros = modelo.count_params()
    megas = parametros * 4 / 1024 ** 2
    nota(f"<h3 style='margin:4px 0'>Ficha del modelo · «{html.escape(cfg['nombre'])}»</h3>")
    mostrar(_diagrama(modelo, res))
    tabla_html(pd.DataFrame([{"Capa": c.name, "Tipo": _tipo_capa(c), "Forma de salida": _forma(c),
                              "Parámetros": num(c.count_params())} for c in modelo.layers]))
    nota('<span style="font-size:14px"><b>Forma de salida</b> = alto × ancho × número de mapas (uno por filtro). '
         "«Aplanar» pone todos esos números en una sola fila. La última neurona da una puntuación (logit) y la "
         "sigmoide la convierte en una probabilidad de neumonía entre 0 y 1.</span>")

    arquitectura, completo = diseno(cfg)   # re-ejecutar lo mismo no cambia la comparación
    if ESTADO["red_actual"] is None or ESTADO["red_actual"][0] != completo:
        ESTADO["red_anterior"], ESTADO["red_actual"] = ESTADO["red_actual"], (completo, arquitectura, parametros)
    anterior = ESTADO["red_anterior"]
    if anterior is None:
        comparacion = "Primera red de la sesión."
    elif anterior[1] == arquitectura:
        comparacion = "Misma arquitectura que la red anterior (cambian las épocas o la compensación del desbalanceo)."
    elif parametros == anterior[2]:
        comparacion = "Mismo número de parámetros que la red anterior."
    else:
        dif, veces = parametros - anterior[2], parametros / anterior[2]
        factor = f" ({num(veces, 1)} veces más)" if veces >= 1.1 else f" ({num(1 / veces, 1)} veces menos)" \
            if veces <= 1 / 1.1 else ""
        comparacion = f"<b>{'+' if dif > 0 else '−'}{num(abs(dif))} parámetros</b> respecto a la red anterior{factor}."
    nota(f"<b>Total: {num(parametros)} parámetros</b> ({num(megas, 2)} MB: cada parámetro ocupa 4 bytes). "
         f"{comparacion}")

    if parametros > MAX_PARAMETROS:
        error(f"Esta red tiene {num(parametros / 1e6, 1)} millones de parámetros ({num(megas)} MB): en la CPU de "
              "Colab tardaría del orden de una hora en entrenar. Activa el pooling, baja la resolución a 28 o usa "
              "menos filtros o neuronas.")
        return
    estado = display(HTML(f"⏱️ Midiendo cuánto tarda un lote de imágenes {DONDE}…"), display_id=True)
    segundos = segundos_estimados(medir_lotes(modelo), cfg["epocas"])
    minutos = f" (unos {num(segundos / 60)} minutos)" if segundos >= 120 else ""
    estado.update(HTML(f'<div style="font-size:15px">⏱️ Tiempo estimado de entrenamiento {DONDE}: '
                       f'<b>~{num(segundos)} s</b>{minutos}, sin contar el dibujo de las curvas</div>'))
    if segundos > MAX_SEGUNDOS:
        arreglos = (["activa el pooling"] if not cfg["pooling"] else []) + \
                   (["baja la resolución a 28 en la celda 1.2"] if res != 28 else []) + \
                   [f"baja las épocas (ahora {cfg['epocas']})", "usa menos filtros o neuronas"]
        error(f"Unos {num(segundos / 60)} minutos de entrenamiento no caben en el laboratorio. Soluciones: "
              f"{', '.join(arreglos[:-1])} o {arreglos[-1]}.")
        return
    if segundos > AVISO_SEGUNDOS:
        aviso(f"Esta red tardará unos {duracion(segundos)} en entrenar. Puedes seguir, pero quizá prefieras una más "
              "pequeña.")
    lado = modelo.get_layer(f"relu_{b}").output.shape[1]
    if lado <= 3:
        aviso(f"El último mapa de activación mide {lado}×{lado} px: el mapa de Grad-CAM (1.7) tendrá solo "
              f"{lado}×{lado} cuadros y apenas señalará zonas.")
    ESTADO["config"], ESTADO["config_invalida"] = cfg, False
    ok("Red lista. Ahora ejecuta <b>1.4 · Entrena la red</b>.")


_nombre = " ".join(str(nombre_experimento).replace(";", ",").split()) or "sin_nombre"
_cfg = dict(bloques=int(bloques_convolucionales), filtros=int(filtros_primer_bloque),
            duplicar=bool(duplicar_filtros_en_cada_bloque), kernel=int(tamano_kernel), pooling=bool(usar_pooling),
            batchnorm=bool(usar_batchnorm), dropout=round(float(dropout), 1), densa=int(neuronas_capa_densa),
            compensar=bool(compensar_desbalanceo), epocas=int(epocas), nombre=_nombre)
if "ESTADO" not in globals():
    __FALTA__
elif requiere("datos"):
    _disenar(_cfg)
''')

# ---------------------------------------------------------------------------------------------
# 1.4 Entrena
# ---------------------------------------------------------------------------------------------
md(r'''
## 1.4 Entrena la red
La red ve las imágenes de entrenamiento en lotes de 128 y ajusta sus parámetros para reducir la
**pérdida** (el error medio de sus probabilidades: cuanto más baja, mejor). Al final de cada época se
mide también en **validación**, con imágenes que no usa para aprender. Cada vez que ejecutas esta celda
la red empieza desde cero, con la misma semilla. Al final verás un diagnóstico: **sobreajuste** = memoriza
las imágenes de entrenamiento y falla más con radiografías nuevas; **infraajuste** = aún no ha aprendido lo
suficiente; **colapso** = responde siempre lo mismo.

Si has cambiado algo en 1.3, pulsa antes su ▶: si no, se entrena la red de la vez anterior.
''')

codigo("1.4 Entrena la red", r'''
def _entrenar_red():
    cfg = ESTADO["config"]
    if cfg["resolucion"] != ESTADO["resolucion"]:
        error(f"Diseñaste la red para imágenes de {cfg['resolucion']}×{cfg['resolucion']} px, pero en 1.2 has "
              f"cargado las de {ESTADO['resolucion']}×{ESTADO['resolucion']} px. Vuelve a ejecutar 1.3.")
        return
    nota(f"Entrenando «{html.escape(cfg['nombre'])}»: {num(ESTADO['red_actual'][2])} parámetros, según la última vez "
         "que ejecutaste 1.3.")
    total = cfg["epocas"]

    class CurvasEnVivo(keras.callbacks.Callback):
        """Redibuja las curvas al final de cada época en la misma figura."""
        def on_train_begin(self, logs=None):
            self.h = {k: [] for k in ("loss", "val_loss", "accuracy", "val_accuracy")}
            self.tiempos = []
            self.fig, self.axs = plt.subplots(1, 2, figsize=(12.5, 4.5))
            self.dibujar()
            self.grafico = display(png(self.fig), display_id=True)
            self.linea = display(HTML("Entrenando la primera época…"), display_id=True)

        def on_epoch_begin(self, epoch, logs=None):
            self.inicio = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            self.tiempos.append(time.perf_counter() - self.inicio)
            for k in self.h:
                self.h[k].append(float(logs[k]))
            self.dibujar()
            self.grafico.update(png(self.fig))
            hechas = len(self.tiempos)
            quedan = (total - hechas) * float(np.mean(self.tiempos[-3:]))
            self.linea.update(HTML(
                f'<div style="font-size:15px"><b>Época {hechas}/{total}</b> · {num(self.tiempos[-1], 1)} s esta '
                f'época' + (f" · quedan ≈ {num(quedan)} s" if hechas < total else "") + "</div>"))

        def dibujar(self):
            epocas_hechas = range(1, len(self.h["loss"]) + 1)
            for ax, clave, titulo in ((self.axs[0], "loss", "Pérdida (más baja = mejor)"),
                                      (self.axs[1], "accuracy", "Acierto (accuracy)")):
                ax.clear()
                ax.plot(epocas_hechas, self.h[clave], "o-", color=AZUL, lw=2, label="entrenamiento")
                ax.plot(epocas_hechas, self.h["val_" + clave], "s--", color=NARANJA, lw=2, label="validación")
                ax.set_xlim(0.5, total + 0.5)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_xlabel("Época")
                ax.set_title(titulo)
                ax.legend(loc="best")
            self.axs[0].yaxis.set_major_formatter(FORMATO_DEC)
            self.axs[1].yaxis.set_major_formatter(FORMATO_PCT)
            self.fig.tight_layout()

    curvas = CurvasEnVivo()
    try:
        modelo, historial, _ = entrenar(cfg, [curvas])
    except KeyboardInterrupt:
        plt.close("all")
        aviso("Has detenido el entrenamiento. Vuelve a ejecutar la celda cuando quieras.")
        return
    except Exception as e:   # p. ej., memoria insuficiente con una red enorme
        plt.close("all")
        error(f"El entrenamiento ha fallado ({type(e).__name__}). Prueba una red más pequeña (con pooling, menos "
              "filtros o a 28 px) y vuelve a ejecutar 1.3 y 1.4."
              f"<br><small>Detalle técnico: {html.escape(str(e)[:300])}</small>")
        return
    plt.close("all")
    segundos = sum(curvas.tiempos)   # sin el dibujo de las curvas: comparable con el Plan B, que no las dibuja

    etiqueta, explicacion, cifras = diagnosticar(modelo, historial)
    ESTADO["contador"] += 1
    entreno = dict(id=ESTADO["contador"], nombre=cfg["nombre"], reto=ESTADO["reto"], hipotesis=ESTADO["hipotesis"],
                   config=dict(cfg), parametros=modelo.count_params(), segundos=segundos, historial=historial,
                   diagnostico=etiqueta)
    ESTADO["modelo"], ESTADO["entreno"] = modelo, entreno
    registrar(entreno)
    ok(f"Entrenamiento terminado en <b>{num(segundos, 1)} s</b> ({num(segundos / total, 1)} s por época de media). "
       f"Queda anotado como experimento n.º {entreno['id']} («{html.escape(cfg['nombre'])}»).")
    recuadro = {"colapso": error, "sobreajuste": aviso, "infraajuste": aviso}.get(etiqueta, info)
    recuadro(f"<b>Diagnóstico: {etiqueta}.</b> {explicacion}")
    nota(f'<span style="font-size:14px">{REGLAS}</span>')
    nota(f'<details style="font-size:14px"><summary>Cifras de este entrenamiento</summary>{cifras}</details>')
    nota("Fíjate: el acierto de validación puede ir por encima del de entrenamiento (con la red por defecto suele "
         "pasar en todas las épocas). Es normal: el dropout solo actúa al entrenar y la cifra de entrenamiento es la "
         "media de toda la época" + ("; además, al compensar, en la de entrenamiento las normales pesan más."
                                    if cfg["compensar"] else "."))
    nota("Siguiente paso: <b>1.5 · Evalúa en test</b>.")


if "ESTADO" not in globals():
    __FALTA__
elif requiere("datos", "config"):
    _entrenar_red()
''')

# ---------------------------------------------------------------------------------------------
# 1.5 Evalúa en test
# ---------------------------------------------------------------------------------------------
md(r'''
## 1.5 Evalúa en test
El test son 624 radiografías que la red no ha visto nunca. Umbral 0,5: si la probabilidad de neumonía
es 0,5 o más, la red dice «Neumonía».

*Nota de rigor: en un proyecto real la configuración se elige mirando solo validación y el test se mira una
vez, al final. Aquí lo miramos en cada prueba para aprender.*
''')

codigo("1.5 Evalúa en test", r'''
def _evaluar_test():
    entreno = ESTADO["entreno"]
    m = evaluar(ESTADO["modelo"])
    if m is None:
        error("La red ha dado resultados no válidos (NaN): el entrenamiento se ha roto. Cambia la configuración en "
              "1.3 y vuelve a entrenar.")
        return
    yva, yte = ESTADO["datos"]["val"][1], ESTADO["datos"]["test"][1]
    referencia = float(yte.mean())
    tarjetas = [("Acierto (test)", pct(m["accuracy"]), "aciertos sobre el total"),
                ("Sensibilidad", pct(m["sensibilidad"]), "de las neumonías, cuántas detecta"),
                ("Especificidad", pct(m["especificidad"]), "de las normales, cuántas deja tranquilas"),
                ("AUC", num(m["auc"], 3), "elegidas al azar una neumonía y una normal, probabilidad de que la red "
                                           "puntúe más alto la neumonía (0,5 = azar, 1 = perfecto)")]
    display(HTML('<div style="display:flex; flex-wrap:wrap; gap:10px; margin:8px 0">' + "".join(
        f'<div style="border:2px solid #0072B2; border-radius:8px; padding:8px 14px; min-width:170px; color:inherit">'
        f'<div style="font-size:14px">{t}</div><div style="font-size:28px; font-weight:bold">{v}</div>'
        f'<div style="font-size:13px">{d}</div></div>' for t, v, d in tarjetas) + "</div>"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1, 1.3]})
    cm = m["cm"]
    ax1.imshow(cm, cmap="cividis")
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, f"{num(cm[i, j])}\n({pct(cm[i, j] / cm[i].sum(), 0)})", ha="center", va="center",
                     fontsize=16, color="white" if cm[i, j] < cm.max() * 0.6 else "black")
    ax1.set_xticks([0, 1], ["Predicho:\nNormal", "Predicho:\nNeumonía"])
    ax1.set_yticks([0, 1], ["Real:\nNormal", "Real:\nNeumonía"])
    ax1.grid(False)
    ax1.set_title("Matriz de confusión (test)")
    nombres = ["Acierto", "Sensibilidad", "Especificidad"]
    tuyos = [m["accuracy"], m["sensibilidad"], m["especificidad"]]
    tonta = [referencia, 1.0, 0.0]
    posiciones, ancho = np.arange(3), 0.38
    for desplazamiento, valores, color, trama, etiqueta in ((-ancho / 2, tuyos, AZUL, "", "Tu red"),
                                                           (ancho / 2, tonta, NARANJA, "//", "Siempre «Neumonía»")):
        barras = ax2.bar(posiciones + desplazamiento, valores, ancho, color=color, hatch=trama, edgecolor="black",
                         label=etiqueta)
        for barra, v in zip(barras, valores):
            ax2.annotate(pct(v), (barra.get_x() + barra.get_width() / 2, v), xytext=(0, 3),
                         textcoords="offset points", ha="center", fontsize=12)
    ax2.set_xticks(posiciones, nombres)
    ax2.set_ylim(0, 1.3)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.yaxis.set_major_formatter(FORMATO_PCT)
    ax2.legend(loc="upper center", ncol=2)
    ax2.set_title("Tu red frente a la referencia tonta")
    fig.tight_layout()
    mostrar(fig)
    nota("En la matriz, cada fila es lo que la radiografía es de verdad y cada columna lo que dice la red: la "
         "diagonal son aciertos. El porcentaje es sobre el total de su fila.")
    diferencia = m["accuracy"] - referencia
    nota(f"Tu red acierta el <b>{pct(m['accuracy'])}</b> en test; responder siempre «Neumonía» acierta el "
         f"{pct(referencia)} ({'+' if diferencia >= 0 else '−'}{num(abs(diferencia) * 100, 1)} puntos).")
    info("<b>¿Por qué importa tanto la sensibilidad?</b> En un cribado, una neumonía que se escapa (falso negativo) "
         "suele ser peor que una falsa alarma (falso positivo), que se aclara con otra prueba. Por eso se suele "
         "priorizar la sensibilidad. El equilibrio lo decide el equipo clínico según el contexto, no el modelo.")
    acc_val = entreno["historial"]["val_accuracy"][-1]
    if acc_val - m["accuracy"] > 0.05:
        pred_val = probabilidades(ESTADO["modelo"], ESTADO["datos"]["val"][0]) >= 0.5
        recall_val = {c: float((pred_val[yva == c] == c).mean()) for c in (0, 1)}
        recall_test = {0: m["especificidad"], 1: m["sensibilidad"]}
        c = max((0, 1), key=lambda c: recall_val[c] - recall_test[c])   # la clase en la que más empeora
        motivo = (f", y la red falla más con sus radiografías {('normales', 'con neumonía')[c]} "
                  f"({('especificidad', 'sensibilidad')[c]} del {pct(recall_val[c])} en validación frente al "
                  f"{pct(recall_test[c])} en test)") if recall_val[c] - recall_test[c] > 0.02 else ""
        nota(f"En validación acertaba el {pct(acc_val)} y en test el {pct(m['accuracy'])}. No tiene por qué ser "
             f"sobreajuste: el test sale de otro lote de radiografías, con menos neumonías ({pct(yte.mean())} frente "
             f"a {pct(yva.mean())}){motivo}. La validación sale del mismo lote que el entrenamiento y puede ser algo "
             "optimista.")
    if registrar(entreno, m):
        ok(f"Resultados anotados en el registro de experimentos (1.8) como n.º {entreno['id']}.")
    nota("Si estás haciendo un reto, ve a <b>1.8</b> y copia tu línea; luego, <b>1.6</b>. Si no, sigue en <b>1.6</b>.")


if "ESTADO" not in globals():
    __FALTA__
elif modelo_listo():
    _evaluar_test()
''')

# ---------------------------------------------------------------------------------------------
# 1.6 Qué ve cada capa
# ---------------------------------------------------------------------------------------------
md(r'''
## 1.6 Qué ve cada capa
Elige una radiografía del test. Verás los filtros que ha aprendido la primera capa y los **mapas de
activación** (feature maps) de cada bloque: dónde responde cada filtro dentro de la imagen.

Con «Neumonía aleatoria» o «Normal aleatoria» sale otra radiografía cada vez. Para volver a una concreta,
elige «Índice concreto» y escribe en `indice_imagen` su número (de 0 a 623: es el «Test n.º» del título).
Con las opciones aleatorias ese número no se usa.
''')

codigo("1.6 Qué ve cada capa", r'''
imagen_a_analizar = "Neumonía aleatoria"  # @param ["Neumonía aleatoria", "Normal aleatoria", "Índice concreto"]
indice_imagen = 0  # @param {type:"integer"}


def _que_ve(opcion, indice):
    modelo = ESTADO["modelo"]
    xte, yte = ESTADO["datos"]["test"]
    if opcion == "Índice concreto":
        try:
            i = int(indice)
        except (TypeError, ValueError):
            i = -1
        if not 0 <= i < len(yte):
            error(f"El índice tiene que ser un número entero entre 0 y {len(yte) - 1}. Cámbialo en el formulario y "
                  "vuelve a ejecutar la celda.")
            return
    elif opcion in ("Neumonía aleatoria", "Normal aleatoria"):
        ESTADO["aleatoria"] += 1   # otra imagen en cada ejecución, pero reproducible
        rng = np.random.default_rng(SEMILLA + ESTADO["aleatoria"])
        i = int(rng.choice(np.flatnonzero(yte == (1 if opcion.startswith("Neumonía") else 0))))
    else:
        error("Elige una opción del desplegable «imagen_a_analizar» y vuelve a ejecutar la celda.")
        return
    x = xte[i:i + 1]
    clase, prob = respuesta(float(probabilidades(modelo, x)[0]))

    # Imagen ampliada + filtros de la primera convolución
    pesos = modelo.get_layer("conv_1").get_weights()[0]   # (k, k, 1, filtros)
    k, n_filtros = pesos.shape[0], pesos.shape[-1]
    filas_filtros = int(np.ceil(n_filtros / 8))
    fig = plt.figure(figsize=(13.5, max(3.8, 1.2 * filas_filtros + 1.4)), layout="constrained")
    izquierda, derecha = fig.subfigures(1, 2, width_ratios=[1, 1.9])
    ax = izquierda.subplots()
    ax.imshow(x[0, ..., 0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.axis("off")
    ax.set_title(f"Test n.º {i} · Real: {CLASES[int(yte[i])]}\nPredicción: {clase} · {prob}", fontsize=14)
    axs = np.atleast_1d(derecha.subplots(filas_filtros, 8)).ravel()
    tope = float(np.abs(pesos).max()) or 1.0   # escala común y simétrica: gris medio = 0 en todos los filtros
    for j, a in enumerate(axs):
        a.axis("off")
        if j < n_filtros:
            a.imshow(pesos[:, :, 0, j], cmap="gray", vmin=-tope, vmax=tope, interpolation="nearest")
    derecha.suptitle(f"Los {n_filtros} filtros de la primera capa ({k}×{k} píxeles cada uno)", fontsize=14)
    mostrar(fig)
    nota(f"Cada filtro es una cuadrícula de {k}×{k} números aprendidos (pesos) que recorre toda la imagen (claro = peso "
         "positivo, oscuro = negativo, gris medio = cero; la escala es la misma para todos). Algunos se parecen a "
         "detectores de bordes o de cambios de brillo.")

    # Mapas de activación (salida de cada ReLU), los 8 más activos por bloque
    relus = [c.name for c in modelo.layers if c.name.startswith("relu_")]
    extractor = keras.Model(modelo.input, [modelo.get_layer(n).output for n in relus])
    mapas = extractor(x, training=False)
    if not isinstance(mapas, (list, tuple)):
        mapas = [mapas]
    n_bloques = len(relus)
    fig = plt.figure(figsize=(13.5, 2.2 * n_bloques + 0.3), layout="constrained")
    for b, (sub, mapa) in enumerate(zip(np.atleast_1d(fig.subfigures(n_bloques, 1)), mapas), start=1):
        a = mapa.numpy()[0]
        orden = np.argsort(a.mean(axis=(0, 1)))[::-1][:8]
        tendencia = ("detalle fino: brillo, bordes y contrastes" if b == 1 else "zonas más amplias (en esta red "
                     "pequeña, muchos filtros responden sobre todo al brillo)" if b == n_bloques else
                     "combinaciones de bordes: texturas y formas")
        sub.suptitle(f"Bloque {b} · {a.shape[0]}×{a.shape[1]} px · {tendencia}", fontsize=14, x=0.01, ha="left")
        for ax, canal in zip(sub.subplots(1, 8), list(orden) + [None] * 8):
            ax.axis("off")
            if canal is not None:
                ax.imshow(a[:, :, canal], cmap="viridis", vmin=0, vmax=float(a[:, :, canal].max()) or 1.0,
                          interpolation="nearest")   # morado = 0 de verdad; amarillo = el máximo de ese mapa
    mostrar(fig)
    nota("Cada mapa muestra dónde responde un filtro (de cada bloque, los 8 que más se activan con esta imagen): "
         "morado = nada, amarillo = donde más responde. En una red tan pequeña, muchos mapas se parecen a la propia "
         "radiografía, algo borrosa; en redes grandes, los primeros bloques suelen marcar bordes y los últimos, "
         "patrones más abstractos. Si varios mapas se parecen, esos filtros han aprendido cosas parecidas.")
    if opcion != "Índice concreto":
        nota("Cada vez que ejecutes la celda con una opción «aleatoria» saldrá otra radiografía.")


if "ESTADO" not in globals():
    __FALTA__
elif modelo_listo():
    _que_ve(imagen_a_analizar, indice_imagen)
''')

# ---------------------------------------------------------------------------------------------
# 1.7 Grad-CAM
# ---------------------------------------------------------------------------------------------
md(r'''
## 1.7 Dónde mira el modelo (Grad-CAM)
Grad-CAM pinta encima de la radiografía un mapa de calor con las zonas que más han empujado a la red
hacia su respuesta: amarillo = mucho, morado = poco. Es una aproximación, no una explicación completa
(Selvaraju et al., 2017). Verás 3 aciertos y 3 fallos del test.
''')

codigo("1.7 Dónde mira el modelo (Grad-CAM)", r'''
def _gradcam():
    modelo = ESTADO["modelo"]
    xte, yte = ESTADO["datos"]["test"]
    y = yte.astype(int)
    p = probabilidades(modelo, xte)
    pred = (p >= 0.5).astype(int)
    capa = [c.name for c in modelo.layers if c.name.startswith("relu_")][-1]
    lado = modelo.get_layer(capa).output.shape[1]
    submodelo = keras.Model(modelo.input, [modelo.get_layer(capa).output, modelo.get_layer("logit").output])

    # Grad-CAM de todo el test por lotes: gradiente de +logit (Neumonía) o −logit (Normal) según la predicción
    trozos = []
    for i in range(0, len(xte), 128):
        signo = tf.constant(np.where(pred[i:i + 128] == 1, 1.0, -1.0), dtype=tf.float32)
        with tf.GradientTape() as cinta:
            activaciones, logit = submodelo(tf.convert_to_tensor(xte[i:i + 128]), training=False)
            objetivo = tf.reduce_sum(logit[:, 0] * signo)
        gradientes = cinta.gradient(objetivo, activaciones)
        pesos = tf.reduce_mean(gradientes, axis=(1, 2))
        trozos.append(tf.nn.relu(tf.reduce_sum(activaciones * pesos[:, None, None, :], axis=-1)).numpy())
    mapas = np.concatenate(trozos)
    maximos = mapas.max(axis=(1, 2))
    vacio = maximos <= 0

    # 3 aciertos y 3 fallos, reproducibles; se prefieren mapas no vacíos y se mezclan las dos clases predichas
    orden = np.random.default_rng(SEMILLA).permutation(len(y))

    def elegir(candidatos):
        por_clase = {c: [i for i in candidatos if pred[i] == c and not vacio[i]] for c in (1, 0)}
        elegidos = []
        while len(elegidos) < 3 and (por_clase[1] or por_clase[0]):
            for c in (1, 0):
                if por_clase[c] and len(elegidos) < 3:
                    elegidos.append(por_clase[c].pop(0))
        return elegidos + [i for i in candidatos if i not in elegidos][:3 - len(elegidos)]

    aciertos = elegir([i for i in orden if pred[i] == y[i]])
    fallos = elegir([i for i in orden if pred[i] != y[i]])

    # Cada casilla del mapa se pinta sobre el píxel en el que está centrada: sin relleno, la capa no llega a los
    # bordes. centro = píxel del centro de la primera casilla; paso = píxeles entre casillas.
    centro, paso = 0.0, 1.0
    for c in modelo.layers:
        if isinstance(c, layers.Conv2D):
            centro += (c.kernel_size[0] - 1) / 2 * paso
        elif isinstance(c, layers.MaxPooling2D):
            centro, paso = centro + paso / 2, paso * 2
        if c.name == capa:
            break
    a, b = centro - paso / 2, centro + paso * (lado - 1) + paso / 2
    res = xte.shape[1]

    grupos = [("Aciertos", aciertos)] + ([("Fallos", fallos)] if fallos else [])
    fig = plt.figure(figsize=(13.5, 3.7 * len(grupos)), layout="constrained")
    for sub, (titulo, indices) in zip(np.atleast_1d(fig.subfigures(len(grupos), 1)), grupos):
        sub.suptitle(titulo, fontsize=16, fontweight="bold")
        for par, i in zip(sub.subfigures(1, 3), list(indices) + [None] * 3):
            axs = par.subplots(1, 2)
            for ax in axs:
                ax.axis("off")
            if i is None:
                continue
            clase, prob = respuesta(p[i])
            par.suptitle(f"Real: {CLASES[y[i]]}\nPredicción: {clase} · {prob}", fontsize=14)
            for ax, rotulo in zip(axs, ("Radiografía", "Dónde miró")):
                ax.imshow(xte[i, ..., 0], cmap="gray", vmin=0, vmax=1)
                ax.set_title(rotulo, fontsize=12)
            if vacio[i]:
                axs[1].text(0.5, 0.5, f"Grad-CAM no\nencuentra ninguna\nzona que empuje\nhacia «{clase}»",
                            transform=axs[1].transAxes, ha="center", va="center", fontsize=11,
                            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
            else:
                axs[1].imshow(mapas[i] / maximos[i], cmap="viridis", alpha=0.45, vmin=0, vmax=1,
                              extent=(a, b, b, a), interpolation="bilinear")
                axs[1].set_xlim(-0.5, res - 0.5)
                axs[1].set_ylim(res - 0.5, -0.5)
    mostrar(fig)
    info(f"El color indica en qué zonas se ha fijado la red para esta predicción: amarillo = mucho, morado = poco. "
         f"Es un mapa aproximado: sale de la capa <b>{capa}</b>, de {lado}×{lado} px, ampliada a {res}×{res} (el "
         "marco sin color es la zona que esa capa no cubre). "
         "Enseña dónde miró, no por qué acertó o falló; en radiografías reales estos mapas localizan peor que un "
         "radiólogo.")
    nota("El porcentaje es la probabilidad que da la red a su respuesta; cerca del 50 % significa que duda.")
    if lado <= 3:
        aviso(f"Con una última capa de solo {lado}×{lado} px el mapa es tan grueso que apenas señala zonas.")
    if not fallos:
        aviso("La red no falla ninguna radiografía del test: solo se muestran aciertos.")
    elif len(fallos) < 3:
        aviso(f"En test solo hay {len(fallos)} fallo{'s' if len(fallos) > 1 else ''}: se muestra todo lo que hay.")
    if any(vacio[i] for i in aciertos + fallos):
        nota("Cuando no hay mapa, ninguna zona empuja hacia esa clase. Puede pasar cuando la red decide «Normal» "
             "sobre todo porque no encuentra indicios de neumonía.")


if "ESTADO" not in globals():
    __FALTA__
elif modelo_listo():
    _gradcam()
''')

md(r'''
> 🗣️ **Para el debate: aprendizaje por atajos.** En 2018, Zech y colaboradores entrenaron CNN para detectar
> neumonía en unas 158.000 radiografías de adultos de tres sistemas hospitalarios de EE. UU. En 3 de 5
> comparaciones, los modelos rendían peor fuera del hospital en el que habían aprendido, y una CNN adivinaba
> de qué hospital venía cada placa en más del 95 % de los casos, a veces fijándose en una marca metálica de
> la esquina. Como la proporción
> de neumonías cambiaba mucho entre hospitales, saber de dónde venía la placa ya ayudaba a acertar sin mirar el
> pulmón. Un buen resultado en el test no garantiza que el modelo mire donde creemos.
>
> Según los autores del conjunto original, las radiografías de PneumoniaMNIST vienen de un único centro, así que
> ese atajo concreto no debería darse aquí: es un riesgo cuando un modelo entrenado en un sitio se usa en otro.
>
> 1. Nuestras radiografías son de niños y, según sus autores, salen de un único centro. ¿Qué atajos podría estar
>    aprendiendo el modelo que no veríamos en el test?
> 2. Mira tus mapas de Grad-CAM: ¿se fijan en los pulmones o en otras zonas? ¿Te fiarías de este modelo en otro
>    hospital?
>
> <small>Zech JR et al. *PLOS Medicine* 15(11): e1002683 (2018), doi:10.1371/journal.pmed.1002683.</small>
''')

# ---------------------------------------------------------------------------------------------
# 1.8 Registro de experimentos
# ---------------------------------------------------------------------------------------------
md(r'''
## 1.8 Registro de experimentos
Cada entrenamiento de 1.4 queda anotado aquí, y 1.5 le añade las métricas de test. Ejecuta la celda para
ver la tabla, el gráfico y la línea que tienes que copiar en la hoja común de la clase.
''')

codigo("1.8 Registro de experimentos", r'''
def _linea_resumen(fila):
    """Cabecera y fila separadas por tabuladores, con coma decimal: se pegan en columnas en la hoja de cálculo."""
    def valor(v, decimales):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.{decimales}f}".rstrip("0").rstrip(".").replace(".", ",")
        return " ".join(str(v).split())
    columnas = [c for c in COLUMNAS if c != "id"]
    return "\t".join(columnas) + "\n" + "\t".join(valor(fila[c], 1 if c == "tiempo_entrenamiento_s" else 3)
                                                   for c in columnas)


def _registro():
    filas = ESTADO["experimentos"]
    if not filas:
        aviso("Todavía no hay experimentos. Entrena una red en 1.4, evalúala en 1.5 y vuelve a ejecutar esta celda.")
        return
    mostrar_experimentos(filas)
    ultima = filas[-1]
    cabecera, datos = _linea_resumen(ultima).split("\n")
    nota(f"<b>Línea para la hoja común de la clase</b> (experimento n.º {ultima['id']}, "
         f"«{html.escape(ultima['experimento'])}»). Salen dos líneas: la de arriba son los nombres de las columnas, "
         "que ya están en la hoja. Copia la de abajo, entera, y pégala en la pestaña «Resultados» de la hoja común, "
         "en tu fila (en la pestaña «Reparto» ves cuál es; si no tienes, debajo de «Desde aquí, libres»): haz clic "
         "en la columna A de tu fila y pega. Cada valor cae en su columna.")
    display(HTML('<pre style="font-size:12px; padding:6px; overflow-x:auto; opacity:0.7">'
                 + html.escape(cabecera) + "</pre>"))
    display(HTML('<pre style="font-size:13px; padding:8px; border:1px solid #8888; overflow-x:auto; user-select:all">'
                 + html.escape(datos) + "</pre>"))
    if not ultima["reto"]:
        aviso("Esta línea sale sin reto ni hipótesis: la 1.9 no estaba ejecutada cuando entrenaste. Si estás haciendo "
              "un reto, escríbelos a mano en esas dos columnas de la hoja y ejecuta la 1.9 antes de volver a entrenar.")
    if ultima["accuracy"] is None:
        aviso("Este experimento aún no tiene métricas de test: ejecuta 1.5 y vuelve aquí.")
    if not guardar_csv():
        return
    que_es = "tu tabla con todos tus experimentos (se abre en Excel). Para la hoja común basta con la línea de arriba"
    if not EN_COLAB:
        ok(f"Tabla guardada en <code>{html.escape(os.path.abspath(RUTA_CSV))}</code>. Es {que_es}.")
        return
    firma = hash(repr(filas))   # solo se descarga si el registro ha cambiado desde la última descarga
    if ESTADO.get("descargado") == firma:
        nota(f"<code>experimentos.csv</code> es {que_es}. No ha cambiado desde la última descarga; también está en "
             f"el panel de archivos (a la izquierda), carpeta <code>{CARPETA_SALIDAS}</code>.")
        return
    try:
        from google.colab import files
        files.download(RUTA_CSV)
        ESTADO["descargado"] = firma
        ok(f"Descargando <code>experimentos.csv</code>: {que_es}. Si no se descarga, búscalo en el panel de "
           f"archivos (a la izquierda), carpeta <code>{CARPETA_SALIDAS}</code>.")
    except Exception:
        aviso(f"No he podido lanzar la descarga de <code>experimentos.csv</code>, {que_es}. Búscalo en el panel de "
              f"archivos (a la izquierda): <code>{RUTA_CSV}</code>, menú ⋮ → Descargar.")


if "ESTADO" not in globals():
    __FALTA__
else:
    _registro()
''')

# ---------------------------------------------------------------------------------------------
# 1.9 Retos guiados
# ---------------------------------------------------------------------------------------------
md(r'''
## 1.9 Retos guiados
Cada uno trabaja en su portátil. El profesor dirá en clase qué reto te toca (A, B, C o D). En
`nombre_experimento` (1.3) pon tus iniciales y el reto, por ejemplo «ALM_A». Antes de entrenar, elige tu reto,
**escribe tu hipótesis** en la celda de abajo (¿qué crees que pasará?) y pulsa ▶ en esa celda: si no la
ejecutas, tu línea saldrá sin reto ni hipótesis.

Para cada entrenamiento, cambia en 1.3 lo que pide el reto y ejecuta 1.3 → 1.4 → 1.5 → 1.8. En 1.8, copia
tu línea en la hoja común (el enlace está en el chat de la clase). En los retos A y C entrenas dos veces,
así que copias dos líneas. Si vas a entrenar otra vez, pega antes tu línea en la hoja: 1.8 solo enseña la del
último entrenamiento.

Empieza cada reto desde la red por defecto: __RED_DEFECTO__ Si has tocado algo, vuelve a estos valores
antes de empezar.

**A · Profundidad: 1 frente a 3 bloques.** En 1.3 pon `bloques_convolucionales` en 1 y entrena; luego en 3
y entrena otra vez. Mira el número de parámetros de la ficha, el tamaño de los mapas en 1.6 y la
sensibilidad y la especificidad en 1.8.

**B · Sin pooling.** En 1.3 desmarca `usar_pooling` y entrena. Compara con la red por defecto el número de
parámetros, el tiempo de entrenamiento y las métricas.

**C · Dropout 0 frente a 0,5 (sin pooling, 20 épocas).** En 1.3 desmarca `usar_pooling`, pon `epocas` en 20
y entrena dos veces: con `dropout` en 0 y con `dropout` en 0,5 (en el deslizador sale 0.5). Mira las curvas
de pérdida en 1.4: ¿se separan la de entrenamiento y la de validación? ¿Qué dice el diagnóstico en cada caso?
Si vas justo de tiempo, entrena solo con dropout 0 y compara con la fila de referencia que pega el profesor.

**D · Compensar el desbalanceo.** En 1.3 marca `compensar_desbalanceo` y entrena. Compara la sensibilidad y la
especificidad con las de la red por defecto.

**Libre.** Cualquier otra combinación: anota en la hipótesis qué cambias y qué esperas.
''')

codigo("1.9 Elige tu reto y escribe tu hipótesis", r'''
reto = "A · Profundidad (1 frente a 3 bloques)"  # @param ["A · Profundidad (1 frente a 3 bloques)", "B · Sin pooling", "C · Dropout 0 frente a 0,5 (sin pooling, 20 épocas)", "D · Compensar el desbalanceo", "Libre"]
hipotesis = ""  # @param {type:"string"}


def _guardar_reto(reto, hipotesis):
    letra = str(reto).split(" ·")[0].strip()
    if letra not in ("A", "B", "C", "D", "Libre"):
        error("Elige un reto del desplegable y vuelve a ejecutar la celda.")
        return
    texto = " ".join(str(hipotesis).split())
    ESTADO["reto"], ESTADO["hipotesis"] = letra, texto
    if texto:
        ok(f"Guardado: reto <b>{html.escape(str(reto))}</b>. Hipótesis: «{html.escape(texto)}». Se anotarán en los "
           "experimentos que entrenes a partir de ahora. Comprueba que es el reto que te ha dado el profesor.")
    else:
        aviso(f"Reto <b>{html.escape(str(reto))}</b> guardado, pero la hipótesis está vacía. Comprueba que es el reto "
              "que te ha dado el profesor, escribe qué crees que pasará y vuelve a ejecutar esta celda antes de "
              "entrenar.")
    nota("Ahora sube a <b>1.3</b>, cambia lo que pide el reto y ejecuta 1.3 → 1.4 → 1.5 → 1.8.")


if "ESTADO" not in globals():
    __FALTA__
else:
    _guardar_reto(reto, hipotesis)
''')

# ---------------------------------------------------------------------------------------------
# 1.10 Plan B del profesor
# ---------------------------------------------------------------------------------------------
md(r'''
## 1.10 Plan B (solo el profesor)
**No ejecutes esta celda.** La usa el profesor si en clase no da tiempo a entrenar los retos: entrena una
detrás de otra las siete configuraciones de referencia (A y C comparan dos variantes; B y D se comparan con
la red por defecto), las evalúa en test y las añade al registro. Con la casilla desmarcada solo muestra qué
entrenaría y cuánto tardaría.
''')

codigo("1.10 Plan B (solo el profesor)", r'''
ejecutar_plan_b = False  # @param {type:"boolean"}

RETO_C = __RETO_C__
PLAN_B = [("ref_defecto", "—", {}),
          ("ref_A_1_bloque", "A", {"bloques": 1}),
          ("ref_A_3_bloques", "A", {"bloques": 3}),
          ("ref_B_sin_pooling", "B", {"pooling": False}),
          ("ref_C_dropout_0_20_epocas", "C", {**RETO_C, "dropout": 0.0}),
          ("ref_C_dropout_05_20_epocas", "C", {**RETO_C, "dropout": 0.5}),
          ("ref_D_compensar_desbalanceo", "D", {"compensar": True})]


def _plan_b(ejecutar):
    configuraciones = [(nombre, reto_ref, {**DEFECTO, **cambios_ref, "resolucion": ESTADO["resolucion"],
                                           "nombre": nombre})
                       for nombre, reto_ref, cambios_ref in PLAN_B]
    if not ejecutar:
        estado = display(HTML("⏱️ Midiendo cuánto tardaría cada configuración…"), display_id=True)
        filas, total, medidas = [], 0.0, {}
        for nombre, reto_ref, cfg in configuraciones:
            modelo = construir_modelo(cfg)
            arquitectura = tuple(cfg[k] for k in ("resolucion", "bloques", "filtros", "duplicar", "kernel", "pooling",
                                                  "batchnorm", "densa"))   # el dropout apenas cambia el tiempo
            if arquitectura not in medidas:
                medidas[arquitectura] = medir_lotes(modelo)
            segundos = segundos_estimados(medidas[arquitectura], cfg["epocas"])
            total += segundos
            filas.append({"Configuración": nombre, "Reto": reto_ref, "Cambios respecto a la red por defecto": cambios(cfg),
                          "Tiempo estimado": f"~{num(segundos)} s"})
        estado.update(HTML(""))
        tabla_html(pd.DataFrame(filas))
        info(f"Marca <b>ejecutar_plan_b</b> y vuelve a ejecutar la celda para entrenar estas {len(filas)} redes una "
             f"detrás de otra: unos {duracion(total)} {DONDE}. Se evalúan en test y se añaden al registro de "
             "experimentos (1.8).")
        return
    lineas, nuevas = [], []
    progreso = display(HTML(""), display_id=True)
    try:
        for n, (nombre, reto_ref, cfg) in enumerate(configuraciones, start=1):
            lineas.append(f"▶ {n}/{len(configuraciones)} · {nombre} · entrenando…")
            progreso.update(HTML('<div style="font-size:15px">' + "<br>".join(lineas) + "</div>"))
            modelo, historial, segundos = entrenar(cfg)
            etiqueta, _, _ = diagnosticar(modelo, historial)
            metricas = evaluar(modelo)
            ESTADO["experimentos"] = [f for f in ESTADO["experimentos"] if f["experimento"] != nombre]
            ESTADO["contador"] += 1
            entreno = dict(id=ESTADO["contador"], nombre=nombre, reto=reto_ref, hipotesis="", config=cfg,
                           parametros=modelo.count_params(), segundos=segundos, historial=historial,
                           diagnostico=etiqueta)
            registrar(entreno, metricas)
            nuevas.append(ESTADO["experimentos"][-1])
            # Así 1.5-1.7 funcionan aunque nadie entrenara; si el modelo actual ya es un ref_defecto (Plan B repetido),
            # se sustituye para que 1.5 no vuelva a añadir la fila antigua que se acaba de quitar.
            if nombre == "ref_defecto" and (ESTADO["modelo"] is None or ESTADO["entreno"]["nombre"] == "ref_defecto"):
                ESTADO["modelo"], ESTADO["entreno"] = modelo, entreno
            resumen = "sin métricas (valores no válidos)" if metricas is None else (
                f"acierto (test) {pct(metricas['accuracy'])} · sensibilidad {pct(metricas['sensibilidad'])} · "
                f"especificidad {pct(metricas['especificidad'])}")
            lineas[-1] = f"✔ {n}/{len(configuraciones)} · {nombre} · {num(segundos, 1)} s · {resumen} · {etiqueta}"
            progreso.update(HTML('<div style="font-size:15px">' + "<br>".join(lineas) + "</div>"))
    except KeyboardInterrupt:
        aviso(f"Plan B detenido: se han registrado {len(nuevas)} de {len(configuraciones)} configuraciones.")
    if nuevas:
        ok(f"Plan B terminado: {len(nuevas)} configuraciones añadidas al registro de experimentos (1.8).")
        mostrar_experimentos(nuevas)


if "ESTADO" not in globals():
    __FALTA__
elif requiere("datos"):
    _plan_b(bool(ejecutar_plan_b))
''')

# ---------------------------------------------------------------------------------------------
# 1.11 Cierre
# ---------------------------------------------------------------------------------------------
md(r'''
## 1.11 Cierre

**Para pensar**
1. En la tabla de la clase, ¿qué cambio de arquitectura ha movido más la sensibilidad y la especificidad?
   ¿Y el número de parámetros? ¿Son diferencias grandes o de pocos puntos, que podrían ser azar?
2. ¿Más parámetros han dado siempre mejores resultados en test?
3. Si este modelo se usara para cribar, ¿qué preferirías: más falsas alarmas o más neumonías que se escapan?
   ¿Quién debería decidirlo?

**Puente al cuaderno 2.** El detector de personas del cuaderno 2, YOLO26 nano, es sobre todo convolucional,
como tu red, pero con unos 2,6 millones de parámetros (la tuya por defecto tiene unos 56.000) y un par de
bloques de atención (una pieza que deja que cada zona de la imagen tenga en cuenta lo que pasa en las demás).
Viene ya entrenado con COCO, un gran conjunto de fotos etiquetadas con 80 tipos de objeto, y allí no vas a
entrenar nada: solo a usarlo.
''')


def construir():
    nb = new_notebook()
    for i, celda in enumerate(CELDAS):
        celda.id = f"nb1-{i:02d}"
        if celda.cell_type == "code":
            celda.source = celda.source.replace("__FALTA__", FALTA_PREPARACION).replace("__RETO_C__", repr(RETO_C))
    nb.cells = CELDAS
    nb.metadata = {"colab": {"provenance": [], "toc_visible": True},
                   "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}}
    nbformat.validate(nb)
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, DESTINO)
    return nb


if __name__ == "__main__":
    nb = construir()
    print(f"{DESTINO.relative_to(RAIZ)}: {len(nb.cells)} celdas "
          f"({sum(c.cell_type == 'code' for c in nb.cells)} de código)")
