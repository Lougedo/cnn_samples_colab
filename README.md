# Una CNN por dentro + Contar personas en vídeo

Dos cuadernos de Google Colab y su material docente para la parte de herramientas de la última sesión de IASP
(S7, jueves 24-sep-2026, online). El alumno no escribe código: cambia controles de formulario y pulsa ▶.

| Cuaderno | Qué hace el alumno | Abrir |
|---|---|---|
| `notebooks/01_laboratorio_cnn_radiografias.ipynb` | Diseña una CNN con formularios, la entrena con radiografías de PneumoniaMNIST (28 o 64 px) y ve su ficha, las curvas en vivo, sensibilidad y especificidad, los mapas de activación y el Grad-CAM. Cada entrenamiento queda en un registro exportable a CSV. Retos A-D y un Plan B para el profesor. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lougedo/cnn_samples_colab/blob/main/notebooks/01_laboratorio_cnn_radiografias.ipynb) |
| `notebooks/02_conteo_personas_cafeteria.ipynb` | Aplica YOLO26 nano, ya entrenado, a un vídeo: detecta personas, las sigue, cuenta cuántas hay por zona y cuántas cruzan una línea, y exporta solo recuentos (CSV y JSON). Envío opcional a un webhook de n8n: con el campo vacío no se envía nada. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lougedo/cnn_samples_colab/blob/main/notebooks/02_conteo_personas_cafeteria.ipynb) |

Los dos son prototipos docentes. **POC ≠ producción:** llevar algo así a un hospital o a un local real exige medir el
error con datos propios, decidir dónde se procesa y qué se guarda, y cumplir la normativa. El cuaderno 2 lo dice en su
cierre; el cuaderno 1 avisa de que no es apto para uso clínico.

**En la S7** hay 40 minutos, no los 60 del brief: bloque 8 (CNN Explainer, cuaderno 1 y Grad-CAM, 20:30-20:58) y
bloque 9 (cuaderno 2, 20:58-21:10). Cada alumno trabaja solo en su portátil, con un reto asignado; pone de nombre de
experimento `iniciales_reto` (p. ej. `ALM_A`) y pega su línea de la celda 1.8 en su fila de una hoja común cuyo enlace va
al chat. Entregable: su cuaderno 1 ejecutado, con la arquitectura que ha elegido y su Grad-CAM.

**Antes de clase.** Resumen; el detalle está en `docs/guion_docente_60min.md` (§1), `INFORME_FINAL.md` (secciones 5 y 6)
y `sesiones/S7/05-checklist.md`.

- ✅ Publicado en GitHub (https://github.com/Lougedo/cnn_samples_colab); los badges ya apuntan ahí. Copia en Drive: ver sección 1.
- 🔴 Crear la hoja común de resultados, con la cabecera que da la primera línea de la celda 1.8 y una fila asignada a
  cada alumno.
- 🟡 Abrir los dos cuadernos en Colab desde el enlace que vas a repartir, con una ventana privada: vista de
  formulario, primera instalación, tiempos, vídeo y descargas.
- 🟡 Ejecutar el Plan B (1.10) en tu Colab para tener la tabla de referencia a mano, y la demo 2 si la vas a usar.

**Documentación**
- [`docs/guion_docente_60min.md`](docs/guion_docente_60min.md): guion minuto a minuto, versión de 60 min y versión de 40 min para la S7, con los resultados de referencia medidos.
- [`docs/hoja_alumno.md`](docs/hoja_alumno.md): una página para el alumno.
- [`DECISIONS.md`](DECISIONS.md): cada decisión tomada ante una ambigüedad (D01-D52).
- [`INFORME_FINAL.md`](INFORME_FINAL.md): qué se construyó, medidas, limitaciones y lo que hay que comprobar a mano en Colab.
- [`BRIEF.md`](BRIEF.md): el encargo original.

---

## 1. Abrir en Colab

### Desde GitHub (badge)

Los badges apuntan a este repositorio: <https://github.com/Lougedo/cnn_samples_colab>. El enlace para el chat:

```
https://colab.research.google.com/github/Lougedo/cnn_samples_colab/blob/main/notebooks/01_laboratorio_cnn_radiografias.ipynb
https://colab.research.google.com/github/Lougedo/cnn_samples_colab/blob/main/notebooks/02_conteo_personas_cafeteria.ipynb
```

Si haces un fork, cambia el enlace en el generador y regenera (sección 5); el comando solo toca las líneas de los badges:

```bash
sed -i '' '/badge\.svg/ s#Lougedo/cnn_samples_colab#tu_usuario/tu_repo#' README.md src/build_nb1.py src/build_nb2.py   # macOS
.venv/bin/python src/build_nb1.py && .venv/bin/python src/build_nb2.py
```

Al abrir desde GitHub, Colab avisa de que el cuaderno no lo ha creado Google. Es normal: «Ejecutar de todos modos»
(🟡 rótulo exacto sin comprobar). Los dos cuadernos lo explican en su portada.

### Desde Drive

1. Sube el `.ipynb` a Drive y ábrelo con doble clic o con «Abrir con → Google Colaboratory».
2. Otra vía: en Colab, «Archivo → Subir cuaderno». 🟡 La interfaz cambió en julio de 2026 y el diálogo clásico de
   «Abrir cuaderno» está ahora en «Recientes → Ver más» (colabtools#6069).
3. Reparte un **enlace directo** (`colab.research.google.com/drive/<ID>`) en lugar de describir menús en clase.
   🟡 Compártelo como lector: cada alumno ejecuta en su propio entorno y, si quiere conservar el cuaderno con sus
   resultados, guarda una copia en su Drive.

Si repartes por Drive, el badge de la portada no sirve: cámbialo o quítalo para que nadie pulse un enlace roto.

---

## 2. Requisitos y tiempos medidos

**Alumnos:** cuenta de Google y navegador. Mejor **Chrome**: las descargas, la subida de archivos y el vídeo
incrustado dan menos problemas. Colab gratuito en CPU; no hace falta instalar nada en el ordenador.

**Lo que instalan los cuadernos**, solo si falta: `medmnist==3.0.2` (NB1); `ultralytics==8.4.161`,
`supervision==0.30.5`, `lap>=0.5.12` y `shapely>=2.0.0` (NB2). Lo que Colab ya trae (numpy, TensorFlow, Keras, torch,
pandas…) se congela con un archivo de restricciones para que pip no lo actualice y no haya que reiniciar la sesión
(D03, D04; 🟡 sin probar en Colab real).

**Lo que se descarga en clase**

| Qué | Tamaño | De dónde |
|---|---|---|
| PneumoniaMNIST 28 px / 64 px | 4,2 MB / 20,6 MB | Zenodo (en local: 2,4-5,3 s y 8,8-11,4 s) |
| Pesos `yolo26n.pt` | unos 5 MB | GitHub (releases de Ultralytics) |
| Demo 1 «personas caminando» / demo 2 «andén de metro» | 7,6 MB / 128,5 MB | media.roboflow.com, vía `supervision` |

**Tiempos.** Medidos el 24-sep-2026 en un Apple M3 Pro (11 núcleos) con Python 3.12.7, TensorFlow 2.21.0,
Keras 3.15.1 y ultralytics 8.4.161, en serie y sin otros procesos. La columna «Limitado» imita la CPU de Colab.

| Qué | Todos los hilos | Limitado | Colab gratuito 🟡 | Presupuesto del brief |
|---|---|---|---|---|
| NB1 · 1.4 con la red por defecto (8 épocas) | 5,5 s (entrena 4,0 s) | 8,2 s, 2 hilos (entrena 6,7 s) | ~20-30 s | ≤ 90 s ✅ |
| NB1 · cuaderno completo por defecto | 26 s | 30 s | ~1-1,5 min | — |
| NB1 · 1.10 Plan B (7 entrenamientos) | 54 s | 121 s, 2 hilos | ~5-7 min | — |
| NB1 · 1.1 con todo instalado | ~9 s | ~9 s | primera instalación [🔴 Pendiente de confirmar] | — |
| NB2 · 2.3 + 2.5 + 2.6, demo 1 (10 s, salto 2) | 18,6 s | 24,7 s, 1 hilo | ~60-90 s | ≤ 2 min ✅ |
| NB2 · 2.3 + 2.5 + 2.6, demo 2 (10 s, salto 2) | 25,2 s | 31,3 s, 1 hilo | ~80-110 s, al límite | ≤ 2 min |
| NB2 · cuaderno completo, demo 1 | 26 s | 31 s | ~1,5-2 min + primera instalación [🔴 Pendiente de confirmar] | — |

Por qué ese «Limitado»: Colab gratuito tiene 2 vCPU (🟡: Google no lo publica; sale de fuentes de la comunidad) y
Ultralytics fija torch a `min(8, núcleos − 1)` hilos, o sea 1 en Colab. La proyección aplica un factor de 2,5-3,5 sobre
la medida limitada, que sale de comparar un núcleo del M3 Pro con un Xeon a 2,2 GHz (🟡, no medido en Colab). Con la
demo 2, la celda 2.3 estima el tiempo antes de procesar y, si se pasa, propone saltar 3 fotogramas.

El NB1 también pasó con las versiones de Colab (Python 3.13, Keras 3.13.2, numpy 2.1.3, pandas 2.2.3) con las mismas
métricas.

---

## 3. Problemas frecuentes

**Zenodo va lento o no responde (celda 1.2).** La celda descarga con su propio código: cada intento espera como
mucho 60 s sin recibir datos (un servidor que envía muy despacio sin llegar a pararse no se corta) y el archivo se
comprueba con su MD5. A 28 px, si Zenodo falla, prueba un espejo de Hugging Face: no es oficial, pero es idéntico byte
a byte (D14). A 64 px no hay espejo: vuelve a 28 px. Si aun así falla, descarga `pneumoniamnist.npz` (28 px) o
`pneumoniamnist_64.npz` de <https://doi.org/10.5281/zenodo.10519652>, súbelo desde el panel Archivos a
`datos/medmnist/` (la celda crea la carpeta en el primer intento) y vuelve a ejecutar 1.2.

**No hay GPU.** Es lo esperado. Todo está pensado para CPU y los tiempos de arriba son de CPU; no cambies el tipo de
entorno. Los cuadernos no piden GPU al abrirse. Si el entorno tiene una, la detectan y la usan (🟡 no probado: toda la
verificación fue en CPU).

**El vídeo anotado no se ve (2.5 y 2.6).** El cuaderno lo recodifica a H.264 (`libx264`, `yuv420p`, máximo 720p) y lo
incrusta en la salida. Prueba en Chrome. El archivo queda en `datos/tmp/aforo.mp4` y `datos/tmp/linea.mp4`: panel
Archivos → ⋮ → Descargar. Si pasa de 20 MB no se incrusta, para no bloquear el navegador, y la celda dice dónde está
(los de la demo 1, 10 s a 720p, ocupan unos 1,3-1,4 MB). Si falla la conversión, la celda lo avisa
y los recuentos siguen siendo válidos.

**Se ha reiniciado o desconectado la sesión.** Lo que vive en memoria se pierde, incluido el registro de experimentos
de 1.8, que no se vuelve a leer del CSV. Ejecuta otra vez desde 1.1 (NB1) o 2.1 (NB2), en orden; si te saltas un paso,
la celda dice cuál falta. Para no perder experimentos, ejecuta 1.8 después de cada entrenamiento: en Colab descarga
`experimentos.csv` (separado por `;`, con coma decimal; se abre en Excel) y te da la línea para la hoja común. Si has
subido un vídeo propio, vuelve a subirlo.

**Colab avisa de que el cuaderno no lo ha creado Google.** Pasa al abrir desde GitHub. «Ejecutar de todos modos» (🟡).

**Colab no conecta o pide una cuenta.** Hace falta iniciar sesión con una cuenta de Google. Si el ▶ sigue girando varios
minutos: Chrome, u otra pestaña con el mismo enlace.

**1.8 descarga `experimentos.csv` cada vez que hay experimentos nuevos.** En Colab, 1.8 lo descarga siempre que el registro ha cambiado desde la
última descarga: quien entrena dos veces lo recibe dos veces. Es normal; si el navegador pregunta por descargas
múltiples, permítelas. Para la hoja común no hace falta abrirlo.

**La línea pegada en la hoja común cae entera en la columna A.** Datos → Dividir texto en columnas (🟡 rótulo sin
comprobar), o que el alumno escriba en el chat reto, parámetros, sensibilidad, especificidad y diagnóstico.

**Python 3.13.** Colab usa Python 3.13 desde el 25-ago-2026 (colabtools#6081). El NB1 se probó en local con Python 3.13
y las versiones de Colab; el NB2 no se pudo emular en macOS (ultralytics 8.4.161 excluye numpy 2.0-2.3.4 solo en
macOS). Si la instalación falla o pide reiniciar: paleta de comandos de Colab → «Cambiar versión del entorno» (en
inglés, «Change runtime version») → 2026.07, que es Python 3.12. 🟡 Rótulo en español sin comprobar; el cambio no se
mantiene entre sesiones.

**Muchas conexiones desde la misma red.** Colab ha llegado a bloquear IP de centros educativos por tráfico
simultáneo (colabtools#5996). En la S7 cada alumno se conecta desde su casa, así que el riesgo es bajo; en una clase
presencial con una sola wifi, tenlo en cuenta.

**No se descarga el CSV o el JSON.** `files.download` solo funciona en la pestaña que ejecutó la celda y ha dado
problemas en Safari y Firefox. Usa Chrome o el panel Archivos → ⋮ → Descargar. En 2.8 el navegador puede pedir
permiso para descargar varios archivos: acepta.

**Subir un vídeo propio (2.2).** El widget de Colab es lento con archivos grandes y da problemas en Firefox. Mejor un
clip de 10-30 s, en 1280×720 o 1920×1080 y de menos de 50 MB. Fuera de Colab, cópialo a `datos/videos/mi_video.mp4`.
Si vas a usar la demo 2 (128,5 MB), ejecútala una vez antes de clase para que ya esté descargada.

**La celda 1.3 no deja entrenar.** Es a propósito: si la imagen se queda en 0×0 píxeles, o la red pasa de 20 millones
de parámetros o de 600 s estimados, se bloquea con un mensaje que dice qué cambiar; entre 90 y 600 s solo avisa (D16).

---

## 4. Estructura

```
cnn-sesion-final/
├── README.md
├── BRIEF.md                          el encargo
├── DECISIONS.md                      decisiones ante ambigüedades
├── INFORME_FINAL.md                  verificación, limitaciones y comprobaciones en Colab
├── notebooks/
│   ├── 01_laboratorio_cnn_radiografias.ipynb   generado, sin salidas
│   └── 02_conteo_personas_cafeteria.ipynb      generado, sin salidas
├── src/
│   ├── build_nb1.py                  genera el cuaderno 1 con nbformat
│   └── build_nb2.py                  genera el cuaderno 2 con nbformat
├── scripts/
│   └── verificar.py                  ejecuta los cuadernos sin interfaz y resume
├── docs/
│   ├── guion_docente_60min.md        guion de 60 min y versión de 40 min para la S7
│   ├── hoja_alumno.md                una página para el alumno
│   └── notas_verificacion/           a-h: APIs, versiones, fuentes, construcción y ensayo de CNN Explainer
├── requirements-verificacion.txt     entorno local de verificación
└── .gitignore
```

Fuera de git (`.gitignore`), se crean al ejecutar: `datos/` (los `.npz`, los vídeos de demo y `tmp/` con los vídeos
anotados), `pesos/` (`yolo26n.pt`), `salidas/` (CSV y JSON que exportan los cuadernos), `verificacion/` (copias
ejecutadas, informes y medidas) y los entornos `.venv/` y `.venv-colab/`.

---

## 5. Regenerar y verificar

Los `.ipynb` no se editan a mano: se generan con `src/build_nb*.py`, que son deterministas (dos ejecuciones dan el
mismo archivo byte a byte), fijan `cellView: "form"` en las celdas de código y guardan los cuadernos sin salidas.

```bash
# Entorno local (Python 3.12, con uv). Los alumnos no lo necesitan.
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-verificacion.txt

# Regenerar
.venv/bin/python src/build_nb1.py
.venv/bin/python src/build_nb2.py

# Verificar sin interfaz
.venv/bin/python scripts/verificar.py                   # modos rapido + defecto, los dos cuadernos
.venv/bin/python scripts/verificar.py --construir       # regenera antes de verificar
.venv/bin/python scripts/verificar.py --nb 1 --modos rapido defecto imposible referencias
.venv/bin/python scripts/verificar.py --nb 1 --modos defecto referencias --hilos 2        # proxy de Colab
.venv/bin/python scripts/verificar.py --nb 2 --modos defecto --hilos 1
.venv/bin/python scripts/verificar.py --nb 2 --modos defecto --set fuente_video="Demo 2: andén de metro (interior)"
```

| Modo | Qué ejecuta |
|---|---|
| `rapido` | NB1 con 2 épocas; NB2 con 3 s de vídeo |
| `defecto` | los valores por defecto de los formularios |
| `referencias` | NB1 con el Plan B activado (las 7 configuraciones de los retos) |
| `imposible` | NB1 con 4 bloques a 28 px: tiene que salir el aviso de 0×0, sin excepciones |

| Opción | Uso |
|---|---|
| `--nb 1\|2\|todos` | qué cuaderno (por defecto, los dos) |
| `--modos …` | uno o varios modos (por defecto, `rapido defecto`) |
| `--hilos N` | limita TensorFlow y PyTorch a N hilos, como aproximación a Colab gratuito |
| `--set NOMBRE=VALOR` | fija un control de formulario, como si se tocara en Colab; se puede repetir |
| `--timeout S` | segundos máximos por celda (1800 por defecto) |
| `--construir` | ejecuta los dos generadores antes |

Cómo funciona: reescribe la línea `nombre = valor  # @param` antes de ejecutar con nbclient (D05), sin tocar los
`.ipynb` del repositorio. Deja las copias ejecutadas en `verificacion/<modo>/` y un `informe_<fecha>.md` y un
`resultados_<fecha>.json` en `verificacion/`. Devuelve 1 si hay errores, si falta el aviso del modo `imposible` o si
algún cuaderno del repositorio incumple la estructura (celda sin `cellView: "form"`, primera línea distinta de
`# @title` o salidas guardadas). Borra `salidas/nb1` o `salidas/nb2` antes de cada ejecución.

Los dos entornos locales:

- `.venv`: Python 3.12.7 con las versiones exactas de `requirements-verificacion.txt`. Sin ffmpeg del sistema en el
  Mac, el NB2 usa el de `imageio-ffmpeg` (D36). Sin pip: la celda de instalación solo llama a pip si falta algo (D04).
- `.venv-colab`: Python 3.13 con las versiones de Colab (Keras 3.13.2, numpy 2.1.3, pandas 2.2.3, matplotlib 3.10.0,
  IPython 7.34.0), solo para el NB1: `.venv-colab/bin/python scripts/verificar.py --nb 1 --modos rapido defecto`.
  No hay fichero de requisitos en git para este entorno; las versiones salen de googlecolab/backend-info (23-sep-2026).

---

## 6. Licencias y avisos

**PneumoniaMNIST: CC BY 4.0. No apto para uso clínico.** Radiografías de tórax pediátricas reducidas a 28×28 o
64×64 px; sus autores advierten que a esa resolución se pierde información necesaria para diagnosticar. El único
cambio que hace el cuaderno es pasar el brillo de 0-255 a 0-1. Citas:
- Yang J et al. *MedMNIST v2 – A large-scale lightweight benchmark for 2D and 3D biomedical image classification.*
  Scientific Data 10: 41 (2023). doi:10.1038/s41597-022-01721-8. Datos: Zenodo, doi:10.5281/zenodo.10519652 (v3.0).
- Kermany DS et al. *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell 172(5):
  1122-1131.e9 (2018). doi:10.1016/j.cell.2018.02.010. Datos originales: doi:10.17632/rscbjbr9sj.3 (CC BY 4.0).
- El código de `medmnist` es Apache-2.0. El espejo de Hugging Face (`albertvillanova/medmnist-v2`, revisión fijada)
  no es oficial: MedMNIST solo reconoce Zenodo como canal de distribución.

**Ultralytics YOLO: AGPL-3.0**, también los pesos `yolo26n.pt`. En llano, y sin que esto sea asesoramiento jurídico:
usarlo en clase para aprender está permitido. Quien distribuya software que lo incluya, o ofrezca por red una versión
modificada, tiene que dar el código fuente bajo AGPL-3.0. Ultralytics entiende que eso alcanza a toda la aplicación y a
los modelos entrenados con su código (🟡 es la interpretación del licenciante). Para uso comercial cerrado vende una
licencia Enterprise.

**supervision: MIT.** Solo se usa para descargar los vídeos de demo; su licencia no cubre los vídeos.

**Vídeos: licencia de su fuente.** Roboflow no publica el origen de sus vídeos de demo. El de «personas caminando»
coincide con un clip de Pexels (autor: Coverr; 🟡 inferido por metadatos). El del andén no tiene origen conocido y
muestra caras cerca de la cámara. Se usan solo para la demostración en clase, no se redistribuyen y no están en git. Un
clip propio de Pexels va con la licencia de Pexels (uso gratuito y sin atribución), que no garantiza que las personas
grabadas hayan dado su consentimiento.

**Privacidad.** Del NB2 solo salen recuentos (CSV y JSON), sin imágenes ni identificadores. Los vídeos anotados sí
muestran personas: el NB2 lleva `private_outputs` para que Colab no guarde las salidas (D38, 🟡 sin comprobar en Colab)
y la portada explica cómo borrarlas antes de compartir. Divulgación, no asesoramiento jurídico.

**Otras fuentes citadas en los cuadernos y el guion:** Zech JR et al., PLOS Medicine 15(11): e1002683 (2018),
doi:10.1371/journal.pmed.1002683 (aprendizaje por atajos); Selvaraju RR et al., ICCV 2017, doi:10.1109/ICCV.2017.74
(Grad-CAM); Wang ZJ et al., IEEE TVCG 27(2) (2021), doi:10.1109/TVCG.2020.3030418 (CNN Explainer, licencia MIT; la abre
solo el profesor compartiendo pantalla, D51).

**Licencia de este repositorio: AGPL-3.0** (archivo `LICENSE`, D50), compatible con Ultralytics, que es AGPL-3.0.
Los datos y vídeos no forman parte del repositorio y mantienen su propia licencia.

---

## Fuentes utilizadas

- Medidas: `verificacion/final/hechos_para_docs.md`, `verificacion/final/log.txt` y los informes
  `verificacion/informe_20260924-1119*.md` a `…-1125*.md`. Todo `verificacion/` está fuera de git (`.gitignore`). Los
  informes y las copias ejecutadas se regeneran con `scripts/verificar.py`; `hechos_para_docs.md`, `log.txt`,
  `resumen_medidas.txt` y las vistas del alumno se compilaron a mano el 24-sep y el script no los produce.
- Salidas reales de los cuadernos: `verificacion/final/*_vista_alumno.txt`.
- Tamaño de los vídeos anotados: `docs/notas_verificacion/g_construccion_nb2.md` (1,30 y 1,37 MB en la demo 1).
- Descarga de datos, espejo y MD5: `docs/notas_verificacion/a_medmnist.md` y `src/build_nb1.py`.
- Pesos y solutions: `docs/notas_verificacion/c_ultralytics.md`.
- Vídeos de demo y licencias: `docs/notas_verificacion/d_supervision_videos.md`.
- Colab (Python 3.13, badges, Drive, descargas, subida, vídeo, bloqueo de IP): `docs/notas_verificacion/e_colab.md`
  (googlecolab/backend-info, colabtools#6081, #6069, #5996, #3479, #1909).
- Citas, licencias y privacidad: `docs/notas_verificacion/f_fuentes_y_afirmaciones.md`.
- Entornos de verificación: `docs/notas_verificacion/g_construccion_nb1.md`, `g_construccion_nb2.md` y `scripts/verificar.py`.
- Decisiones citadas (D03-D05, D14, D16, D36, D38, D50, D51): `DECISIONS.md`.
- Contexto de la S7: `sesiones/S7/01-esqueleto.md` y `sesiones/S7/05-checklist.md`.

## Pendientes de validar

- 🟡 Todos los tiempos de Colab: proyección con factor 2,5-3,5, sin medir en Colab.
- [🔴 Pendiente de confirmar] Tiempo de la primera instalación en Colab (`medmnist`; `ultralytics` + `supervision` + `lap`).
- 🟡 Que la instalación no pida reiniciar la sesión con Python 3.13.
- 🟡 Rótulos de Colab en español: «Ejecutar de todos modos», «Archivo → Subir cuaderno», «Cambiar versión del entorno»,
  «Editar → Borrar todos los resultados».
- 🟡 Uso de GPU si el entorno la tiene: no probado.
- 🟡 Efecto de `private_outputs` en el NB2.
- 🟡 Origen del vídeo «personas caminando» (inferido por metadatos) y licencia del vídeo del andén (no consta).
- 🔴 Hoja común de resultados.
- ✅ `verificacion/final/hechos_para_docs.md` y `resumen_medidas.txt` se versionan (excepción en `.gitignore`) para
  que las cifras de referencia tengan fuente rastreable.
- 🟡 Rótulos de Google Sheets («Datos → Dividir texto en columnas»).
