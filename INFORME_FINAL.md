# Informe final · «Una CNN por dentro» + «Contar personas en vídeo»

Fecha: 24-sep-2026. Encargo: `BRIEF.md`. Decisiones: `DECISIONS.md`. Pruebas de APIs y fuentes: `docs/notas_verificacion/`.

## 1. Qué se ha construido

| Pieza | Qué es |
|---|---|
| `notebooks/01_laboratorio_cnn_radiografias.ipynb` | Laboratorio de arquitectura CNN con PneumoniaMNIST. 23 celdas (10 de código, todas en vista de formulario). Secciones 1.0-1.11 del brief: datos, diseño con 11 controles y ficha del modelo, entrenamiento con curvas en vivo y diagnóstico, métricas en test, qué ve cada capa, Grad-CAM con debate sobre atajos, registro exportable, cuatro retos, Plan B del profesor y cierre. |
| `notebooks/02_conteo_personas_cafeteria.ipynb` | Conteo de personas con YOLO26 nano preentrenado y las *solutions* `RegionCounter` y `ObjectCounter`. 21 celdas (9 de código). Secciones 2.0-2.9: aviso ético, dos demos y vídeo propio, detector y curva recuento/umbral, zonas en %, aforo por zonas, entradas y salidas, permanencia, CSV + JSON y webhook opcional a n8n. |
| `src/build_nb1.py`, `src/build_nb2.py` | Generadores con `nbformat`. Deterministas: dos ejecuciones dan el mismo SHA-1. Los `.ipynb` se guardan sin salidas. |
| `scripts/verificar.py` | Ejecución headless con nbclient. Simula el formulario reescribiendo las líneas `# @param`. Modos `rapido`, `defecto`, `referencias` e `imposible`; `--set` para cualquier control; `--hilos N` para imitar la CPU de Colab. Comprueba la estructura (`cellView`, `# @title`, sin salidas) y resume tiempos por celda, errores, stderr y exportaciones. |
| `docs/guion_docente_60min.md` | Guion minuto a minuto: hora completa del brief y versión de 40 min para la S7 (20:30-21:10). |
| `docs/hoja_alumno.md` | Una página para el alumno: pasos, retos, tabla de hipótesis y resultados, preguntas. |
| `README.md` | Cómo abrir en Colab, requisitos, tiempos, problemas frecuentes, regeneración y verificación, licencias. |
| `requirements-verificacion.txt` | Entorno local fijado (Python 3.12). |

Proceso: verificación de APIs en seis frentes (fase 1) → construcción con bucle de auto-verificación, auditoría contra la
especificación y corrección (fase 2) → dos revisiones adversariales independientes por cuaderno, una de «alumna no técnica» y otra
técnica, contrastadas por un escéptico y aplicadas (fase 3: 90 hallazgos, 2 refutados, 40 matizados) → medición final en serie y
documentación con revisión (fase 4). Un commit por fase.

## 2. Resultados de la verificación

### 2.1 Ejecuciones

Todas con **0 errores, 0 salidas por stderr y 0 problemas de estructura** (ejecución final en serie, sin otros procesos).

| Cuaderno | Modo | Entorno | Total (incluye arranque del kernel) |
|---|---|---|---|
| NB1 | rápido (2 épocas) | `.venv` · Python 3.12 · Keras 3.15.1 | 25,3 s |
| NB1 | por defecto | `.venv` | 26,3 s |
| NB1 | imposible (4 bloques a 28 px) | `.venv` | 18,2 s · aviso «se queda en 0×0» detectado, sin trazas |
| NB1 | referencias (Plan B) | `.venv` | 76,7 s |
| NB1 | por defecto y referencias con 2 hilos | `.venv` | 30,4 s y 145,1 s |
| NB1 | rápido y por defecto | `.venv-colab` · Python 3.13 · Keras 3.13.2 · numpy 2.1.3 · pandas 2.2.3 · matplotlib 3.10.0 · IPython 7.34 (versiones de Colab) | 24,7 s y 26,2 s · métricas idénticas |
| NB2 | rápido (3 s de vídeo) | `.venv` | 17,3 s |
| NB2 | por defecto (demo 1) | `.venv` | 26,2 s |
| NB2 | demo 2 (andén) | `.venv` | 32,6 s |
| NB2 | por defecto con 1 hilo, demo 1 y demo 2 | `.venv` | 31,1 s y 38,4 s |

Además, durante la construcción y las revisiones se probaron: celdas fuera de orden, re-ejecuciones en el mismo kernel, zona mal
escrita, línea con dos puntos iguales, vídeo propio fuera de Colab, webhook a un servidor local con 404 y con respuesta rota,
fallo simulado de pip, primera ejecución desde cero con descarga de pesos y vídeo. Ninguno deja una traza de Python.

### 2.2 Tiempos en CPU local

Máquina: Apple M3 Pro, 11 núcleos. «Limitado» imita la CPU de Colab gratuito: 2 hilos para TensorFlow y 1 hilo para el
detector (Ultralytics fija `torch` a `min(8, núcleos−1)`, que en Colab gratuito es 1).

| Qué | Todos los hilos | Limitado | Proyección en Colab gratuito 🟡 | Presupuesto del brief |
|---|---|---|---|---|
| NB1 · entrenamiento por defecto (8 épocas; celda 1.4 completa) | 4,0 s (5,5 s) | 6,7 s (8,2 s) | ~20-30 s | ≤ 90 s ✅ |
| NB1 · Plan B, 7 entrenamientos | 54 s | 121 s | ~5-7 min | — |
| NB2 · 2.3 + 2.5 + 2.6, demo 1 (10 s, salto 2) | 18,6 s | 24,7 s | ~60-90 s | ≤ 2 min ✅ |
| NB2 · lo mismo, demo 2 (origen 4K vertical) | 25,2 s | 31,3 s | ~80-110 s | ≤ 2 min, al límite |

🟡 La proyección aplica un factor de 2,5-3,5 sobre la medida limitada (un núcleo de M3 Pro frente a un Xeon de 2,2 GHz). **No se ha
medido en Colab.** Si la demo 2 pasa de 100 s, la celda 2.3 lo avisa antes y sugiere saltar 3 fotogramas.

### 2.3 Métricas de las configuraciones de referencia (NB1)

Plan B, semilla 42, 28 px, todos los hilos. Test de 624 radiografías (234 normales, 390 con neumonía).

| Configuración | Parámetros | Entrenamiento | Acierto test | Sensibilidad | Especificidad | AUC | Diagnóstico |
|---|---|---|---|---|---|---|---|
| Referencia tonta (siempre «Neumonía») | — | — | 62,5 % | 100 % | 0 % | 0,500 | — |
| Por defecto | 56.129 | 4,0 s | 87,7 % | 94,9 % | 75,6 % | 0,936 | razonable (aún mejoraba) |
| A · 1 bloque | 173.345 | 3,0 s | 84,1 % | 97,9 % | 61,1 % | 0,924 | razonable (aún mejoraba) |
| A · 3 bloques | 27.521 | 4,2 s | 86,2 % | 94,6 % | 72,2 % | 0,936 | razonable (aún mejoraba) |
| B · sin pooling | 1.184.577 | 7,7 s | 86,5 % | 97,9 % | 67,5 % | 0,927 | razonable |
| C · sin pooling, 20 épocas, dropout 0 | 1.184.577 | 15,0 s | 87,5 % | 98,2 % | 69,7 % | 0,939 | **sobreajuste** |
| C · sin pooling, 20 épocas, dropout 0,5 | 1.184.577 | 14,8 s | 88,1 % | 97,9 % | 71,8 % | 0,946 | razonable |
| D · compensar el desbalanceo | 56.129 | 4,1 s | 87,8 % | 93,8 % | 77,8 % | 0,931 | razonable (aún mejoraba) |

Con 2 hilos cambian los decimales (por ejemplo, por defecto: 86,9 % / 94,9 % / 73,5 %) y no las tendencias. Entre semillas, la
especificidad de la misma red varió de 58,5 % a 75,6 % durante la calibración: las diferencias de 1-3 puntos entre dos entrenamientos
sueltos pueden ser azar. Validación ≈ 95 % frente a test ≈ 87 % no es sobreajuste: el test es otro conjunto, con menos neumonías
(62,5 % frente a 74,2 %), y cae sobre todo la especificidad (88 % → 76 %).

### 2.4 Recuentos en los vídeos de demo (NB2)

Confianza 0,35, salto 2, 10 s, ByteTrack, fotogramas reducidos a 1280 px de lado largo.

| | Demo 1 · personas caminando (0-10 s) | Demo 2 · andén de metro (22-32 s) |
|---|---|---|
| Fotogramas · resolución de proceso | 125 · 1280×720 | 150 · 720×1280 |
| Zonas: máximo / media | Izquierda 14 / 11,0 · Derecha 13 / 9,5 | Izquierda 6 / 2,9 · Derecha 7 / 4,4 |
| Línea «auto» | horizontal al 50 %: 8 entradas / 8 salidas | diagonal `62,15; 22,100`: 9 entradas (se alejan del tren) / 3 salidas (van hacia él) |
| Permanencia media | 5,6 s / 5,7 s · 35 estancias, 30 recortadas por los bordes del tramo | 2,3 s / 2,1 s · 29 estancias, 8 recortadas |
| Vídeos H.264 anotados | 1280×720 a 12,5 fps, ~1,3-1,6 MB | 406×720 a 14,985 fps |

En el primer fotograma de la demo 1 el detector marca 17 personas con 0,35, donde a simple vista hay más de 40: el aforo se queda
corto, y el cuaderno lo dice.

## 3. Criterios de aceptación del brief (§9)

| Criterio | Estado |
|---|---|
| Ambos notebooks de principio a fin en local, por defecto y rápidos | ✅ §2.1 |
| NB1 ≤ 90 s y NB2 ≤ 2 min en CPU; tiempos medidos y reportados | ✅ en local (§2.2). 🟡 En Colab es una proyección: comprobar a mano |
| `cellView: form` en todas las celdas técnicas; configuración en formularios | ✅ comprobado por `verificar.py` en cada ejecución |
| Texto visible en español de España | ✅ revisión adversarial de «alumna no técnica» aplicada |
| NB1: diagrama, ficha, curvas en vivo, sensibilidad y especificidad, mapas de activación, Grad-CAM y tabla exportable | ✅ |
| Configuraciones imposibles detectadas con mensaje útil antes de construir | ✅ «Con 4 bloques, kernel 3×3 y pooling, la imagen de 28×28 se queda en 0×0 píxeles…» con soluciones que funcionan |
| NB2: detección, zonas en %, aforo, entradas/salidas y exportación; el webhook no envía nada si está vacío | ✅ «No se envía nada: el campo webhook_n8n está vacío.» |
| Vídeo anotado inline en H.264 | ✅ codificado y comprobado (`libx264`, `yuv420p`, `+faststart`). 🟡 Reproducción en Colab: comprobar a mano |
| Avisos de licencia, uso no clínico y privacidad | ✅ portadas de ambos cuadernos |
| Guion con resultados de referencia medidos | ✅ tablas de §2.3 y §2.4 |
| Lista de comprobaciones manuales en Colab | ✅ §5 |
| Nada de datos, pesos ni vídeos en git | ✅ `git ls-files` solo lista texto, código y notebooks sin salidas |

## 4. Limitaciones conocidas

- **Nada se ha ejecutado en Colab real.** Los tiempos de Colab son proyecciones, y la prueba con las versiones de Colab (Python 3.13)
  solo cubre el NB1: en macOS no se puede reproducir el NB2 con numpy 2.1.3 (ultralytics excluye numpy 2.0-2.3.4 en macOS).
- **El brief pedía emular Python 3.11-3.12**, pero Colab usa Python 3.13 desde el 25-ago-2026. Se verificó con 3.12 (brief) y con 3.13 (NB1).
- **Ruido entre semillas** comparable a varios efectos de los retos. Los retos B y C tienen efectos claros (parámetros, tiempo, diagnóstico);
  el A y el D, más pequeños. Se presentan como tendencias.
- **Reto C modificado:** con la red por defecto no hay sobreajuste en 20 épocas, así que el reto quita el pooling (D20).
- **El detector pierde personas** pequeñas o agrupadas a 640 px: los aforos de la demo 1 se quedan cortos. Con confianza < 0,25
  ByteTrack no abre seguimientos nuevos (hay aviso).
- **El sentido de la línea** lo decide Ultralytics por geometría: en el andén, «entrada» es alejarse del tren. El cuaderno lo rotula así.
- **Permanencia** con 10 s de vídeo: la mayoría de estancias quedan cortadas por los bordes del tramo (30 de 35 en la demo 1). El cuaderno lo avisa.
- **Licencias de los vídeos de demo:** Roboflow no publica su origen. El de personas caminando coincide con un clip de Pexels (Coverr);
  el del andén, sin origen conocido, y muestra caras cerca de la cámara.
- **Grad-CAM a 28 px** es grueso (mapas de 11×11 con la red por defecto; 3×3 o menos con redes profundas, con aviso). Señala zonas, no causas.
- **Dependencias de terceros en clase:** Zenodo (con espejo solo para 28 px), GitHub (pesos de YOLO), media.roboflow.com (vídeos).
- **Rótulos del gráfico del registro con n.º** en lugar de nombre (D23): desviación del brief pendiente de aceptar.

## 5. Lo que el profesor debe comprobar a mano en Colab

Antes de clase, con una cuenta de alumno o una ventana privada, abriendo los cuadernos desde el enlace que se pegará en el chat:

1. **Vista de formulario:** que el código queda oculto (`cellView`) y se ven títulos y controles; tildes en títulos y desplegables;
   que los desplegables numéricos devuelven números; que el deslizador de dropout muestra 0.2.
2. **Aviso de GitHub:** si aparece «este cuaderno no lo ha creado Google», que el botón se llama como dicen los textos («Ejecutar de todos modos»).
3. **Instalación:** que 1.1 y 2.1 terminan sin pedir reiniciar la sesión (Python 3.13) y cuánto tardan la primera vez.
4. **Tiempos reales:** 1.4 con la configuración por defecto (≤ 90 s) y 2.3 + 2.5 + 2.6 con la demo 1 (≤ 2 min); también la demo 2 y el Plan B.
5. **Curvas en vivo:** que la figura de 1.4 se actualiza en el mismo sitio en cada época y que el desplegable «Cifras de este entrenamiento» se abre.
6. **Descargas:** CSV del registro en 1.8; CSV y JSON en 2.8 (el navegador puede pedir permiso para descargas múltiples).
7. **Subida de vídeo:** «Subir mi vídeo» en 2.2 con un clip corto (el widget de Colab va lento con archivos grandes; Chrome recomendado).
8. **Reproducción de vídeo:** que los vídeos de 2.5 y 2.6 se reproducen en Chrome y Safari.
9. **Privacidad:** que `private_outputs` hace que Colab no guarde las salidas del NB2, y el nombre del menú «Editar → Borrar todos los resultados».
10. **CNN Explainer:** repetir el recorrido del guion una vez en el navegador de clase.
11. **Enlaces:** sustituir `<USUARIO>/<REPO>` en los badges o quitar el badge antes de subir al aula virtual.

## 6. Pendientes para el profesor

- 🔴 Crear la hoja común de resultados y poner su enlace en el chat (cabecera: la primera línea que da la celda 1.8).
- 🔴 Subir los cuadernos a GitHub o Drive y cambiar el enlace de los badges.
- 🔴 Decidir la licencia del repositorio si se publica (D50: si se publica, AGPL-3.0 al menos para el NB2).
- 🔴 Aceptar o no la desviación de los rótulos con n.º en el gráfico del registro (D23).
- 🟡 Ejecutar el Plan B una vez en Colab antes de clase para tener la tabla de referencia a mano.
- 🟡 Ajustar el bloque en la S7: `sesiones/S7/01-esqueleto.md` reserva 40 min (bloques 8 y 9) y queda pendiente de este guion; la versión de 40 min está en el guion.

## Fuentes utilizadas

- Mediciones: `verificacion/final/resumen_medidas.txt`, `verificacion/informe_20260924-111900.md` a `…-112517.md` (no versionados; se regeneran con `scripts/verificar.py`).
- APIs y versiones: `docs/notas_verificacion/a_medmnist.md`, `b_keras_gradcam.md`, `c_ultralytics.md`, `d_supervision_videos.md`, `e_colab.md`.
- Afirmaciones y licencias: `docs/notas_verificacion/f_fuentes_y_afirmaciones.md` (MedMNIST v2, Kermany et al. 2018, Zech et al. 2018, Grad-CAM, CNN Explainer, YOLO26, AGPL-3.0, RGPD/CEPD/AEPD).
- Construcción, calibración y correcciones: `docs/notas_verificacion/g_construccion_nb1.md`, `g_construccion_nb2.md`, `h_ensayo_cnn_explainer.md`.
- Versiones de Colab: googlecolab/backend-info (pip-freeze y os-info, 23-sep-2026) y colabtools#6081.

## Pendientes de validar

- 🟡 Tiempos en la CPU de Colab gratuito (proyección con factor 2,5-3,5).
- 🟡 CPU y RAM de Colab gratuito (2 vCPU, ~12,7 GB): fuentes comunitarias, Google no las publica.
- 🟡 Rótulos de Colab en español («Ejecutar de todos modos», «Editar → Borrar todos los resultados») y efecto de `colab.private_outputs`.
- 🟡 Edades de los pacientes y hospital de origen de Kermany et al. (2018): confirmados solo en fuentes secundarias; los cuadernos dicen «según los autores».
- 🟡 Origen de los vídeos de demo de supervision (el de personas caminando, inferido por metadatos).
