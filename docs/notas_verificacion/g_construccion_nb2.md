# Frente g) Construcción del Notebook 2 (conteo de personas en vídeo)

Fecha: 24-sep-2026. Entorno: `.venv` (Python 3.12.7, macOS arm64, M3 Pro con 11 núcleos), ultralytics 8.4.161, supervision 0.30.5, torch 2.14.0, OpenCV 5.0.0, numpy 2.5.3, pandas 3.0.6, ffmpeg 7.1 (imageio-ffmpeg).
Otro agente construía el Notebook 1 a la vez: carga media de 9 a 15 sobre 11 núcleos durante todas las medidas. **Los tiempos tienen ruido**; los recuentos no (son deterministas).

Entregables:
- `src/build_nb2.py`: generador con nbformat. Dos ejecuciones seguidas dan el mismo SHA-1 (`df993e08f2ad1ff54c9533ca54db7ffe63bd8b4b` tras las correcciones de la auditoría).
- `notebooks/02_conteo_personas_cafeteria.ipynb`: 21 celdas (12 markdown, 9 de código), sin salidas, ids `nb2-00` … `nb2-20`, todas las de código con `cellView: form` y `# @title … { display-mode: "form" }` en la primera línea.
- Material de trabajo en `verificacion/build_nb2/` (ignorado por git): `inspeccionar.py` (vuelca salidas y extrae PNG y MP4), `fotograma.py`, `linea_check.py`, `casos.py`, `repetir.py`, las copias ejecutadas en `final/` y las imágenes revisadas.

---

## 1. Estructura del cuaderno

| Celda | Qué hace | Formulario |
|---|---|---|
| 2.0 (md) | Portada, aviso de privacidad (texto de la nota f), licencias (AGPL, vídeos, telemetría desactivada), índice, badge con `<USUARIO>/<REPO>` | — |
| 2.1 | Instala solo lo que falta, importa, silencia, descarga `pesos/yolo26n.pt`, tabla de versiones y hardware. Define todos los helpers | — |
| 2.2 | Descarga o sube el vídeo, tabla de metadatos, primer fotograma. Nota para el profesor (md) debajo | `fuente_video`, `segundos_a_procesar`, `saltar_fotogramas` |
| 2.3 | Una predicción con conf 0,05 → cajas por encima del umbral, curva recuento-umbral, compromiso FP/FN con las cifras del fotograma y estimación de tiempo de 2.5 + 2.6 | `confianza_minima` |
| 2.4 | Parseo de zonas y línea, vista previa con rejilla en % y flecha de «entrada» | `nombre_zona_1`, `zona_1`, `nombre_zona_2`, `zona_2`, `linea_puerta` («auto»), `invertir_sentido` |
| 2.5 | `RegionCounter`: vídeo inline, serie temporal, tabla máximo/media, tiempos. Guarda fotogramas por (zona, ID) para 2.7 | — |
| 2.6 | `ObjectCounter`: vídeo inline con rótulos propios, entradas/salidas | — |
| 2.7 | Permanencia por zona (≥ 1 s), histograma, tabla y aviso de estancias recortadas y cambios de ID | — |
| 2.8 | CSV + JSON en `salidas/nb2/`, descarga en Colab | — |
| 2.8 (opcional) | Muestra el JSON que recibiría n8n y hace POST si hay URL. Debajo, flujo de alerta descrito en md | `webhook_n8n` |
| 2.9 (md) | Límites, POC ≠ producción, cuatro sectores, tres preguntas de privacidad | — |

Los nombres de parámetros son exactamente los de la especificación, más `invertir_sentido` (decisión de la fase 1). `scripts/verificar.py` los encuentra todos.

---

## 2. Decisiones (con su motivo)

1. **Errores sin traza: `Parar` + `with celda():`.** Cada celda (salvo la 2.1) es `formulario → if "celda" not in globals(): print(...) else: with celda(): cuerpo`. Un error previsto lanza `Parar("texto en castellano")`; el gestor de contexto lo pinta como recuadro rojo y la celda termina limpia, sin salida `error` (así `verificar.py` no lo cuenta como fallo y en Colab no aparece ninguna traza). Los errores imprevistos sí dejan traza a propósito, para que la verificación los detecte. El generador indenta el cuerpo solo (`paso()`), así el código de cada celda se escribe a columna 0 en el script.
2. **Todos los helpers en la 2.1** (mensajes, `es()` para números en formato español, `tabla`, `mostrar`, lectura en streaming, `pasada`, conversión a H.264). La 2.1 es obligatoria de todas formas y la vista de formulario oculta el código.
3. **`requiere()` y dependencias explícitas.** Cada celda comprueba las claves de `ESTADO` que necesita y el mensaje nombra la celda exacta («Falta el paso anterior: ejecuta sin errores la celda 2.4 · Zonas y línea de puerta…»). Cada celda que produce una clave llama a `reiniciar(clave)` antes de validar: borra su propia clave y las que dependen de ella (`DEPENDEN`, en la 2.1). Ver «Correcciones tras auditoría». `ESTADO` se crea solo si no existe.
4. **`mostrar(fig)` guarda la figura en PNG y la muestra con `Image`.** `verificar.py` fija `MPLBACKEND=Agg`, y con ese backend `display(fig)` no garantiza una imagen. Así el resultado no depende del backend y nunca sale duplicada.
5. **Etiqueta de caja «ID 12» en lugar de «12 persona».** El formato de Ultralytics es `"{id} {clase}"`, y «12 persona» se lee como «12 personas». Se sobrescribe `sol.adjust_box_label` en la instancia (una línea). `sol.names[0] = "persona"` se mantiene, como pide la especificación.
6. **Rótulos propios dentro del vídeo** (`rotulo()`: texto blanco sobre recuadro oscuro, escala 1,1, ASCII). Zonas: «Izquierda: 12» en la esquina de cada zona. Línea: «Entradas: N» / «Salidas: M» arriba a la izquierda y una flecha naranja con «entrada». `display_analytics` de Ultralytics con `line_width=2` da un texto de escala 0,67, que en proyector se queda pequeño (tras la auditoría, `line_width=3`: escala 1,0 para las etiquetas «ID n» y el recuento central de cada zona). Los colores de las zonas se cambian a Okabe-Ito (`counting_regions[i]["region_color"]`) para que coincidan con la vista previa.
7. **Cajas de 2.3 dibujadas con matplotlib**, no con `Results.plot()`: el texto queda nítido a cualquier tamaño, la confianza sale con coma decimal («0,64») y se evita cualquier descarga de fuentes de Ultralytics.
8. **Vídeos anotados en `datos/tmp/`, no en `salidas/nb2/`.** En `salidas/` solo hay agregados (CSV y JSON), coherente con «del cuaderno solo salen agregados». El andén de metro enseña caras. Tamaños: 1,30 y 1,37 MB (demo 1, 10 s) y 0,70 y 0,81 MB (demo 2); se incrustan en base64 con `style="max-width:100%"` (854 px de ancho en horizontal, 360 en vertical).
9. **Hilos en la verificación local.** Ultralytics hace `torch.set_num_threads(min(8, núcleos−1))` en la primera inferencia de cada predictor, así que `--hilos 1` de `verificar.py` no tenía efecto sobre torch. La 2.1 lee `OMP_NUM_THREADS` **antes** de importar ultralytics (que lo fija a 1 si no existe) y, solo fuera de Colab, sustituye `ultralytics.utils.torch_utils.NUM_THREADS`. En Colab no toca nada: se queda el comportamiento por defecto (1 hilo con 2 vCPU). La tabla de la 2.1 muestra «Hilos de CPU para el detector».
10. **Estimación de tiempo en 2.3.** `2 × (n_fotogramas × (t_inferencia × 1,1 + t_lectura + 0,01) + 2,5)` s, con `t_inferencia` = mínimo de 3 predicciones tras una de calentamiento y `t_lectura` medido sobre 6 fotogramas del tramo **después** del primero (el primero incluye el salto al segundo 22 del vídeo 4K; con él, la estimación del andén salía 67 s frente a 26 s reales). Calibración en la sección 3: en el tramo por defecto sobreestima entre un 12 y un 31 % (el lado seguro); en el modo rápido acierta (11 s frente a 11,2 s). Si pasa de 100 s, aviso con la acción concreta (salto 3 o menos segundos).
11. **Aviso de umbral < 0,25 en 2.3.** ByteTrack no abre IDs nuevos por debajo de `new_track_thresh = 0,25` (nota c), así que en 2.5/2.6 se contaría menos gente de la que se ve en 2.3.
12. **Parseo de zonas tolerante**: admite paréntesis (`(0,0); (50,0)…`), espacios y puntos y coma finales; decimales con punto. Errores con el texto problemático, el trozo concreto y un ejemplo correcto. Validaciones: 0-100 %, ≥ 3 puntos por zona, polígono válido y con superficie (shapely), nombres distintos, línea de exactamente 2 puntos distintos. Nombre vacío → «Zona 1».
13. **Sentido de la línea calculado en píxeles**, con el mismo criterio que `ObjectCounter` (|dx| < |dy| → vertical, entrada hacia la derecha; si no, horizontal, entrada hacia abajo). En % daría otro resultado en vídeos verticales para diagonales. `invertir_sentido` intercambia etiquetas, flecha de la vista previa y flecha del vídeo.
14. **Descargas.** Demo: si el archivo existe, no se descarga ni se recalcula el md5. Si no, `download_assets` (log de supervision a WARNING **después** de importarlo, porque supervision configura su logger al importarse; la barra de tqdm se ve en Colab como widget y fuera se descarta) y, si falla, descarga directa de `media.roboflow.com` a `.parcial` + renombrado. Pesos: `YOLO("pesos/yolo26n.pt")` con stdout/stderr redirigidos (la barra de Ultralytics va a stdout) y mensaje propio «Descargando el detector (5 MB)…».
15. **«Subir mi vídeo».** Ruta fija `datos/videos/mi_video.mp4`. Si ya existe, se reutiliza y se explica cómo cambiarlo (así reejecutar 2.2 para cambiar los segundos no obliga a subirlo otra vez). En Colab, `files.upload()`; si se cancela, mensaje en castellano. Fuera de Colab, error con la ruta absoluta donde dejarlo.
16. **Webhook en una celda aparte**, con `urllib` de la biblioteca estándar. Vacío → «No se envía nada». Sin `http(s)://` → error. 404 → explica la URL de prueba (`webhook-test`) frente a la de producción. Sin conexión → mensaje llano. Siempre enseña el JSON que recibiría n8n y recuerda que llega dentro de `body`.
17. **JSON (`salidas/nb2/resumen.json`)**: los campos del contrato más `inicio_s`. `resolucion_proceso` es texto (`"1280x720"`). `entradas`/`salidas` quedan en `null` si no se ejecutó 2.6 (con aviso) y `permanencia_media_s` en `null` si no se ejecutó 2.7. `nota`: «Solo agregados: sin imágenes ni identificadores.»
18. **NMS no se menciona en 2.3.** La nota f propone «da las cajas sin NMS», pero la nota c comprobó que en 8.4.161 la predicción usa por defecto la cabeza con NMS. No aporta nada al alumno, así que se omite en lugar de arriesgar una afirmación incorrecta.
19. **Ruido técnico silenciado de forma específica**: `YOLO_VERBOSE`/`YOLO_AUTOINSTALL` antes del import, `LOGGER.setLevel(ERROR)`, `events.enabled = False`, `warnings.filterwarnings("ignore", message="IProgress not found")` (tqdm sin ipywidgets fuera de Colab) y el logger de supervision. Resultado: **cero líneas en stderr y ninguna en stdout** en todas las ejecuciones, incluida la primera con descargas.
20. **Línea «auto» de la demo 1 al 50 %** (`0,50; 100,50`), como fijan la especificación y `DECISIONS.md` D32: 8 entradas / 8 salidas. La primera versión la dejó al 65 % por error; corregido tras la auditoría.

---

## 3. Mediciones

Tramo por defecto (10 s, salto 2, conf 0,35, 640 px de entrada al detector). Tiempos de celda de `verificar.py`; «bucle» = solo el recorrido de fotogramas, sin crear la solución ni convertir el vídeo.

| Ejecución | Hilos | 2.3 | 2.5 | 2.6 | 2.3 + 2.5 + 2.6 | Bucle por pasada | Estimación de 2.3 para 2.5 + 2.6 | Real 2.5 + 2.6 |
|---|---|---|---|---|---|---|---|---|
| Demo 1 (125 fotogramas por pasada) | todos (8) | 2,2 s | 9,0 s | 8,5 s | 19,7 s | 7,7 s (16,2 fps) | 23 s | 17,5 s |
| Demo 1 | **1** | 2,3 s | 11,6 s | 10,8 s | **24,7 s** | 10,2 s (12,3 fps) | 27 s | 22,4 s |
| Demo 2 (150 fotogramas, origen 4K vertical) | todos (8) | 2,7 s | 11,8 s | 11,4 s | 25,9 s | 10,7-10,9 s (14 fps) | 26 s | 23,2 s |
| Demo 2 | **1** | 3,8 s | 14,9 s | 14,3 s | **33,0 s** | 13,8 s (10,9 fps) | 35 s | 29,2 s |
| Rápido (3 s, demo 1) | todos | 4,0 s | 5,9 s | 5,3 s | 15,2 s | 4,4-4,8 s | 11 s | 11,2 s |

- Una inferencia (mínimo de 3): 36-44 ms con 8 hilos y 57-64 ms con 1 hilo.
- Ejecución completa del cuaderno (con arranque del kernel): 28,5 s por defecto, 31,2 s con 1 hilo, 33,0 s y 40,8 s con la demo 2. La 2.1 tarda 3-4 s con todo instalado y unos 10 s la primera vez (descarga de pesos). La descarga del andén (128 MB) tardó unos 14 s en la 2.2.
- **Proyección a Colab gratuito 🟡** (factor de 2,5-3,5× por hilo de la nota c, no medido): demo 1, 2.3 + 2.5 + 2.6 ≈ 60-85 s, dentro de los 2 minutos del BRIEF. Demo 2 ≈ 80-115 s, más la descodificación 4K, que en 2 vCPU pesa más que aquí; puede rozar el umbral de 100 s y entonces la 2.3 avisará de subir el salto.
- Reejecutar 2.1, 2.5 y 2.6 en el mismo kernel da exactamente los mismos resultados (`repetir.py`).

### Recuentos (idénticos con 1 y 8 hilos)

| | Demo 1 (0-10 s) | Demo 2 (22-32 s) |
|---|---|---|
| Resolución de proceso | 1280×720 | 720×1280 (vídeo inline 406×720) |
| Personas en el primer fotograma (conf 0,35 / 0,10 / 0,70) | 17 / 39 / 4 | 9 / 16 / 2 |
| Zonas Izquierda / Derecha: máximo | 14 / 13 | 6 / 7 |
| Zonas: media | 11,0 / 9,5 | 2,9 / 4,4 |
| Línea «auto» | `0,50; 100,50` → **8 entradas / 8 salidas** | `62,15; 22,100` → **9 entradas / 3 salidas** |
| Permanencia (IDs ≥ 1 s; mediana / media / máx., s) | Izq. 19 IDs, 5,7 / 5,6 / 9,8 · Der. 16 IDs, 5,2 / 5,7 / 10,0 | Izq. 11 IDs, 1,3 / 2,3 / 6,5 · Der. 18 IDs, 1,7 / 2,1 / 7,6 |
| Estancias recortadas por los extremos del tramo (ID dentro de esa zona en el primer o el último fotograma) | 30 | 8 |

Las zonas de la demo 1 coinciden exactamente con la referencia de la fase 1 (máx. 14/13, media 11,0/9,5).

---

## 4. Hallazgo: las referencias de línea de la nota d no son válidas

La especificación daba como referencia «línea a 65 % → 4 entradas / 6 salidas» (nota d, tabla de zonas y puerta). Mi implementación da 11/6. Lo comprobé con `verificacion/build_nb2/linea_check.py`, que cuenta con `ObjectCounter` y, además, de forma independiente con el primer cruce de cada trayectoria:

| Tracker | Línea 50 % | Línea 65 % |
|---|---|---|
| ByteTrack (ObjectCounter / recuento independiente) | 8/8 / 8/8 | 11/6 / 11/6 |
| BoT-SORT | 8/8 / 8/8 | 11/6 / 11/6 |

**Causa:** `verificacion/d/zonas.py` pasa la misma lista de fotogramas primero por `RegionCounter` y luego por `ObjectCounter`, y las solutions dibujan **sobre** el fotograma que reciben (nota c, punto 7). El conteo de línea de la nota d se hizo sobre fotogramas ya anotados. Reproduje el efecto: con fotogramas reutilizados salen recuentos erráticos (0/0, 0/1, 2/4). Consecuencias:
- «Línea a 50 % → 0/2, evitar» (nota d) es falso: a 50 % da 8/8, igual que midió la nota c con fotogramas limpios.
- El 7/3 del andén con `62,15; 22,100` tampoco es comparable. Con fotogramas limpios, 22-32 s: 9/3.
- **Para el coordinador:** si el guion o el informe citan 4/6 o 7/3, hay que sustituirlos por 8/8 (demo 1, línea al 50 %, la del cuaderno) y 9/3 (demo 2, 22-32 s). El notebook lee cada fotograma del archivo en cada pasada, así que no tiene este problema.
- Corrección posterior: esta sección comparaba con la referencia antigua (4/6) y no con la del coordinador (10/6 al 65 %). El 11/6 frente a 10/6 es un píxel: el cuaderno pone la línea en `round(0,65·719) = 467` y la referencia, según la auditoría, en `int(0,65·720) = 468`; medido, 467 → 11/6 y 468 → 10/6. La línea al 65 % está sobre un borde sensible; la del 50 % da 8/8 en 359, 360 y 361 px (`verificacion/fix_nb2/linea_pixel.py`).

---

## 5. Verificación hecha

| Prueba | Resultado |
|---|---|
| `verificar.py --nb 2 --modos rapido` | ✅ sin errores ni stderr (38 fotogramas; 2/3 entradas/salidas) |
| `--modos defecto` | ✅ (tabla de la sección 3) |
| `--modos defecto --set 'fuente_video="Demo 2: andén de metro (interior)"'` | ✅ |
| `--modos rapido --set 'zona_1="10,10; 20"'` | ✅ «No entiendo la zona 1 («Izquierda»): «10,10; 20». El punto «20» debe tener dos números, x,y… Ejemplo correcto: 0,0; 50,0; 50,100; 0,100». Las celdas siguientes muestran «Falta el paso anterior…» sin traza |
| `--hilos 1` (demo 1 y demo 2) | ✅ «Hilos de CPU para el detector: 1» |
| Primera ejecución desde cero (sin `pesos/` ni el vídeo de la demo 1) + `invertir_sentido=True` | ✅ descargas silenciosas; 3/2 (frente a 2/3 sin invertir); flecha hacia arriba en la vista previa y en el vídeo |
| `casos.py`: cuaderno sin la 2.1 | ✅ las 8 celdas dicen «Primero ejecuta la celda 2.1…» |
| `casos.py`: «Subir mi vídeo» fuera de Colab | ✅ error con la ruta absoluta de `mi_video.mp4` |
| `casos.py`: zonas fuera de 0-100, 2 puntos, lados cruzados, nombres iguales, línea de 3 puntos, «abc», «12,5,30», paréntesis y nombre vacío | ✅ mensaje específico en cada caso; el último pasa como «Zona 1» |
| `casos.py`: conf 0,20 + 30 s | ✅ aviso de recorte a 13,6 s y aviso de umbral < 0,25 |
| `casos.py`: webhook `ftp://…`, puerto cerrado, servidor local que responde 404 y 200 | ✅ error de URL, «No he podido conectar…», explicación del 404 y «Enviado… código 200» |
| Reejecución de 2.1, 2.5 y 2.6 en el mismo kernel | ✅ mismos resultados |
| Estructura (`cellView`, `# @title`, sin salidas) | ✅ `verificar.py` no informa de problemas |
| Generador determinista | ✅ mismo SHA-1 en dos ejecuciones |
| Compatibilidad | Todas las celdas compilan; ninguna f-string reutiliza la comilla que la delimita (válido en Python < 3.12); nada exclusivo de pandas 3, IPython > 7.34 ni matplotlib 3.11 |

**Imágenes revisadas** (extraídas a `verificacion/build_nb2/`): primer fotograma, detección con confianzas y curva recuento-umbral, vista previa de zonas (horizontal, vertical e invertida), serie temporal de ambas demos, histograma de permanencia y fotogramas de los cuatro vídeos anotados. Todo el texto en castellano, legible a tamaño de proyector; tildes correctas en las figuras (matplotlib) y ASCII dentro del vídeo («Izquierda: 12», «Entradas: 11», «entrada»). Vídeos releídos con OpenCV: `h264`, 12,5 fps (demo 1) y 14,98 fps (demo 2), la misma duración que el tramo real.

---

## 6. Sin resolver o que se debe comprobar a mano en Colab

- 🟡 **Tiempo real en Colab CPU**: solo proyectado. La demo 2 puede pasar de 100 s; la 2.3 lo avisa con su estimación.
- 🟡 **Instalación en Colab (Python 3.13)**: `pip install -q ultralytics==8.4.161 supervision==0.30.5 lap shapely` con restricciones de las versiones instaladas. No se puede probar aquí: el `.venv` no tiene pip y todo está instalado. `.venv-colab` no sirve para NB2, porque ultralytics excluye numpy 2.0-2.3.4 en macOS (nota e, 6.2).
- 🟡 **Subida y descarga de archivos**: `files.upload()` escribe «Saving … to …» en inglés (texto de Colab, no se puede cambiar). `files.download()` de dos archivos seguidos puede pedir permiso al navegador para descargas múltiples.
- 🟡 **Reproducción del vídeo inline** y vista de formulario: controles, tildes en el desplegable de `fuente_video` y `# @markdown` entre parámetros.
- 🟡 **Barra de descarga de supervision en Colab**: debería verse como widget (tqdm.auto con ipywidgets). No verificado.
- Limitaciones aceptadas: solo se admite `mi_video.mp4` como nombre del vídeo propio; un nombre de zona con `$` activaría el modo fórmula de matplotlib en la vista previa; el md5 de una demo ya descargada no se revisa (si el archivo está dañado, el mensaje de la 2.2 dice que se borre y se reejecute).
- 🔴 **Clip de cafetería de Pexels**: el ángulo de los clips sugeridos no se ha comprobado a ojo (nota d). El profesor debe probarlo antes de clase.

## Fuentes utilizadas
- `BRIEF.md`, especificaciones `comun.md` y `nb2.md`.
- Notas `c_ultralytics.md`, `d_supervision_videos.md`, `e_colab.md` y `f_fuentes_y_afirmaciones.md` (los textos de privacidad, AGPL, licencia de los vídeos y descripción de YOLO26 salen de f y d).
- Fuente instalado de ultralytics 8.4.161: `solutions/{solutions,region_counter,object_counter}.py`, `utils/{torch_utils,tqdm,events}.py`, `__init__.py`.
- Ejecuciones propias: `verificacion/build_nb2/final/*.ipynb` y los informes `verificacion/informe_*.md`.

## Pendientes de validar
- Las comprobaciones manuales en Colab de la sección 6.
- Corregir en la documentación común cualquier cita de 4/6 o 7/3 (sección 4).

---

## Correcciones tras auditoría (24-sep-2026)

Solo se tocó `src/build_nb2.py`; el `.ipynb` se regeneró (determinista, SHA-1 `df993e08…`). Material en `verificacion/fix_nb2/`.

| # | Hallazgo | Cambio |
|---|---|---|
| 1 (mayor) | La línea «auto» de la demo 1 estaba al 65 % (11/6), no al 50 % de la especificación y D32 | `DEMOS[...]["puerta"]` y `EJEMPLO_LINEA` → `0,50; 100,50`. Da **8/8**, como D32. Notas corregidas (§1: 9 celdas de código, no 10; §2.20; §3; §4 con el píxel 467/468) |
| 2 (mayor) | El aviso de la demo 2 decía «el cuaderno no guarda ni exporta fotogramas»: falso, los vídeos anotados quedan en `datos/tmp/` y en base64 dentro del cuaderno | Aviso de 2.2: el CSV y el JSON solo llevan números; los vídeos anotados muestran las caras, quedan en `datos/tmp/` y dentro del cuaderno; bórralas antes de guardarlo o compartirlo. En 2.0, la viñeta de agregados dice ahora «los archivos que exporta (CSV y JSON)» y otra viñeta explica dónde quedan los vídeos |
| 3 | Una celda que fallaba al reejecutarse dejaba su resultado anterior en `ESTADO`; reejecutar 2.3 no invalidaba 2.5/2.6 | `reiniciar(clave)` en la 2.1 con un mapa `DEPENDEN`. 2.2, 2.3, 2.4, 2.5, 2.6, 2.7 y 2.8 la llaman justo después de `requiere()`: borran su clave y las que dependen de ella. 2.3 invalida aforo, línea, permanencia y resumen, así que el JSON ya no puede mezclar umbrales |
| 4 | Etiquetas de confianza de 9 pt en 2.3; «ID n» y recuento central diminutos en los vídeos | `fontsize=12` en 2.3; `line_width=3` en las soluciones (etiquetas a escala 1,0, unos 15 px a 854 de ancho) |
| 5 | Estancias «recortadas» contaban cualquier ID visible en el primer/último fotograma, aunque estuviera en la otra zona | Se guarda el conjunto (zona, ID) dentro de cada zona por fotograma; recortada = ese par está en el primer o el último. Demo 2: 10 → **8**; demo 1 sigue en 30. Texto: «ya estaban en curso al empezar el tramo o seguían al terminarlo» |
| 6 | «Cuanto más agregado, más seguro.» (moraleja); «el detector pierde a algunas» como certeza | Sustituida por la frase de la nota f: «Un agregado solo deja de ser dato personal si de verdad no permite señalar a nadie.» 2.9: «suele perder a algunas y el recuento se queda corto» |
| 7 | Línea con dos puntos iguales: «necesita 2 puntos distintos y tiene 2» | Dos comprobaciones: número de puntos y, ya en píxeles, que no caigan en el mismo sitio («Los dos puntos de la línea de puerta («50,50; 50,50») caen en el mismo sitio: sepáralos…») |
| extra | Con la línea al 50 %, los nombres de las zonas de la vista previa quedaban encima de la línea | El nombre se coloca en el polígono menos una franja de 8 puntos alrededor de la línea (`difference(...buffer(8))`) |

**Verificación (todas ✅, sin salidas `stream` ni `error` en ninguna celda):**

| Ejecución | Resultado | 2.3 + 2.5 + 2.6 |
|---|---|---|
| `--modos rapido` | 38 fotogramas; zonas máx. 13/11; 2/3 | 16,8 s |
| `--modos defecto` | zonas máx. 14/13, media 11,0/9,5; **8/8**; 30 estancias recortadas | 19,5 s |
| `--modos defecto --set 'fuente_video="Demo 2: andén de metro (interior)"'` | 720×1280; zonas 6/7, 2,9/4,4; **9/3**; 8 recortadas | 26,0 s |
| `--modos rapido --set 'zona_1="10,10; 20"'` | «No entiendo la zona 1 («Izquierda»)…» y «Falta el paso anterior…» en las siguientes, sin traza | — |
| `--set 'linea_puerta="50,50; 50,50"'` | mensaje nuevo del hallazgo 7, sin traza | — |
| `--modos defecto --hilos 1` | 8/8; una detección 57 ms; estimación 27 s | **24,8 s** |
| `--modos defecto --hilos 1` demo 2 | 9/3; estimación 35 s | **32,8 s** |
| `reejecucion.py` (mismo kernel) | A: 2.3(0,35) → 2.5 → 2.3(0,50) → 2.6 → 2.8 da «Falta… 2.5»; tras 2.5, JSON con 0,5 y 2/3. B: 2.4 con zona mala → 2.5 da «Falta… 2.4». C: 2.2 fallida («Subir mi vídeo» fuera de Colab) → 2.3 y 2.4 dan «Falta… 2.2» | — |

Carga media del equipo de 6 a 8 durante las medidas: tiempos con ruido. Imágenes revisadas: detección de 2.3 (etiquetas a 12 pt), vistas previas de zonas de ambas demos, fotogramas de los cuatro vídeos anotados a su ancho de pantalla (854 y 360 px) e histograma de permanencia de la demo 2.

**Queda:** en los grupos apretados de la demo 1 dos etiquetas de confianza de 2.3 se siguen tapando (con letra mayor, algo más que antes); el recuento y la curva no dependen de ellas. En la demo 2 (vídeo vertical mostrado a 360 px) las etiquetas «ID n» quedan en unos 11 px. El texto «borra esas salidas» no nombra el menú de Colab: los nombres de menú en español no están comprobados (nota e).

---

## Correcciones tras las revisiones adversariales (24-sep-2026)

Entrada: dos revisiones (alumno y técnica) y el veredicto de un escéptico por hallazgo; cuando no coincidían, se aplicó la corrección del escéptico. Solo se tocó `src/build_nb2.py` y el `.ipynb` se regeneró. Dos ejecuciones dan el mismo SHA-1: `66c06c020fc98511e2713c2ed45314fa87c7c8cc`. No cambió ningún nombre de parámetro de formulario ni ningún campo del CSV. El JSON conserva todos los campos del contrato y suma dos. Copias de antes y material de prueba en `verificacion/fix2_nb2/`.

### Qué cambió

| Dónde | Cambio |
|---|---|
| 2.3 ℹ️ (mayor) | Dice que cada persona sin caja es un falso negativo (suelen ser pequeñas, del fondo o en grupo) y que entonces 2.5 y 2.6 se quedan cortos. A 0,70 ya no pone «menos errores», sino «casi no pone cajas falsas, pero pierde muchas más personas reales». No se afirma que falten «sobre todo las del fondo», porque el mismo texto sale con el vídeo del alumno |
| 2.9 Límites | Viñeta nueva, **Tamaño**: el detector reduce la imagen a 640 px de lado largo y pierde a las personas pequeñas o lejanas. **Perspectiva** reescrita (centro de la caja, más o menos a la cintura; con cámara inclinada alguien cuenta en una zona que no pisa) |
| Demo 2, sentido (mayor) | Clave `sentido` en `DEMOS` (solo demo 2). Con la línea «auto», el ℹ️ de 2.4 dice que entrada es alejarse del tren hacia el andén (por ejemplo, al bajar) y salida es ir hacia el tren. El ✅ de 2.6 queda «Entradas (se alejan del tren): 9 · Salidas (van hacia el tren): 3». Con `invertir_sentido` los textos se intercambian. No se escribe «bajan del tren»: nadie ha comprobado que los 9 bajen (uno es el ID 3, contado por un temblor de 1 px) |
| 2.4 ℹ️, línea inclinada | Si la línea forma más de unos 11° con la horizontal y con la vertical (en px), se añade: «solo importa si la persona se mueve hacia la derecha o hacia la izquierda, no a qué lado de la línea acaba» |
| Demo 2, historia | Clave `historia`: ℹ️ en 2.2 «Empezamos en el segundo 22, con el tren ya parado y las puertas abiertas: unos bajan, otros suben y el andén se vacía». En 2.4 (md): la izquierda es el lado del tren y se puede renombrar |
| 2.4 md y título | La coma separa x de y y los decimales van con punto (`12.5,30`). El título pasa a «(se escriben, no se dibujan con el ratón)», también en el índice |
| 2.4 formulario | Nota de la línea: «escribe 2 puntos (`x,y; x,y`) o deja «auto»», con lo que hace «auto» en tu vídeo y para qué sirve `invertir_sentido` |
| 2.4 vista previa | La línea se dibuja en `#7B0068`, el mismo morado que usa ObjectCounter en el vídeo. La flecha y el rótulo siguen en naranja. En el md de 2.6: «la línea de puerta es la morada y la flecha naranja marca el sentido de entrada» |
| Privacidad del CSV (portada y 2.8) | Portada: «solo llevan recuentos… Aun así, un recuento muy fino puede señalar a alguien (lo vemos en 2.8)» y «exportar solo recuentos, sin imágenes» en lugar de «los totales». ℹ️ de 2.8: se quita el ejemplo de las 9:02, que queda solo para la pregunta 3 de 2.9. Se añade que el CSV va instante a instante (cada 0,08 s en la demo 1 y 0,07 s en la demo 2) porque es una práctica, y que en un sistema real se agregaría por minutos y no se guardarían los recuentos bajos |
| Borrar resultados | Aviso de 2.2 (demo 2) y viñeta de la portada: «Editar → Borrar todos los resultados» (en inglés, *Edit → Clear all outputs*). En Colab, `datos/tmp/` se borra al cerrar la sesión. Además, metadatos `colab.private_outputs = true`, para que Colab no guarde los resultados (los vídeos con caras) al guardar el cuaderno |
| Portada, demo del andén | En lugar de «La usamos solo para la demostración en clase»: vídeo de pruebas sin origen publicado, base legal desconocida, no se redistribuye, y en un proyecto real sería lo primero que habría que resolver. No se dice «no lo guardamos», porque el cuaderno sí lo descarga y lo incrusta |
| Portada, licencias | AGPL: «si distribuyes… o una versión modificada que otros usen por internet… darles el código fuente; Ultralytics entiende que eso alcanza a toda tu aplicación» (lo 🟡 de la nota f ya no va como hecho). Vídeos: MIT cubre el código y no los vídeos; el del andén no tiene origen conocido; «su licencia es la de su fuente, que en el del andén no consta». Telemetría: sin «anónimas»; se dice que envía un identificador fijo del equipo calculado a partir de su dirección de red (hash SHA-256 del MAC en `events.py`) |
| Portada, otros | Enlace a las Directrices 3/2019 del CEPD. «Vuelve a ejecutar esa celda y las que vienen detrás». Aviso condicional de Colab («Si al pulsarlo Colab avisa… pulsa «Ejecutar de todos modos»») |
| 2.1 | Fila GPU «no hay (no hace falta: el cuaderno está pensado para CPU)». Instalación: «Instalando el detector y sus herramientas…» sin versiones. El ✅ dice que la tabla es para el profesor |
| 2.2 formulario | `saltar_fotogramas`: 1 = todos, 2 = uno de cada dos, 3 = uno de cada tres; con saltos grandes el seguimiento suele perder más gente |
| 2.2 plan B | `urlretrieve` (sin timeout) → `urlopen(url, timeout=60)` + `shutil.copyfileobj` a `.parcial` y renombrado |
| Nota para el profesor | Empieza por «(si eres alumno, sáltala y sigue con 2.3)». El clip de Pexels lleva «(ángulo sin comprobar: pruébalo antes)» |
| 2.3 md | Reescrito sin jerga suelta: versión más pequeña, casi todo convolucional y mucho mayor que la red del NB1, un par de bloques de atención («piezas que miran la imagen entera a la vez», para no romper la regla 6 de la nota f), *backbone* y cabezas explicados, qué es la confianza |
| 2.3 otros | Nota del umbral: «El valor que dejes aquí es el que usan 2.5 y 2.6». Aviso < 0,25: la caja dudosa no inicia un seguimiento, solo mantiene uno que ya existía (`new_track_thresh` 0,25 y `track_low_thresh` 0,1). Aviso de tiempo: «vuelve a ejecutar 2.2, 2.3 y 2.4» |
| 2.5 | Md: qué es ocupación y qué es aforo permitido, y que el ID es una etiqueta y no un recuento. ℹ️ de la gráfica reescrito: la línea es la ocupación de cada instante, y círculos y cuadrados solo distinguen las dos líneas |
| Tiempos (2.5 y 2.6) | Una sola línea de progreso: «Procesados 125 fotogramas en 8,5 s (14,6 por segundo); 9,1 s contando la conversión del vídeo». Se quitan los ℹ️ de tiempo de 2.5 y 2.6, que mezclaban segundos totales con la velocidad del bucle |
| 2.6 md | Doble conteo: cada ID cuenta una vez, en su primer cruce, y si entra y vuelve a salir su salida no cuenta. Viñeta nueva **Temblor sobre la línea** |
| 2.7 | Si más de la mitad de las estancias están recortadas: «Con un tramo de 10,0 s estas cifras dicen poco: 30 de 35 estancias…; para medir permanencia hacen falta minutos de vídeo». Si no, el aviso de antes con «8 de 29 estancias» en lugar de «estancia(s)». Sin estancias, no hay aviso |
| 2.8 JSON | Campos nuevos: `permanencia_estancias` y `permanencia_recortadas` (`null` si no se ejecutó 2.7). `generado_en` con zona horaria (`astimezone()`, p. ej. `2026-09-24T10:53:07+02:00`). En `ESTADO`, `permanencia` pasa a ser `{"media_s", "estancias", "recortadas"}` (dato interno; el JSON sigue exportando `permanencia_media_s` igual que antes) |
| 2.8 md y tabla | Aviso de las descargas múltiples en Colab. Cabecera: «Segundos desde el inicio del tramo (columna t_s)» |
| 2.8 webhook | Nota del formulario: «Opcional. Si tienes un flujo de n8n con un nodo Webhook (método POST) escuchando…». Un 404 explica método POST (en GET por defecto), «Listen for test event» y flujo activado (publicado), y muestra los primeros 200 caracteres de la respuesta de n8n. `http.client.HTTPException` (BadStatusLine, IncompleteRead) va con los errores de conexión y ya no deja traza |
| Flujo n8n (md) | Una línea: el máximo es de un solo fotograma y puede ser un parpadeo; en un sistema real se usaría el que se mantiene unos segundos |

### Qué no se aplicó, y por qué

- **Badge `<USUARIO>/<REPO>`**: el generador se queda como está, porque BRIEF §11 exige ese marcador. **Pendiente para el profesor:** sustituirlo o quitar el badge antes de subir el cuaderno al aula virtual (añadir al checklist previo a clase).
- **Renombrar 2.5 a «Ocupación por zonas»**: no. «Aforo por zonas» es el nombre del spec y del BRIEF y está también en `PASOS` y en el índice. Se añadió la línea que distingue ocupación de aforo permitido.
- **Subir el markdown del flujo de n8n por encima de la celda del webhook**: no, porque rompería su «recibe el JSON de arriba». Solo cambió la nota del formulario.
- **Campo `maximo_1s` en el JSON (mediana móvil de 1 s)**: no. Añade un campo y su explicación para alumnos no técnicos, y el ejemplo del IF (> 12) salta igual con 14 que con 13. Se dejó la línea de aviso bajo el flujo.
- **CSV agregado por segundo**: no. El `agg(['max','mean'])` propuesto da columnas MultiIndex y rompe el formato del spec (`t_s` + una columna por zona). El texto explica cómo se haría en un sistema real.
- **Cabecera de la tabla de 2.4 («la coma separa x de y»)**: no, es ruido. La aclaración está en el markdown.
- **Título nuevo en el histograma de 2.7**: no hacía falta. Los ejes ya dicen qué es cada barra.
- **Redirección http → https en el webhook** (urllib convierte el POST en un GET sin cuerpo): no se trata. Es poco probable con la URL copiada de n8n, y el caso acaba en el 404 explicado.
- **Repetir «no es asesoramiento jurídico» en Licencias**: no. Es opcional, y el aviso ya está en el recuadro de privacidad de la misma portada.
- **`ESTADO["permanencia"] = medias`** (lo que pedía el escéptico): no se siguió al pie de la letra. El contrato es el JSON, no `ESTADO`. Un diccionario en una sola clave evita tener que añadir otra clave a `DEPENDEN` y a `reiniciar`.

### Verificación

| Ejecución | Resultado | 2.3 + 2.5 + 2.6 |
|---|---|---|
| `--modos rapido` | ✅ 38 fotogramas; 13/11; 2/3; aviso «20 de 20 estancias… dicen poco» | 8,9 s |
| `--modos defecto` | ✅ 17 / 39 / 4 personas en 2.3; zonas 14/13, media 11,0/9,5; **8/8**; «30 de 35 estancias», aviso fuerte; JSON con `permanencia_estancias` 35 y `permanencia_recortadas` 30 | 20,8 s |
| `--modos defecto --set 'fuente_video="Demo 2: andén de metro (interior)"'` | ✅ ℹ️ de historia; ℹ️ de 2.4 con línea inclinada y sentido del andén; «Entradas (se alejan del tren): 9 · Salidas (van hacia el tren): 3»; zonas 6/7; «8 de 29 estancias», aviso suave; CSV cada 0,07 s | 26,2 s |
| `--modos rapido --set 'zona_1="10,10; 20"'` | ✅ «No entiendo la zona 1…» y «Falta el paso anterior…» en las siguientes, sin traza | — |
| `--modos rapido --set webhook_n8n=…:8765` (servidor local que responde 404 con el cuerpo de n8n) | ✅ consejo de POST, «Listen for test event» y la respuesta de n8n | — |
| `--modos rapido --set webhook_n8n=…:8766` (respuesta sin línea de estado HTTP) | ✅ «No he podido conectar…», sin traza | — |

Ninguna ejecución tiene salidas `error` ni `stderr`. Todas las celdas de código compilan con Python 3.9, así que ninguna f-string reutiliza su comilla. Imágenes revisadas: vista previa de zonas de las dos demos (línea morada, flecha naranja) y último fotograma del vídeo de 2.6 de la demo 2 (línea morada, «Entradas: 9 · Salidas: 3»).

### Pendiente o sin comprobar

- 🟡 **`colab.private_outputs`**: es la opción de Colab «Omitir el resultado de las celdas de código al guardar este cuaderno». No se ha comprobado en Colab que la clave se respete al abrir el cuaderno desde GitHub o Drive. Si no se respeta, queda el aviso de borrar los resultados.
- 🟡 **Nombres de menús y botones**: «Editar → Borrar todos los resultados» y «Ejecutar de todos modos» (Colab en español), y «Listen for test event» y «publicado» (n8n, según su versión). Comprobarlos a mano.
- **Guion y D30**: el 9/3 de la demo 2 no mide pasajeros que suben. Entrada = alejarse del tren (dx > 0). Si el guion lo interpreta, que sea así, sin hablar de embarque.
