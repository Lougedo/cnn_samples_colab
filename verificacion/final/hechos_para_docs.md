# Hechos medidos para la documentación (coordinador, 24-sep-2026)

Fuente: ejecución final en serie, sin otros procesos, con `scripts/verificar.py`. Máquina: Apple M3 Pro (11 núcleos),
Python 3.12.7, TF 2.21.0 / Keras 3.15.1, ultralytics 8.4.161. Detalle en `verificacion/final/resumen_medidas.txt` y
`verificacion/informe_20260924-1119*.md` … `1125*.md`. Todo pasó: 0 errores, 0 salidas por stderr, estructura OK.
El NB1 también pasó (rápido y por defecto) en `.venv-colab` (Python 3.13 + Keras 3.13.2, numpy 2.1.3, pandas 2.2.3,
matplotlib 3.10.0, IPython 7.34.0: las versiones de Colab) con métricas idénticas.

## Tiempos

| Qué | Todos los hilos | Limitado (proxy de Colab) | Proyección Colab gratuito 🟡 |
|---|---|---|---|
| NB1 · celda 1.4 con la red por defecto (8 épocas) | 5,5 s (entrenamiento 4,0 s) | 8,2 s con 2 hilos (entrenamiento 6,7 s) | ~20–30 s (factor 2,5–3,5 sobre 2 hilos) — presupuesto del brief: ≤ 90 s ✅ |
| NB1 · notebook completo por defecto (incluye arranque del kernel y 1.1) | 26 s | 30 s | ~1–1,5 min |
| NB1 · Plan B (7 entrenamientos) | 54 s | 121 s con 2 hilos | ~5–7 min |
| NB1 · 1.1 Preparación (ya instalado) | ~9 s | ~9 s | la primera vez en Colab instala `medmnist` (tiempo no medido 🔴) |
| NB2 · 2.3 + 2.5 + 2.6, demo 1 (10 s, salto 2) | 18,6 s | 24,7 s con 1 hilo | ~60–90 s — presupuesto del brief: ≤ 2 min ✅ |
| NB2 · 2.3 + 2.5 + 2.6, demo 2 (andén, 10 s, salto 2) | 25,2 s | 31,3 s con 1 hilo | ~80–110 s (al límite; la 2.3 avisa y sugiere salto 3) |
| NB2 · notebook completo (demo 1) | 26 s | 31 s | ~1,5–2 min + primera instalación (no medida 🔴) |

Por qué esos proxies: Colab gratuito tiene 2 vCPU (🟡, fuentes comunitarias) y Ultralytics fija torch a `min(8, núcleos−1)` hilos,
o sea 1 hilo en Colab. El factor 2,5–3,5 sale de comparar un núcleo de M3 Pro con un Xeon de 2,2 GHz (🟡, no medido en Colab).

## Resultados de referencia del NB1 (Plan B, semilla 42, 28 px)

Con todos los hilos. Entre paréntesis, con 2 hilos (cambian los decimales, no las tendencias).

| Experimento | Reto | Parámetros | Tiempo | Acierto val. | Acierto test | Sensibilidad | Especificidad | AUC | Diagnóstico |
|---|---|---|---|---|---|---|---|---|---|
| ref_defecto | — | 56.129 | 4,0 s (6,7) | 94,5 % | 87,7 % (86,9) | 94,9 % (94,9) | 75,6 % (73,5) | 0,936 | razonable (aún mejoraba) |
| ref_A_1_bloque | A | 173.345 | 3,0 s (5,5) | 95,2 % | 84,1 % (84,5) | 97,9 % (97,9) | 61,1 % (62,0) | 0,924 | razonable (aún mejoraba) |
| ref_A_3_bloques | A | 27.521 | 4,2 s (7,4) | 92,6 % | 86,2 % (85,7) | 94,6 % (94,6) | 72,2 % (70,9) | 0,936 | razonable (aún mejoraba) |
| ref_B_sin_pooling | B | 1.184.577 | 7,7 s (16,5) | 95,6 % | 86,5 % (87,3) | 97,9 % (97,9) | 67,5 % (69,7) | 0,927 | razonable |
| ref_C_dropout_0_20_epocas | C | 1.184.577 | 15,0 s (38,3) | 95,8 % | 87,5 % (86,9) | 98,2 % (98,5) | 69,7 % (67,5) | 0,939 | **sobreajuste** |
| ref_C_dropout_05_20_epocas | C | 1.184.577 | 14,8 s (37,6) | 96,0 % | 88,1 % (87,8) | 97,9 % (97,9) | 71,8 % (70,9) | 0,946 | razonable |
| ref_D_compensar_desbalanceo | D | 56.129 | 4,1 s (6,9) | 92,6 % | 87,8 % (88,1) | 93,8 % (94,4) | 77,8 % (77,8) | 0,931 | razonable (aún mejoraba) |

Referencia tonta (responder siempre «Neumonía»): 62,5 % de acierto en test (sensibilidad 100 %, especificidad 0 %).

**Cómo leerlo (tendencias, no garantías):**
- Ruido entre semillas: con la misma red y otra semilla, la especificidad pasó de 75,6 % a 58,5 % (calibración de la fase 2).
  Diferencias de 1-3 puntos entre dos entrenamientos sueltos pueden ser azar. Lo que sí se ve con claridad: parámetros, tiempo y diagnóstico.
- **Reto A (profundidad):** con pooling, **más bloques = menos parámetros** (1 bloque 173 mil, 3 bloques 27 mil: 6 veces menos). La red de 1 bloque
  tiende a decir «Neumonía» más a menudo (sensibilidad 98 %, especificidad 61 %). 3 bloques ≈ la red por defecto.
- **Reto B (sin pooling):** **21 veces más parámetros** (1,18 M) y el doble de tiempo, sin mejora clara en test.
- **Reto C (sin pooling + 20 épocas, dropout 0 frente a 0,5):** con dropout 0 el diagnóstico sale **sobreajuste** (las curvas de entrenamiento y
  validación se separan); con 0,5 sale **razonable**. Las métricas de test apenas cambian (≤ 1 punto): la lección está en las curvas, no en la cifra.
  Estable en 3 de 3 semillas en la calibración; en Colab los decimales cambian y conviene decir «deberían separarse».
- **Reto D (compensar):** la especificidad sube unos 2 puntos y la sensibilidad baja alrededor de 1: la red pasa a decir «Normal» algo más a menudo.
  Efecto pequeño, del mismo orden que el ruido: presentarlo como tendencia.
- **Validación ~95 % frente a test ~87 % no es sobreajuste:** el test es otro conjunto (el de prueba original) con menos neumonías (62,5 % frente a
  74,2 %) y en él cae sobre todo la especificidad (88 % → 76 % con la red por defecto). El cuaderno lo calcula en vivo.

## Resultados de referencia del NB2 (confianza 0,35, salto 2, 10 s)

| | Demo 1 · personas caminando (0–10 s) | Demo 2 · andén de metro (22–32 s) |
|---|---|---|
| Fotogramas procesados · resolución | 125 · 1280×720 | 150 · 720×1280 (vertical) |
| Personas en el primer fotograma (2.3) | 17 cajas con 0,35 · 39 con 0,10 · 4 con 0,70 (a simple vista hay más de 40) | — |
| Zonas (máx. / media) | Izquierda 14 / 11,0 · Derecha 13 / 9,5 | Izquierda 6 / 2,9 · Derecha 7 / 4,4 |
| Línea de puerta «auto» | horizontal al 50 %: **8 entradas / 8 salidas** | diagonal `62,15; 22,100`: **9 entradas (se alejan del tren) / 3 salidas (van hacia el tren)** |
| Permanencia media | 5,6 s / 5,7 s (35 estancias, 30 recortadas por los bordes del tramo) | 2,3 s / 2,1 s (29 estancias, 8 recortadas) |

Ojo: en la demo 2 la línea **no** cuenta «subir al tren»: cuenta movimiento que se aleja o se acerca al tren.
En la demo 1 el detector pierde a muchas personas visibles (pequeñas, agrupadas): el aforo se queda corto y el cuaderno lo dice.

## Contexto de la sesión real (S7 de IASP, jueves 24-sep-2026, online)

- `sesiones/S7/01-esqueleto.md`: bloque 8 «La CNN por dentro: CNN Explainer, notebook 1 y Grad-CAM», **28 min, 20:30–20:58**;
  bloque 9 «Contar personas en vídeo: notebook 2», **12 min, 20:58–21:10**; después, dudas de Make/n8n/Colab (21:10) y cierre (21:22).
  El brief prevé 60 min; en la S7 hay 40. Entregable del alumno en la S7: «su notebook de la CNN ejecutado, con la arquitectura que ha elegido y su Grad-CAM».
- Clase **online**: sin parejas ni salas separadas. Chat y micrófono abierto. El trabajo del cuaderno es **individual**; el profesor asigna un reto
  a cada alumno (el cuaderno pide nombre `iniciales_reto`, p. ej. `ALM_A`) y cada uno pega su línea de 1.8 en una **hoja común** cuyo enlace
  el profesor pone en el chat (🔴 hay que crearla antes, con la cabecera que da la línea de 1.8).
- 1.10 es «Plan B (solo el profesor)»: el profesor puede ejecutarla antes de clase en su Colab para tener la tabla de referencia a mano.
- Badges «Open in Colab» con `<USUARIO>/<REPO>`: hay que subir los notebooks a GitHub o Drive y cambiar el enlace antes de pegarlo en el chat.
- Si se abre desde GitHub, Colab puede avisar de que el cuaderno no lo ha creado Google (botón «Ejecutar de todos modos», rótulo 🟡 sin comprobar).

## CNN Explainer (ensayado en navegador, `docs/notas_verificacion/h_ensayo_cnn_explainer.md`)

Tiny VGG: entrada 64×64×3, 4 convoluciones de 10 filtros 3×3 sin relleno + ReLU, 2 max pooling 2×2, 10 clases; 64 → 62 → 60 → 30 → 28 → 26 → 13.
Recorrido: cerrar el recuadro «You might also like»; **Show detail**; clic en una neurona de `conv_1_1` → vista intermedia (canales R, G, B y suma);
**segundo clic** en un cuadro *intermediate* → ventana «Convolution» (64×64 → 62×62, pasar el ratón mueve el núcleo); cerrar; clic en `relu_1_1` →
«ReLU Activation» (`max(0, x)`; otro clic en la misma neurona la cierra); clic en `max_pool_1` → «Max Pooling» (60×60 → 30×30); clic en la clase
*espresso* → flatten + softmax (en el ensayo, 0,86). Solo en inglés: el profesor traduce. La web la abre solo el profesor compartiendo pantalla
(carga un script de `polyfill.io`, dominio comprometido en 2024; hoy responde 403 y la página funciona sin él).
Tres preguntas (nota f §2): el mismo filtro recorre toda la imagen (¿qué ganamos?); tras el pooling queda una cuarta parte (¿qué ganamos y qué
perdemos? → reto B); si le enseñamos un perro, ¿qué contesta? (el softmax siempre reparte el 100 % → atajos y Grad-CAM).
Nuestro NB1 usa el mismo padding que CNN Explainer (`valid`): la imagen encoge 2 px por convolución 3×3.

## Lo que está pendiente de comprobar a mano en Colab (no verificable en local)

Vista de formulario (tildes, desplegables numéricos, deslizador de dropout con 0.2), `cellView` oculta el código, curvas en vivo que se
actualizan en la misma figura, `<details>` desplegable en 1.4, descarga del CSV (`files.download`) y de los dos archivos del NB2, subida de vídeo
(`files.upload`), reproducción del vídeo H.264 incrustado (Chrome y Safari), tiempo de la primera instalación (medmnist; ultralytics + supervision
+ lap), aviso «Ejecutar de todos modos», `colab.private_outputs` en el NB2, rótulo del menú «Editar → Borrar todos los resultados»,
tiempos reales en CPU gratuita (default ≤ 90 s y ≤ 2 min), que la instalación no pida reiniciar la sesión con Python 3.13.
