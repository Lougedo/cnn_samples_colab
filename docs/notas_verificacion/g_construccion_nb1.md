# g) Construcción del Notebook 1 — decisiones, calibración y mediciones

Fecha: 24-sep-2026. Ficheros: `src/build_nb1.py` (generador) → `notebooks/01_laboratorio_cnn_radiografias.ipynb`
(23 celdas: 13 de markdown y 10 de código, todas con `# @title … { display-mode: "form" }` y `cellView: "form"`,
sin salidas). El generador es determinista: dos ejecuciones seguidas dan el mismo SHA-1.
Material de trabajo en `verificacion/build_nb1/` (ignorado por git): `calibrar.py` y `calib_*.jsonl` (calibración),
`escenarios.py` (recorrido de alumno fuera de orden), `forzar_rutas.py` (fallo de pip y Grad-CAM vacío),
`ver_salidas.py` (vuelca texto y PNG de un notebook ejecutado), `final_*.ipynb` (ejecuciones finales).

**Aviso sobre los tiempos.** Apple M3 Pro de 11 núcleos con el otro constructor ejecutando el NB2 a la vez
(carga media 9-10). Todos los tiempos son orientativos (±30 %); la medición final la hace el coordinador.

---

## 1. Estructura y contrato

| Celda | Qué hace | Formulario (nombres exactos, contrato con `verificar.py`) |
|---|---|---|
| 1.0 | Portada: objetivo, aviso «no apto para uso clínico», cómo se usa (maqueta en texto de una celda de formulario), índice, badge `<USUARIO>/<REPO>`, atribución CC BY 4.0 con cambios | — |
| 1.1 | Instalación de lo que falte, imports, versiones, GPU, semilla, **todas las funciones compartidas** y `ESTADO` | — |
| 1.2 | Descarga con timeout y MD5, lectura con medmnist, rejilla 4×4, reparto de clases, referencia tonta | `resolucion` |
| 1.3 | Validación, ficha (diagrama + tabla + total + diferencia), estimación de tiempo | `bloques_convolucionales` … `nombre_experimento` (11) |
| 1.4 | Entrenamiento con curvas en vivo, diagnóstico con reglas visibles, registro | — |
| 1.5 | Métricas de test, matriz de confusión, comparación con la referencia tonta, textos de sensibilidad y rigor | — |
| 1.6 | Imagen ampliada, filtros de `conv_1`, 8 mapas de activación por bloque | `imagen_a_analizar`, `indice_imagen` |
| 1.7 | Grad-CAM (3 aciertos + 3 fallos) y recuadro de debate de Zech | — |
| 1.8 | Tabla, gráfico, línea para la hoja común, CSV (descarga en Colab) | — |
| 1.9 | Retos A-D + Libre, hipótesis | `reto`, `hipotesis` |
| 1.10 | Plan B: 7 configuraciones de referencia | `ejecutar_plan_b` |
| 1.11 | Tres preguntas y puente al NB2 | — |

CSV: `salidas/nb1/experimentos.csv`, columnas exactamente en el orden del contrato, `sep=";"`, `decimal=","`,
`utf-8-sig`. Los booleanos se escriben «sí»/«no». Las columnas decimales se fuerzan a `float64` antes de escribir:
con `None` mezclado, pandas las deja como `object` y entonces **ignora `decimal=","`** (comprobado).

## 2. Decisiones y motivos

1. **Todas las funciones compartidas viven en 1.1; cada celda envuelve su lógica en una función** y la llama tras
   una guarda (`if "ESTADO" not in globals(): print(...)` / `elif requiere(...)`). Así un error previsible sale
   con `return` y un recuadro en castellano, nunca con traza, y el Plan B (1.10) funciona justo después de 1.2.
2. **Figuras como PNG con `IPython.display.Image`** (helper `png(fig)`), no con `plt.show()`: no depende del backend
   (`verificar.py` fija `MPLBACKEND=Agg`, que no es el inline) y es el mismo mecanismo para las curvas en vivo
   (`display(png(fig), display_id=True)` + `handle.update(png(fig))`). Siempre `plt.close`.
3. **Arquitectura tal cual la especificación**: `Flatten → [Densa] → [Dropout] → logit → sigmoide`. Ojo: la fase 1
   (nota b) medía con el dropout *antes* de la densa. Con dropout 0 las dos redes son idénticas; con dropout > 0
   no, y por eso el reto C se ha recalibrado entero con esta arquitectura (sección 4).
4. **Predicción con `modelo(x, training=False)` por trozos de 256**, no con `predict`: no pinta barras y evita el
   aviso «N out of the last N calls … triggered tf.function retracing» que salió al predecir con muchos modelos
   seguidos durante la calibración (en el Plan B se construyen 7).
5. **Estimación de tiempo = mediana de 5 lotes** (tras uno de calentamiento), extrapolada a épocas × pasos
   (+ validación a 1/3 del coste) y × 1,2 (nota b). La primera versión usaba la media y un pico de carga convirtió
   una red de ~3 s en «~56 s». En el Plan B se mide **una vez por arquitectura** (4 medidas en lugar de 7; el
   dropout no cambia el coste de forma apreciable): la celda sin ejecutar bajó de 15,1 s a 4,9 s en local.
6. **Umbrales** (constantes en 1.1): > 20 M de parámetros → bloqueo sin estimar (nota b: estimar 103 M cuesta
   30-50 s); estimación > 600 s → bloqueo; 90-600 s → aviso «unos N minutos» y deja entrenar; último mapa ≤ 3×3 →
   aviso de Grad-CAM grueso. Comprobado: 28 px, 4 bloques sin pooling, 32 filtros, densa 128 (13,5 M) → aviso
   «~218 s (unos 4 minutos)»; 64 px con lo mismo (103 M) → bloqueo con «103,1 millones de parámetros (393 MB)».
7. **Configuración imposible**: se recorre `s ← s − k + 1`, `s ← ⌊s/2⌋` antes de construir. El mensaje lleva la
   trayectoria («28 → 13 → 5 → 1 → 0») y **solo las soluciones que funcionan**: cada una se comprueba construyendo
   la red candidata (tamaño ≥ 1 y ≤ 20 M de parámetros). Ej. 64 px, 4 bloques, k=5: «usa 3 bloques como máximo, usa
   kernel 3 o desactiva el pooling» (subir a 64 no aparece porque ya está en 64).
8. **`BatchNormalization(momentum=0.9)`** (fase 1: con 0,99 colapsa). Aun así, en el escenario BN + sin densa +
   dropout 0,3 salió test 78,7 %, sensibilidad 100 %, especificidad 43,2 %: la BN sigue siendo sensible aquí.
9. **Diferencia con la red anterior**: se compara con el último diseño *distinto*. Re-ejecutar 1.3 con lo mismo no
   pierde la comparación (idempotente). Mensajes: «Primera red de la sesión», «Mismo número de parámetros…»,
   «+117.216 parámetros respecto a la red anterior (3,1 veces más)».
10. **Registro**: 1.4 añade la fila (sin métricas de test) y 1.5 la actualiza por `id`. Así ningún entrenamiento
    se pierde aunque el grupo no evalúe. El reto y la hipótesis de 1.9 se adjuntan a los entrenamientos
    posteriores. El Plan B sustituye las filas con el mismo nombre `ref_…` si se repite.
11. **Gráfico del registro**: dos paneles (sensibilidad y especificidad frente a parámetros, eje log) con la
    referencia tonta (100 % / 0 %). **Los puntos se rotulan con el `#` de la tabla, no con el nombre**: con los
    nombres del Plan B se pisaban (B y las dos C comparten 1,18 M de parámetros) y en clase casi todos los
    experimentos de un grupo se llamarán igual. La tabla con nombres está justo encima. Desviación consciente de
    la especificación («con el nombre de cada experimento»).
12. **Línea para la hoja común**: cabecera + fila separadas por tabuladores, columnas del CSV sin `id`, coma
    decimal y enteros **sin** punto de miles (una hoja en inglés leería «56.129» como 56,129).
13. **Mapas de activación**: se mantienen los 8 de mayor activación media (especificación). Probé también por
    desviación típica y por máximo (`comparar_mapas.py`): eligen casi los mismos canales, porque esta red pequeña
    aprende varios filtros de «brillo» parecidos (2 de 16 canales muertos en `relu_1` y 13 de 32 en `relu_2`). Los
    títulos dicen lo que se ve: «detalle fino: brillo, bordes y contrastes» (primer bloque), «combinaciones de
    bordes: texturas y formas» (intermedios), «zonas más amplias y patrones más abstractos» (último), y una línea
    explica que mapas parecidos = filtros que han aprendido cosas parecidas. A 64 px se ven las costillas.
14. **Grad-CAM** en lote para todo el test (signo +1/−1 según la predicción; un `GradientTape` por trozo de 128),
    sobre la última `relu_N`, `tf.image.resize` bilineal, `viridis` con alpha 0,45. Elección reproducible: 3
    aciertos y 3 fallos alternando clase predicha y prefiriendo mapas no vacíos. La ruta «Grad-CAM no encuentra
    ninguna zona que empuje hacia…» se ha forzado (`forzar_rutas.py`) y se lee bien.
15. **Imagen «aleatoria» en 1.6**: `default_rng(42 + contador)`; cambia en cada ejecución y es reproducible.
    `indice_imagen` fuera de 0-623 → error en castellano.
16. **Nombre del experimento** saneado: «;» → «,», espacios colapsados, vacío → «sin_nombre». Hipótesis y nombres
    se escapan en HTML.
17. **Interrupciones**: `KeyboardInterrupt` durante 1.4 o el Plan B → aviso y la celda termina limpia. Cualquier
    otra excepción en `fit` (p. ej., memoria) → error en castellano con la acción y el detalle técnico en pequeño.
18. Si se ejecuta el Plan B sin haber entrenado nada, `ref_defecto` pasa a ser el modelo actual y 1.5-1.7
    funcionan igualmente (Plan B de verdad si el laboratorio falla en directo).
19. **Instalación**: `find_spec` primero; si falta algo, `pip install -q` con restricciones que fijan numpy,
    tensorflow, keras, torch, opencv-python, pandas y matplotlib instalados (keras no se fija si falta TF).
    Probada la ruta de fallo con `pip` simulado: devuelve `False`, pinta el error con el detalle y no crea `ESTADO`.
20. **Ruido**: `TF_CPP_MIN_LOG_LEVEL=2` antes de importar, logger de TF y absl a ERROR, `fit(verbose=0)`, sin
    `predict`. Ningún `warnings.filterwarnings` global. Resultado: **0 bytes en stderr** en todas las ejecuciones.
21. Explicaciones de jerga en su primer uso: CNN, filtro, ReLU, pooling, normalización por lotes, dropout, época,
    parámetro, pérdida, validación, sobreajuste/infraajuste, sensibilidad/especificidad/AUC, gradientes.

## 3. Diagnóstico automático: reglas y calibración

Reglas (constantes `UMBRAL_*` en 1.1; el texto sale literal bajo el diagnóstico, con las cifras de ese entrenamiento):

1. **Colapso** si en validación dice lo mismo en ≥ 97 % de los casos (una red buena dice «Neumonía» en ~73-78 %).
2. **Sobreajuste** si la pérdida de validación acaba ≥ 10 % por encima de su mínimo y ese mínimo quedó ≥ 3 épocas
   antes del final, **o** si al final es ≥ 1,5 veces la de entrenamiento (curvas separadas).
3. **Infraajuste** si el acierto no llega al 90 % ni en entrenamiento ni en validación.
4. **Razonable**; «(aún mejoraba)» solo si el mínimo de validación está en la última época. Si subió ≥ 10 % en las
   dos últimas épocas (sin cumplir la regla 2), dice que puede ser ruido o el principio de un sobreajuste.

Contrastadas con 60 entrenamientos (`calib_*.jsonl`, arquitectura del NB1):

| Casos | Resultado |
|---|---|
| Defecto 8 épocas, semillas 42/1/2 | razonable (aún mejoraba) 3/3 (acierto de entrenamiento 92,4-93,4 %) |
| Defecto 20 épocas, dropout 0 y 0,5, semillas 42/1 | razonable 4/4 (el mínimo de validación es la época 20) |
| Defecto 2 y 3 épocas (modo rápido) | infraajuste (78 % / 87 % en entrenamiento) |
| 1 bloque, 8 filtros, sin densa, 3 épocas | infraajuste |
| 3 bloques, 8 filtros sin duplicar, sin densa, 3 épocas (con y sin dropout) | colapso (siempre «Neumonía», test 62,5 %) |
| 1 bloque, 3 bloques, k=5, 64 px, sin pooling, BN, dropout 0,6 y `class_weight` (8 épocas, s42; `class_weight` también con s1) | razonable 9/9 |
| BN 8 épocas, semilla 1 | sobreajuste (mínimo en la época 4, +32 %) |
| Reto C (ver sección 4) | sobreajuste con dropout 0 en 3/3 semillas; razonable con 0,5 en 3/3 |

La regla de la separación (×1,5) es la que hace estable el reto C: la subida de la pérdida de validación por sí
sola solo supera el 10 % en 1 de 3 semillas.

## 4. Reto C: búsqueda de una red que sobreajuste de forma estable

20 épocas, dropout 0 frente a 0,5, semillas 42 y 1 (más semillas en los candidatos prometedores). «mín» = época del
mínimo de la pérdida de validación; «+» = subida final respecto a ese mínimo; «×» = pérdida de validación / pérdida
de entrenamiento al final. Tiempo = media local con todos los hilos (ruidoso).

| Red (resto por defecto) | Parámetros | Dropout 0 | Dropout 0,5 | s |
|---|---|---|---|---|
| Por defecto | 56.129 | s42: mín 20, +0 %, ×0,94 · s1: mín 20, +0 %, ×0,91 | s42: mín 20, ×0,79 · s1: mín 20, ×0,76 | 10-12 |
| **Sin pooling (densa 64)** | 1.184.577 | **s42: mín 14, +12 %, ×1,73 · s1: mín 15, +8 %, ×2,25 · s2: mín 19, +1 %, ×2,34** | **s42: mín 18, +4 %, ×1,34 · s1: mín 19, +4 %, ×1,13 · s2: mín 17, +0 %, ×1,07** | 14-19 |
| Sin pooling, densa 128 | 2.364.353 | s42: mín 16, +30 %, ×2,69 · s1: mín 14, +10 %, ×2,65 · s2: mín 19, +6 %, ×1,99 · s3: mín 18, +5 %, ×1,45 | s42: mín 17, +9 %, ×1,40 · s1: mín 19, +4 %, ×1,67 · s2: mín 18, +4 %, ×1,47 · s3: mín 17, +18 %, ×1,29 | 19-21 |
| Sin pooling, 8 filtros, densa 128 | 1.181.153 | s42: mín 15, +3 %, ×1,75 · s1: mín 14, +3 %, ×1,99 · s2: mín 14, +4 %, ×1,83 | s42: mín 20, ×1,18 · s1: mín 17, +3 %, ×1,59 · s2: mín 16, +1 %, ×1,08 | 15-18 |
| 1 bloque, 32 filtros, sin pooling, densa 128 | 2.769.473 | s42: mín 20, ×1,69 · s1: mín 20, ×1,47 | s42: ×1,06 · s1: ×1,07 | 15-16 |
| 1 bloque, 32 filtros, pooling, densa 128 | 692.801 | s42: mín 18, +3 %, ×1,12 · s1: mín 20, ×1,18 | s42: ×1,05 · s1: ×0,98 | 8-9 |
| 1 bloque, 16 filtros, pooling, densa 128 | 346.529 | mín 20 en ambas, ×1,09 | ×0,91-0,97 | 7 |
| 2 bloques, 32 filtros, pooling, densa 128 | 223.873 | mín 20 en ambas, ×0,90-0,97 | ×0,81-0,82 | 14-15 |

**Decisión**: reto C = **sin pooling y 20 épocas, dropout 0 frente a 0,5** (densa 64, la de por defecto). Es la
red más barata con el efecto estable: con dropout 0 las curvas se separan (×1,7-2,3) en 3/3 semillas y el
diagnóstico dice «sobreajuste»; con 0,5 quedan juntas (×1,07-1,34) y dice «razonable» en 3/3. La subida de la
pérdida de validación en sí es pequeña e inestable (+1 % a +12 %), así que el enunciado pide mirar **si se
separan las curvas**, no «si sube la validación». El candidato de la fase 1 (sin pooling + densa 128) es más caro
y menos estable con esta arquitectura: s3 sobreajusta *más* con dropout 0,5 que con 0. Enlaza con el reto B (misma
red a 8 épocas). En el Plan B: `ref_C_dropout_0_20_epocas` = sobreajuste y `ref_C_dropout_05_20_epocas` =
razonable (sección 5).

## 5. Mediciones de la verificación final

Ejecuciones con `scripts/verificar.py --nb 1` sobre el notebook final (o sobre la versión inmediatamente anterior,
que solo difería en textos). **Todas: 0 errores, 0 líneas en stderr, sin problemas de estructura** (cada celda de
código con `cellView: "form"` y `# @title` en la primera línea, sin salidas guardadas).

| Entorno | Modo | Total (con arranque del kernel) | 1.1 | 1.3 | 1.4 (entreno) | 1.10 |
|---|---|---|---|---|---|---|
| `.venv` (Py 3.12, Keras 3.15) | rápido (2 épocas) | 27,3 s | 10,2 s | 1,4 s | 4,1 s | 5,2 s (solo estimación) |
| `.venv` | defecto | 29,3 s | 10,4 s | 1,3 s | 6,8 s | 4,2 s (solo estimación) |
| `.venv` | defecto, `--hilos 2` | 32,3 s | 10,7 s | 1,3 s | 8,9 s | 5,1 s |
| `.venv` | imposible (4 bloques a 28 px) | 19,2 s | 9,5 s | 0,2 s | 0 s | 5,1 s |
| `.venv` | referencias (Plan B) | 87,6 s | 11,7 s | 1,3 s | 6,1 s | 62,6 s |
| `.venv-colab` (Py 3.13, Keras 3.13.2, pandas 2.2.3, mpl 3.10, IPython 7.34) | rápido | 24,6 s | 9,4 s | 1,3 s | 2,8 s | 4,4 s |
| `.venv-colab` | referencias | 96,2 s | 10,1 s | 1,6 s | 7,6 s | 70,0 s |

- **Red por defecto** (56.129 parámetros, 8 épocas): entrenamiento de 5,8-7,5 s con todos los hilos y 8,7 s con 2
  hilos. Estimación previa de 1.3: ~5 s (real 6,6 s) y ~8 s con 2 hilos (real 8,7 s). Plan B estimado en «unos 1,1
  minutos» y ejecutado en 62,6 s.
- **Métricas de la red por defecto** (semilla 42, todos los hilos; idénticas en `.venv` y `.venv-colab`): acierto
  de validación 94,5 %, test 87,7 %, sensibilidad 94,9 %, especificidad 75,6 %, AUC 0,936; diagnóstico «razonable
  (aún mejoraba)». Con 2 hilos: validación 94,9 %, test 86,9 %, sensibilidad 94,9 %, especificidad 73,5 %.
  Con 2 épocas (modo rápido): test 82,9 %, sensibilidad 95,9 %, especificidad 61,1 %, «infraajuste».
- **Plan B** (semilla 42, todos los hilos; mismas métricas en los dos entornos, tiempos del `.venv`):

| Configuración | Parámetros | Entreno | Acierto val. | Test | Sensib. | Especif. | AUC | Diagnóstico |
|---|---|---|---|---|---|---|---|---|
| ref_defecto | 56.129 | 4,3 s | 94,5 % | 87,7 % | 94,9 % | 75,6 % | 0,936 | razonable (aún mejoraba) |
| ref_A_1_bloque | 173.345 | 3,3 s | 95,2 % | 84,1 % | 97,9 % | 61,1 % | 0,924 | razonable (aún mejoraba) |
| ref_A_3_bloques | 27.521 | 5,5 s | 92,6 % | 86,2 % | 94,6 % | 72,2 % | 0,936 | razonable (aún mejoraba) |
| ref_B_sin_pooling | 1.184.577 | 7,9 s | 95,6 % | 86,5 % | 97,9 % | 67,5 % | 0,927 | razonable |
| ref_C_dropout_0_20_epocas | 1.184.577 | 18,0 s | 95,8 % | 87,5 % | 98,2 % | 69,7 % | 0,939 | **sobreajuste** |
| ref_C_dropout_05_20_epocas | 1.184.577 | 17,8 s | 96,0 % | 88,1 % | 97,9 % | 71,8 % | 0,946 | razonable |
| ref_D_compensar_desbalanceo | 56.129 | 4,2 s | 92,6 % | 87,8 % | 93,8 % | 77,8 % | 0,931 | razonable (aún mejoraba) |

  Tendencias que se cumplen aquí (y que **no** están escritas en el notebook): A, 1 bloque tiene 6,3 veces los
  parámetros de 3 bloques; B, sin pooling ×21 parámetros y ~2× tiempo; D, sube la especificidad y baja algo la
  sensibilidad. Con una sola semilla, todo como tendencia.
- **Escenario de alumno** (`escenarios.py`, 36 celdas fuera de orden, 82 s): 1.4 antes de 1.2 → «Primero ejecuta
  la celda 1.2»; reto e hipótesis adjuntos a los experimentos siguientes; diferencia «−145.824 parámetros (6,3
  veces menos)»; aviso de Grad-CAM 3×3 con 3 bloques; índice 700 → error; dos «Normal aleatoria» seguidas dan
  imágenes distintas; aviso de ~4 min (13,5 M); k=5 con 3 bloques → imposible con cuatro soluciones; cambio de
  resolución a 64 → aviso en 1.2 y error explicativo en 1.4 y 1.5; bloqueo de 103 M; red a 64 px entrenada en 20 s
  (test 84,8 %); re-ejecutar 1.1 conserva los 4 experimentos; resolución 32 → error.
- Figuras revisadas a ojo (PNG extraídos a `verificacion/build_nb1/`): diagrama de la ficha (2 y 4 bloques), rejilla
  4×4, reparto de clases, curvas en vivo, matriz de confusión + barras, filtros, mapas de activación (28 y 64 px),
  Grad-CAM (2 bloques, 3 bloques, 64 px y ruta de mapas vacíos) y gráfico del registro. Todo en castellano, con
  letra de 12-16 pt a 100 dpi, Okabe-Ito con trama o marcador para las dos series, `viridis`/`cividis`.

## 6. Sin resolver o para comprobar a mano

- 🟡 **Nada se ha ejecutado en Colab real.** Pendiente de ver allí: formularios y tildes, que las curvas se
  actualicen en la misma figura, `files.download` del CSV, la instalación de `medmnist==3.0.2` en Python 3.13 sin
  reiniciar, y los nombres de menú en castellano que cito en los errores («Entorno de ejecución → Desconectar y
  eliminar entorno de ejecución»). En `.venv-colab` (Python 3.13 + versiones de Colab) todo pasa, pero sin pip.
- 🟡 **Tiempo en Colab**: el defecto entrena en ~6-7 s en local con todos los hilos (ver sección 5 para 2 hilos).
  Con el factor 2-3 de la nota e, Colab debería quedar muy por debajo de 90 s. El Plan B completo, unos 1,5 min en
  local, probablemente 3-5 min en Colab.
- 🟡 **Reto C en Colab**: la semilla es la misma, pero otro número de hilos cambia los decimales y 20 épocas los
  amplifican. Con 3/3 semillas estables en local, lo esperable es que el efecto se mantenga; el guion debería decir
  «deberían separarse» y no «verás sobreajuste».
- En Colab con GPU, TensorFlow puede escribir mensajes de CUDA en stderr al importarse antes de que el nivel de log
  surta efecto. No lo he podido comprobar.
- Mapas de activación del primer bloque a 28 px: varios se parecen (filtros de brillo). Es lo que hay en esta red;
  a 64 px el efecto «bordes» se ve mucho mejor. Puede valer como comentario en clase.
- Con 2 épocas (modo rápido) el diagnóstico dice «infraajuste», correcto para ese caso.
- El «1-2 minutos» de la primera ejecución de 1.1 es prudente y no está medido en Colab (en local, 12-14 s sin
  instalar nada).
- Desviaciones conscientes de la especificación: rótulos del gráfico con `#` (decisión 11); títulos de los mapas
  de activación ajustados a lo que se ve (decisión 13); reto C sin densa 128 (sección 4).

## 7. Correcciones tras auditoría (24-sep-2026)

Auditoría independiente del NB1: 1 hallazgo mayor y 11 menores. Todos corregidos en `src/build_nb1.py` salvo el
último (rótulos con `#`), que sigue pendiente de que Lou lo acepte. Material en `verificacion/fix_nb1/`
(`sondas_fix.py` y `sondas_soluciones.py` son las comprobaciones que se pueden volver a ejecutar).

| Hallazgo | Corrección |
|---|---|
| **Mayor.** Rótulos del gráfico de 1.8/1.10 junto al punto equivocado (el `#6` al lado del punto del `#5`, y el `#7` flotando sin punto) | `_etiquetar` se llama después de `tight_layout`, cuando ya se conoce el alto del panel. Agrupa los puntos cercanos en x (menos de 0,3 décadas). Los marcadores superpuestos (menos de 2 pt) comparten un rótulo, «#5, #7». Los demás rótulos se separan al menos 16 pt y llevan una línea fina hasta su punto. `sondas_fix.py` comprueba que cada `#` está en el punto al que apunta su rótulo |
| Redondeo doble (97,9 % en el progreso y 98,0 % en la tabla) | `ESTADO["experimentos"]` guarda las cifras sin redondear. Se redondea una sola vez: al pintarlas, al escribir la línea para la hoja común (3 decimales, el tiempo con 1) y al escribir el CSV (4 decimales, el tiempo con 1). Ahora el Plan B da 97,9 % en las dos partes |
| `_viable` no miraba el tiempo: proponía «desactiva el pooling» para una red que luego se bloqueaba (838 s) | Nueva cota `_segundos_cota`: multiplicaciones por imagen × imágenes × épocas × `SEGUNDOS_POR_GMAC` = 0,25. En local, las redes grandes miden 0,02-0,06 s por cada 10⁹ multiplicaciones × imagen (`calibrar_macs.py`). El 0,25 es unas 6 veces eso, para cubrir la CPU de Colab (🟡: el factor 6 viene de la nota e, no está medido en Colab). Con 64 px, 4 bloques y k=5 ya solo propone «3 bloques» o «kernel 3». Con 28 px, 4 bloques, 32 filtros y densa 128 ya no propone quitar el pooling (en local, 188-218 s; la cota da 2.081 s). `sondas_soluciones.py` aplica cada solución propuesta en 4 casos y todas pasan 1.3 |
| Filtros y mapas de 1.6 con escala automática por imagen | Filtros: una sola escala simétrica ±máx\|w\| para los 16 (gris medio = 0). Mapas: `vmin=0` y cada mapa con su propio máximo, así que morado significa 0 de verdad. Probé también una escala común por bloque, pero aplanaba los mapas más débiles y dejaba de verse dónde responde cada filtro. Leyendas nuevas: «gris medio = cero» y «amarillo = donde más responde ese filtro». Con la escala honesta se ve que 6 de los 8 mapas del bloque 1 son de brillo (responden en toda la imagen), como ya decía la decisión 13 |
| «la validación no empeora al final» cuando la pérdida había subido un 1,6 % | Ahora dice «la pérdida de validación apenas ha subido desde su mínimo (menos de un 10 %)» |
| Origen de un único centro presentado como hecho (🟡 en `f_fuentes`) | «Según los autores del conjunto original… no debería darse aquí», y la pregunta 1 lleva «según sus autores». En 1.3, el dropout es «para que memorice menos» |
| No se avisaba de que un solo entrenamiento tiene ruido | Nueva línea bajo el gráfico de 1.8/1.10: la especificidad de la red por defecto se movió más de 15 puntos solo por la semilla (75,6 % con s42 frente a 58,5 % con s1 en `calib_diag.jsonl`; en la fase 1 fue de 49,6 % a 72,7 %). Una diferencia de pocos puntos puede ser azar. La pregunta 1 de 1.11 pregunta si las diferencias son grandes o de pocos puntos |
| «1-2 minutos» en 1.1 sin medir | «la primera vez tarda algo más» |
| «Acierto val.» junto a «Accuracy test» | Pasa a «Acierto (validación)» / «Acierto (test)» en la tabla, «Acierto (test)» en la tarjeta de 1.5, «Acierto» en las barras y «acierto (test)» en el progreso del Plan B. «accuracy» solo aparece entre paréntesis en 1.4, la primera vez. Los nombres de columna del CSV y de la línea para la hoja común no cambian: son contrato |
| «se muestran todo lo que hay» | «se muestra todo lo que hay» |
| Cambiar solo `compensar_desbalanceo` o `epocas` y volver a ejecutar 1.3 decía «Primera red de la sesión» | El diseño incluye ahora las épocas y la compensación. Si solo cambian esas dos: «Misma arquitectura que la red anterior (cambian las épocas o la compensación del desbalanceo)». Volver a ejecutar 1.3 con lo mismo sigue dando el mismo mensaje |
| Rótulos con `#` en lugar del nombre | **Sin cambiar.** Es la desviación consciente de la decisión 11: los nombres del Plan B no caben, y en clase casi todos los experimentos de un grupo se llaman igual. El otro problema, el de los rótulos mal colocados, ya está resuelto, así que cada `#` lleva a su fila. **Pendiente: Lou tiene que aceptarlo de forma expresa** |

**Verificación tras las correcciones** (generador determinista, mismo SHA-1 en dos ejecuciones). `verificar.py --nb 1`
con `rapido`, `defecto`, `imposible` y `referencias` en `.venv`: 0 errores, 0 líneas en stderr y ningún problema de
estructura. En el modo imposible sigue saliendo el aviso de 0×0. El último informe es `verificacion/informe_20260924-023005.md`.
Las métricas del Plan B no cambian respecto a la sección 5. En `.venv-colab` también pasan `rapido` y `referencias`,
y el gráfico sale igual con matplotlib 3.10. PNG revisados a ojo: gráfico del Plan B, filtros, mapas de activación,
barras de 1.5 y gráfico de un solo punto.

- 🟡 `SEGUNDOS_POR_GMAC` sale de multiplicar la medida local por un factor estimado para Colab. En un ordenador
  rápido puede dejar fuera alguna solución que sí funcionaría (nunca propone una que se bloquee en local: en local,
  la cota queda entre 1,6 y 11 veces por encima de la estimación medida). Si en Colab 1.3 bloquea por tiempo una
  solución que ha propuesto, hay que subir la constante.
- Las carpetas `verificacion/rapido/` y `verificacion/referencias/` guardan ahora la ejecución de `.venv`, que fue la última.

## 8. Correcciones tras las revisiones adversariales (24-sep-2026)

Dos revisiones adversariales del NB1 (lente de alumna y lente técnica) y un escéptico que verificó cada hallazgo.
Cuando el hallazgo y el escéptico no coincidían, se ha aplicado el arreglo del escéptico. Solo se ha tocado
`src/build_nb1.py`. Material en `verificacion/fix2_nb1/`: `sondas.py` (17 escenarios de alumno sobre el código
real: 1.3 cambiada sin entrenar, 1.3 imposible con y sin modelo, bloqueo por tiempo, descarga en Colab simulada,
Plan B dos veces), `t_gradcam_formas.py` (Grad-CAM con 3 bloques, 1 bloque sin pooling, 64 px y mapa vacío),
`t_mapas64.py`, `volcar.py`, los volcados de texto y los PNG.

**Aplicado**

| Dónde | Cambio |
|---|---|
| Portada | Título «Cuaderno 1»; «Cada uno prueba…» con hoja común; aviso condicional de Colab («Si al pulsar ▶ por primera vez Colab te avisa… pulsa **Ejecutar de todos modos**»; el rótulo exacto es 🟡, nota e); «hasta la 1.9»; «Cambios: solo se pasa el brillo… no se recorta ni se retoca ninguna imagen»; «Son radiografías reales, publicadas por sus autores con licencia abierta (CC BY 4.0)» (sin «anonimizadas», que no está comprobado) |
| 1.1 | Texto de la semilla: «en el mismo entorno, repetir un entrenamiento dé lo mismo». Rótulo sobre la tabla de versiones («no hace falta que las mires»). Salida final: «Tarjeta gráfica (GPU): no, y no hace falta · Semilla: 42». Se quitan procesador y núcleos (el brief no los pide) |
| 1.2 | Validación: «imágenes que la red no usa para aprender; sirven para comprobar, mientras entrena, si aprende de verdad o solo memoriza». Figura: «Normales y neumonías en cada conjunto» |
| 1.3 tabla | Glosas nuevas de `tamano_kernel` (lado del filtro, «el kernel de CNN Explainer»), `usar_batchnorm`, `dropout` (con «en el deslizador sale 0.2»), `neuronas_capa_densa`, `compensar_desbalanceo` («cuenta unas 3 veces más», 1,94/0,67 ≈ 2,9) y `nombre_experimento` (iniciales + reto, «ALM_A»). Línea «Red por defecto» con los nombres reales de los controles (la misma cadena, `RED_DEFECTO`, se inserta en 1.3 y 1.9) |
| 1.3 salida | Línea bajo la tabla de capas (forma de salida, aplanar, logit, sigmoide). Tiempos: «en la máquina de Colab» si `EN_COLAB`, «en este ordenador» en local; minutos solo desde 120 s; «sin contar el dibujo de las curvas». Bloqueo por tiempo con soluciones a medida: «activa el pooling» solo si está desactivado, «baja la resolución a 28» solo a 64 px, siempre «baja las épocas (ahora N)» y «usa menos filtros o neuronas». Aviso de Grad-CAM: «tendrá solo N×N cuadros» |
| 1.4 | Markdown con **colapso** y «falla más con radiografías nuevas». Salida: diagnóstico + una línea de reglas visible (spec) + cifras del entrenamiento dentro de `<details>`. Las reglas dicen que la pérdida de entrenamiento es la media de la época con el dropout activo. «Fíjate» corregido: «con la red por defecto suele pasar en todas las épocas» (comprobado en `def_08_4.png`: 8 de 8), y la frase de la compensación solo si se compensa. El tiempo registrado es la suma de los tiempos de época del callback, sin el dibujo: la red por defecto sale 4,1-4,3 s en 1.4 frente a 4,0-4,1 s en el Plan B (antes 4,1 frente a 3,0) |
| 1.5 | Nota de rigor al markdown. Tarjeta AUC: «elegidas al azar una neumonía y una normal, probabilidad de que la red puntúe más alto la neumonía». Recuadro clínico en 3 líneas, con «lo decide el equipo clínico». Línea validación/test dinámica: calcula sensibilidad y especificidad en validación y nombra la clase en la que más cae («especificidad del 88,1 % en validación frente al 75,6 % en test» con la red por defecto); ya no atribuye la diferencia a la prevalencia ni usa el paréntesis «el de validación del estudio original». Cierra con «Siguiente paso: 1.6. Si estás haciendo un reto, ejecuta también 1.8…» |
| `modelo_listo` (1.5-1.7) | Primera línea «Resultados del experimento n.º X («nombre»)». Aviso si 1.3 es inválida o si el diseño de 1.3 (`diseno()`, compartida con 1.3) difiere del entrenado. Sin modelo y con 1.3 inválida: «corrígela… y ejecuta 1.3 y 1.4» en lugar de «ejecuta 1.4» |
| 1.6 | Markdown del índice (0-623, «Test n.º» del título, se ignora con las opciones aleatorias). Título «Test n.º i · Real: X / Predicción: Y · NN %». «Cuadrícula de 3×3 números aprendidos (pesos)». Título del último bloque suavizado («en esta red pequeña, muchos filtros responden sobre todo al brillo») y nota nueva de los mapas. `t_mapas64.py`: también a 64 px muchos mapas se parecen al brillo (bloque 2: r 0,63-0,95), así que la nota no dice «a 28 px» |
| 1.7 | Markdown llano (sin gradientes ni paréntesis). Figura en pares «Radiografía» / «Dónde miró», con real y predicción encima de cada par. Porcentaje con el mismo formato que 1.6 y «> 99 %» en lugar de «100 %»; línea «El porcentaje es la probabilidad que da la red a su respuesta…». **Geometría**: cada casilla de la última ReLU se pinta sobre su centro real (recorrido de capas: la convolución suma (k−1)/2·paso y el pooling suma paso/2 y duplica el paso; por defecto centro 3,5 y paso 2) con `imshow(extent=…, interpolation="bilinear")`; ya no se estira con `tf.image.resize`. El recuadro lo explica: «el marco sin color es la zona que esa capa no cubre». Revisado a 2 y 3 bloques, 1 bloque sin pooling, 64 px y mapa vacío |
| Zech | «En 3 de 5 comparaciones, los modelos rendían peor fuera del hospital…» (f_fuentes §1) |
| 1.8 / 1.10 gráfico | Eje x «Parámetros de la red / (cada marca vale 10 veces más que la anterior)» en dos líneas (en una se pisaban los dos paneles); marcas «1 millón», «10 millones»; «Siempre «Neumonía»: 0 % (muy por debajo, fuera del gráfico)». Nota de ruido: «no saques conclusiones de ella. Fíjate solo en las diferencias grandes» (sin «repetir con compañeros»: misma semilla y misma máquina dan las mismas cifras) |
| 1.8 | «Copia solo la segunda línea… La primera son los nombres de las columnas, que ya están en la hoja». En Colab el CSV solo se descarga si el registro ha cambiado desde la última descarga (`ESTADO["descargado"]`), y se dice qué es («tu tabla con todos tus experimentos… Para la hoja común basta con la línea de arriba») |
| 1.9 | Trabajo individual: «El profesor dirá en clase qué reto te toca», iniciales + reto en `nombre_experimento`, 1.3 → 1.4 → 1.5 → 1.8 en cada entrenamiento, dos líneas en A y C, enlace a la hoja en el chat. Red por defecto escrita. Reto C «(sin pooling, 20 épocas)» en título y desplegable (`_guardar_reto` corta en « ·», sigue dando «C»). La confirmación añade «Comprueba que es el reto que te ha dado el profesor» |
| 1.10 | «Plan B (solo el profesor)» + «**No ejecutes esta celda.**» (markdown, título de la celda e índice). La tabla del ensayo ya no enseña parámetros. Tiempo total «unos 60 s en este ordenador» / «en la máquina de Colab». Plan B repetido: si el modelo actual ya es un `ref_defecto`, se sustituye; antes 1.5 volvía a añadir la fila antigua (comprobado en `sondas.py`: una sola fila `ref_defecto`) |
| 1.11 | «Puente al cuaderno 2», «sobre todo convolucional» y glosa de «bloques de atención». Sin nombrar ChatGPT (regla de herramientas) y manteniendo «la tuya por defecto» |

**No aplicado (y por qué)**

| Hallazgo | Motivo |
|---|---|
| Renombrar `tamano_kernel` → `lado_del_filtro` y la columna `kernel` → `filtro_lado` | Contrato del brief y de `verificar.py`. Solo cambia la prosa |
| Opción «Número concreto» en 1.6 | La especificación fija las opciones y `_que_ve` compara el texto. Se explica en el markdown |
| «desequilibrados» en 1.2 | Se mantiene «desbalanceados» en negrita para que enlace con `compensar_desbalanceo` |
| Cabecera en castellano con tildes en la línea para la hoja común | Opcional según el escéptico. La cabecera sale de `COLUMNAS` (contrato del CSV); el profesor la copia una vez a la hoja. Dos vocabularios para lo mismo confundirían más |
| Opción «— Elige tu reto —» por defecto en 1.9 | Se desvía de la especificación y exigiría registrarlo en `DECISIONS.md`, que este encargo no puede tocar. Se aplica la alternativa del escéptico (confirmación con «Comprueba que es el reto…») |
| Casilla `descargar_tabla` | El brief pide la descarga en Colab. Se evita la repetición con la firma del registro |
| Diagnóstico con pérdidas en modo inferencia y umbral recalibrado (~1,9-2,0) | El escéptico refuta la consecuencia: el contraste del reto C se mantiene en 3/3 semillas. Se aplica el arreglo ligero (las reglas dicen cómo se mide la pérdida de entrenamiento). Recalibrar movería el diagnóstico de todo el Plan B sin necesidad |
| Selección de canales sin duplicados (\|r\| < 0,9) | La especificación fija «los 8 canales con mayor activación media». Solo se suavizan título y nota |
| `tf.config.experimental.enable_op_determinism()` con GPU | Sin GPU aquí para probarlo y puede lanzar `UnimplementedError`. Solo cambia el texto de la semilla |
| Actualizar D18 en `DECISIONS.md` y el 53 % de la nota b | Fuera de los ficheros de este encargo. Dato corregido para quien los actualice: con la arquitectura final y semilla 42, el mapa de «Normal» sale vacío en 2 de 197 predicciones (1 %), según el escéptico con `verificacion/rev_nb1/t_gradcam.py`; el 53 % es de la arquitectura de la fase 1 |
| «anonimizadas» en la portada | 🟡 sin comprobar en Kermany et al. 2018. Si Lou lo confirma, se añade |

**Verificación.** Generador determinista (mismo SHA-1 en dos ejecuciones: `5063d6cd…`). `verificar.py --nb 1` con
`rapido`, `defecto`, `imposible` y `referencias` en `.venv`: 0 errores, 0 líneas en stderr, sin problemas de
estructura, aviso de 0×0 detectado (informe `verificacion/informe_20260924-111111.md`). `.venv-colab` `rapido`:
0 errores, 0 stderr (`informe_20260924-111203.md`). Métricas del Plan B idénticas a la sección 5 (los tiempos
bajan algo porque ya no incluyen el dibujo). `sondas.py`: 17/17. PNG revisados a ojo: Grad-CAM en pares (2 y 3
bloques, 64 px, mapa vacío; también con matplotlib 3.10), gráfico del Plan B, gráfico de un punto, mapas de
activación y título de 1.6.

**Pendiente para Lou**
- Poner el enlace de la hoja común en el chat de la clase (1.9 lo da por hecho) y crear la hoja con la cabecera
  de la línea de 1.8.
- 🟡 Comprobar en Colab real el rótulo del botón del aviso («Ejecutar de todos modos») y que `<details>` se
  despliega en la salida de 1.4.
- Rótulos con `#` en el gráfico (decisión 11) siguen pendientes de su visto bueno.
- Las carpetas `verificacion/rapido/` y `verificacion/defecto/` quedan con la última ejecución: `rapido` es la de
  `.venv-colab`.
