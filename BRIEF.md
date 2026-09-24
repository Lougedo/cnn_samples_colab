# BRIEF — Última sesión: «Una CNN por dentro» + «Contar personas en vídeo»

## 1. El encargo

Construye **dos notebooks de Google Colab** y su **material docente** para la segunda hora (60 min) de la última sesión de una asignatura del Máster en IA de UNIR. Entrega todo en este repositorio, verificado y listo para usar en clase.

**Regla de oro:** no preguntes nada durante la ejecución. Ante cualquier ambigüedad, elige la opción más sencilla para un alumnado no técnico, documéntala en `DECISIONS.md` con su motivo y sigue.

Trabaja en la carpeta actual. Si no es un repositorio git, inicialízalo antes de empezar.

## 2. Contexto

- **Alumnado:** mayoritariamente no técnico. Saben abrir Colab y ejecutar celdas (lo usan desde la sesión 1). Conocen n8n como herramienta de automatización.
- **Restricción de coordinación:** máximo 15 minutos de teoría por sesión. El material debe enseñar *haciendo*: cada explicación son 2-4 líneas junto al resultado que explica, nunca bloques largos de texto.
- **Agenda de la hora:**
  - 0–10 min: intuición con CNN Explainer (web externa, no hay que construir nada).
  - 10–35 min: laboratorio de arquitectura por grupos (Notebook 1).
  - 35–45 min: Grad-CAM y debate sobre dónde mira el modelo (Notebook 1).
  - 45–60 min: demo guiada de conteo de personas en vídeo (Notebook 2) y puente opcional a n8n.

**Decisiones ya tomadas (no reabrir):**

- Colab con una capa visual encima. No usar KNIME. No usar n8n ni Make para entrenar.
- Dataset de imagen: **PneumoniaMNIST** (MedMNIST).
- Vídeo: **Ultralytics YOLO26** preentrenado, con sus *solutions* `RegionCounter` y `ObjectCounter`. No se entrena ni se hace fine-tuning de YOLO.
- n8n solo como epílogo opcional: el Notebook 2 puede enviar agregados a un webhook.

## 3. Estructura del repositorio

```
cnn-sesion-final/
├── BRIEF.md
├── README.md
├── DECISIONS.md
├── INFORME_FINAL.md
├── notebooks/
│   ├── 01_laboratorio_cnn_radiografias.ipynb
│   └── 02_conteo_personas_cafeteria.ipynb
├── src/
│   ├── build_nb1.py            # genera el notebook 1 con nbformat
│   └── build_nb2.py            # genera el notebook 2 con nbformat
├── scripts/
│   └── verificar.py            # ejecuta ambos notebooks headless y resume resultados
├── docs/
│   ├── guion_docente_60min.md
│   ├── hoja_alumno.md
│   └── notas_verificacion/     # hallazgos de la fase 1 sobre APIs y versiones
├── requirements-verificacion.txt
└── .gitignore                  # datos, pesos, vídeos, salidas y entornos fuera de git
```

**Generación de los notebooks.** Genéralos desde `src/build_nb*.py` con `nbformat`, no editando JSON a mano. Así los metadatos de Colab (`cellView`) son deterministas y los cambios se revisan en Python. Los `.ipynb` finales se guardan sin salidas.

## 4. Requisitos transversales (ambos notebooks)

1. **Idioma.** Todo lo que ve el alumno va en español de España: markdown, títulos de formulario, `print`, ejes, leyendas y mensajes de error. Los nombres de variables pueden ir en español si mejoran la lectura.
2. **Colab-first.** Cada notebook se ejecuta de arriba abajo en Colab gratuito (CPU) sin tocar código. Si hay GPU, se detecta y se usa; si no, todo sigue funcionando.
3. **Vista de formulario.** Toda la configuración se hace con `#@param` y `#@title`. Las celdas de código técnico llevan metadata `"cellView": "form"`, de modo que el alumno solo ve títulos, controles y resultados.
4. **Ejecutable también fuera de Colab.** Los notebooks deben correr en Jupyter local para poder verificarlos. Los `#@param` son simples comentarios fuera de Colab, y cualquier `import google.colab` va protegido con `try/except`.
5. **Re-ejecutable.** Cada celda es idempotente: se puede volver a entrenar o reprocesar sin reiniciar el entorno. El estado global está controlado y es explícito.
6. **Errores en castellano llano.** Cada error dice qué pasa y qué hacer. Ejemplo: «Con 4 bloques y pooling la imagen se queda en 0×0 píxeles: quita un bloque o sube la resolución a 64».
7. **Legible en proyector.** Fuentes de 12–14 pt como mínimo, paletas aptas para daltonismo (viridis o cividis) y nada de 3D gratuito.
8. **Presupuesto de tiempo.**
   - Notebook 1: la configuración por defecto entrena en 90 s o menos en la CPU de Colab.
   - Notebook 2: el tramo de vídeo por defecto se procesa en 2 min o menos en CPU.
   - Mide los tiempos reales en local (CPU) y repórtalos en `INFORME_FINAL.md`.
9. **Reproducibilidad.** Semillas fijadas, versiones mínimas declaradas y versiones impresas al inicio de cada notebook.
10. **Instalación robusta.** La celda de preparación funciona tanto si Colab trae TensorFlow preinstalado como si no. Emula en local una versión de Python cercana a la de Colab (3.11–3.12).
11. **Git.** Nada de datos, pesos ni vídeos en el repo. No crees remotos, no hagas push y no pidas credenciales. Los badges «Open in Colab» usan el placeholder `<USUARIO>/<REPO>`.
12. **Licencias y avisos visibles.**
    - PneumoniaMNIST: CC BY 4.0, *no apto para uso clínico*.
    - Ultralytics: AGPL-3.0.
    - Vídeos: licencia de su fuente.
13. **Afirmaciones técnicas.** Cualquier afirmación sobre CNN, datasets o métricas debe ser correcta y prudente. Explica las tendencias como tendencias, no como garantías.

## 5. Notebook 1 — `01_laboratorio_cnn_radiografias.ipynb`

**Objetivo pedagógico.** El alumno cambia la arquitectura sin programar y ve cuatro cosas:

- cómo cambia la red;
- qué «ve» cada capa;
- qué le pasa al rendimiento;
- dónde mira el modelo para decidir.

**Stack**

- Keras 3 con backend TensorFlow: fija `os.environ["KERAS_BACKEND"] = "tensorflow"` antes de importar.
- `medmnist` para los datos.
- `visualkeras` para el diagrama. Si falla con la versión instalada, usa un diagrama propio con matplotlib (bloques apilados con dimensiones y nº de filtros) y documéntalo.
- `scikit-learn` para las métricas.
- **API Funcional** con `Input` explícito y capas con nombre, necesaria para extraer feature maps y calcular Grad-CAM sin los problemas de `Sequential` en Keras 3.

**Secciones**

**1.0 Portada.**
- Qué vamos a hacer, en 3-4 líneas.
- Aviso de «no apto para uso clínico».
- Cómo usar los formularios, con una captura descrita en texto.
- Índice.

**1.1 Preparación.**
- Instalación silenciosa.
- Versiones, GPU sí/no y semillas.

**1.2 Los datos.**
- Formulario `resolucion` [28, 64], por defecto 28. Descarga cacheada.
- Rejilla 4×4 de ejemplos etiquetados «Normal / Neumonía».
- Gráfico de distribución de clases con porcentajes y una frase sobre el desbalanceo.
- **Referencia tonta:** accuracy en test de responder siempre «Neumonía».

**1.3 Diseña tu red (formulario).**

| Control | Valores | Por defecto |
|---|---|---|
| `bloques_convolucionales` | slider 1–4 | 2 |
| `filtros_primer_bloque` | 8 / 16 / 32 | 16 |
| `duplicar_filtros_en_cada_bloque` | bool | True |
| `tamano_kernel` | 3 / 5 | 3 |
| `usar_pooling` | bool | True |
| `usar_batchnorm` | bool | False |
| `dropout` | slider 0–0.6, paso 0.1 | 0.2 |
| `neuronas_capa_densa` | 0 / 32 / 64 / 128 | 64 |
| `compensar_desbalanceo` | bool (`class_weight`) | False |
| `epocas` | slider 3–20 | 8 |
| `nombre_experimento` | texto | "grupo_1" |

Antes de construir la red:

- Valida las configuraciones imposibles: dimensión espacial menor que 1, o parámetros por encima de un umbral razonable (fíjalo y documéntalo).
- En el caso de explosión de parámetros al quitar el pooling, avisa del tiempo estimado en lugar de bloquear, salvo que sea inviable.

Salida, la **«ficha del modelo»**:

- diagrama de la arquitectura con leyenda;
- tabla por capa en castellano (tipo, forma de salida, parámetros);
- total de parámetros y tamaño en MB;
- diferencia respecto a la red anterior de la sesión («+X parámetros»).

**1.4 Entrenar.**
- Curvas en vivo por época (pérdida y accuracy, entrenamiento y validación) que se actualizan en la misma figura.
- Tiempo por época.
- Al terminar, un **diagnóstico automático en llano** con reglas simples y documentadas: sobreajuste, infraajuste o razonable.

**1.5 Evaluar en test.**
- Accuracy, **sensibilidad** (recall de Neumonía), especificidad y AUC.
- Matriz de confusión con etiquetas en castellano y conteos.
- Comparación con la referencia tonta.
- Tres líneas sobre por qué en salud importa la sensibilidad.

**1.6 Qué ve cada capa.**
- Formulario para elegir la imagen: neumonía aleatoria, normal aleatoria o índice concreto.
- La imagen se muestra ampliada.
- Filtros aprendidos de la primera capa.
- Hasta 8 feature maps por bloque, con títulos que expliquen la tendencia: de bordes y contrastes a patrones más abstractos.

**1.7 Dónde mira el modelo (Grad-CAM).**
- Calculado sobre la última capa convolucional.
- 6 imágenes: 3 aciertos y 3 fallos (si los hay), con el mapa de calor superpuesto, la predicción y su probabilidad.
- Recuadro de debate sobre el **aprendizaje por atajos**. Ejemplo documentado: modelos de neumonía que aprendían rasgos propios del hospital de origen (Zech et al., 2018, *PLOS Medicine*). Redáctalo sin exagerar.

**1.8 Registro de experimentos.**
- Cada entrenamiento se añade a una tabla de sesión: configuración, métricas, tiempo y parámetros.
- Gráfico comparativo, por ejemplo parámetros frente a sensibilidad.
- **Línea resumen** lista para copiar a la tabla común de la clase.
- Exportación a CSV: descarga en Colab, fichero local fuera de Colab.

**1.9 Retos guiados.** Cada reto pide anotar una hipótesis antes de ejecutar.
- A: profundidad, 1 frente a 3 bloques.
- B: quitar el pooling y ver la explosión de parámetros.
- C: dropout 0 frente a 0.5 con 20 épocas, para ver el sobreajuste.
- D: compensar el desbalanceo y ver cómo cambian sensibilidad y especificidad.

**1.10 Plan B del profesor (opcional).**
- Celda que entrena en secuencia las 4 configuraciones de referencia de los retos y genera la tabla comparativa.
- Indica su tiempo estimado.

**1.11 Cierre.**
- Tres preguntas de reflexión.
- Puente al Notebook 2: el detector de personas usa un *backbone* CNN como el vuestro, mucho más grande y ya entrenado.

## 6. Notebook 2 — `02_conteo_personas_cafeteria.ipynb`

**Objetivo pedagógico.** Ver una CNN preentrenada aplicada a vídeo sin entrenar nada: detectar, seguir, contar por zonas y por línea, y exportar solo agregados.

**Stack**

- `ultralytics` con YOLO26 nano por defecto (verifica el nombre exacto de los pesos) y las *solutions* `RegionCounter` y `ObjectCounter`.
- `opencv` para leer el vídeo.
- `ffmpeg` para recodificar a H.264 (`libx264`, `yuv420p`, máximo 720p) y mostrar el vídeo inline.
- `supervision` solo para descargar vídeos de demo.
- **Verifica las APIs contra la versión instalada; no las supongas.**

**Secciones**

**2.0 Portada y aviso ético.**
- Contar no es identificar, pero grabar personas es tratamiento de datos personales.
- No subas vídeos con personas reconocibles sin base legal.
- Del notebook solo salen agregados.
- Tono divulgativo, sin asesoramiento jurídico.

**2.1 Preparación.**
- Instalación, versiones, GPU y descarga de pesos.

**2.2 Elegir el vídeo.**
- Formulario `fuente_video`:
  - demo «personas caminando» de supervision;
  - otra demo con personas en interior o comercio, si existe en la versión instalada (lista los assets disponibles y elige);
  - «subir mi vídeo».
- `segundos_a_procesar`: 5–30, por defecto 10.
- `saltar_fotogramas`: 1–5, por defecto 2.
- Muestra el primer fotograma y los metadatos (resolución, fps, duración).
- Nota para el profesor: cómo conseguir un clip de cafetería con licencia libre (por ejemplo, Pexels) y subirlo.

**2.3 Qué ve el detector en un fotograma.**
- Formulario `confianza_minima`: 0.1–0.9, por defecto 0.35.
- Cajas solo para personas (clase 0) y recuento.
- **Mini-gráfico del recuento frente al umbral** en ese mismo fotograma, para ver el compromiso entre falsos positivos y falsos negativos.
- Explicación de 3 líneas: backbone CNN más cabezas de detección, y relación con el Notebook 1.

**2.4 Definir zonas sin dibujar.**
- Primer fotograma con una **rejilla porcentual** (0–100 % en ambos ejes) superpuesta.
- Zonas definidas en coordenadas **relativas** con texto `x1,y1; x2,y2; …` en %, para que valgan en cualquier resolución.
- Dos presets genéricos, izquierda y derecha, renombrables a «barra» y «mesas», que funcionen en cualquier vídeo.
- Línea de «puerta», también en %.
- Previsualización de zonas y línea antes de procesar.

**2.5 Aforo por zonas (`RegionCounter`).**
- Vídeo anotado mostrado inline.
- Serie temporal de personas por zona.
- Máximo y media por zona.

**2.6 Entradas y salidas por línea (`ObjectCounter`).**
- Vídeo anotado con contadores de entrada y salida.
- Tres líneas sobre por qué hace falta *tracking* (IDs persistentes).
- Errores típicos: oclusiones, cambio de ID y doble conteo.

**2.7 Avanzado (opcional): tiempo de permanencia.**
- Con tracking, segundos por ID dentro de una zona.
- Histograma de tiempos.
- Advertencia sobre cambios de ID.

**2.8 Exportar para automatizar.**
- CSV con tiempo relativo y conteos por zona, más un JSON resumen.
- Celda opcional que hace POST a `webhook_n8n`. El campo viene vacío por defecto, y entonces no envía nada.
- Ejemplo del JSON que recibiría n8n y descripción en texto de un flujo de alerta de aforo. No construyas el flujo.

**2.9 Cierre.**
- Límites del sistema.
- Extrapolación a otros sectores: retail, transporte, eventos, industria.
- Dos o tres preguntas sobre privacidad, en llano.

## 7. Documentación

- **`README.md`**
  - Qué hay en el repo y cómo abrirlo en Colab (desde Drive o desde GitHub con badge).
  - Requisitos y tiempos medidos.
  - Problemas frecuentes con su solución: descarga lenta de Zenodo, sin GPU, el vídeo no se ve, sesión reiniciada.
- **`docs/guion_docente_60min.md`**
  - Guion minuto a minuto según la agenda de la sección 2.
  - Qué mostrar exactamente en CNN Explainer (convolución, ReLU, pooling), con 2-3 preguntas para lanzar.
  - Qué reto asignar a cada grupo y qué resultados esperar, según las **ejecuciones reales de referencia**.
  - Frases clave para cada bloque, Plan B si algo falla, y tabla de resultados de referencia medidos.
- **`docs/hoja_alumno.md`**
  - Una página: pasos, tabla para anotar hipótesis y resultados de su reto, y tres preguntas.
- **`DECISIONS.md`**
  - Cada decisión tomada ante una ambigüedad, con su motivo.
- **`INFORME_FINAL.md`**
  - Qué se ha construido.
  - Resultados de la verificación: tiempos en CPU local, métricas de las 4 configuraciones de referencia y recuentos en el vídeo de demo.
  - Limitaciones conocidas.
  - **Lista de lo que el profesor debe comprobar a mano en Colab** porque no se puede verificar en local: vista de formulario, `cellView`, subida de archivos y reproducción de vídeo.

## 8. Plan de ejecución sugerido

**Fase 1 — Verificación de dependencias (en paralelo).** Una línea por frente; cada agente deja sus notas en `docs/notas_verificacion/`.

- a) `medmnist`: API actual, parámetro de tamaño, tamaño de los splits y ruta de descarga. Si Zenodo falla, busca un mirror documentado y regístralo en `DECISIONS.md`.
- b) Keras 3: `visualkeras`, extracción de feature maps y Grad-CAM con la API Funcional.
- c) Ultralytics: nombre de los pesos YOLO26 nano, firma y resultados de `RegionCounter` y `ObjectCounter`, y cómo filtrar por clase persona.
- d) `supervision`: qué vídeos de demo hay disponibles.
- e) Colab: formato de `#@param`, `#@title`, `"cellView": "form"` y reproducción de vídeo inline.

**Fase 2 — Construcción (en paralelo).**
- `src/build_nb1.py` y `src/build_nb2.py` generan los notebooks.
- `scripts/verificar.py` y `requirements-verificacion.txt`.

**Fase 3 — Verificación y corrección.**
- Ejecuta ambos notebooks headless (papermill o nbconvert) con los parámetros por defecto y con parámetros rápidos (1–2 épocas, 3 s de vídeo).
- Ejecuta las 4 configuraciones de referencia del Notebook 1.
- Corrige y repite hasta que todo pase sin errores ni warnings relevantes.
- Después, **dos revisiones adversariales independientes**:
  1. Un «alumno no técnico» revisa que todo texto visible sea claro, esté en castellano y no use jerga sin explicar.
  2. Un «revisor técnico» revisa métricas, Grad-CAM, lógica de conteo y cada afirmación sobre CNN, datasets y licencias.
- Aplica lo que ambos encuentren y vuelve a verificar.

**Fase 4 — Documentación y cierre.**
- README, guion con los resultados reales, hoja del alumno e `INFORME_FINAL.md`.
- Commits locales con mensajes claros, uno por fase como mínimo.

## 9. Criterios de aceptación

- [ ] Ambos notebooks se ejecutan de principio a fin en local sin errores, con parámetros por defecto y rápidos.
- [ ] Configuración por defecto del Notebook 1: 90 s o menos en CPU; Notebook 2: 2 min o menos en CPU. Tiempos medidos y reportados.
- [ ] Todas las celdas técnicas tienen `"cellView": "form"`; toda la configuración está en formularios.
- [ ] Todo texto visible para el alumno está en español de España.
- [ ] El Notebook 1 muestra diagrama, ficha del modelo, curvas en vivo, métricas con sensibilidad y especificidad, feature maps, Grad-CAM y tabla de experimentos exportable.
- [ ] Las configuraciones imposibles se detectan con un mensaje útil antes de construir la red.
- [ ] El Notebook 2 muestra detección, zonas previsualizadas en %, aforo por zonas, entradas y salidas por línea y exportación de agregados. El webhook no envía nada si está vacío.
- [ ] El vídeo anotado se reproduce inline (H.264).
- [ ] Avisos de licencia, de uso no clínico y de privacidad visibles.
- [ ] El guion docente contiene resultados de referencia **medidos**, no inventados.
- [ ] `INFORME_FINAL.md` incluye la lista de comprobaciones manuales en Colab.
- [ ] Nada de datos, pesos ni vídeos en git.

## 10. Fuera de alcance

- Construir el flujo real de n8n (solo el webhook opcional y su descripción).
- KNIME, Make o cualquier herramienta de entrenamiento distinta de Keras.
- Entrenar o hacer fine-tuning de YOLO.
- Datos médicos distintos de MedMNIST.
- Asesoramiento legal.
