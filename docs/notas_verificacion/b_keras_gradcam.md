# Frente b) Keras 3: arquitectura, visualkeras, feature maps, Grad-CAM, curvas y tiempos

Verificado ejecutando código el 24-sep-2026 con el venv del repo (Python 3.12.7): **tensorflow 2.21.0, keras 3.15.1 (backend tensorflow), visualkeras 0.2.0, Pillow 12.3.0, aggdraw 1.4.1, scikit-learn 1.9.1, matplotlib 3.11.2, numpy 2.5.3**. Los scripts y las figuras están en `verificacion/b_keras_gradcam/` (gitignored). Los resultados en bruto están en `resultados*.jsonl`.

**Aviso sobre los tiempos.** Máquina Apple M3 Pro de 11 núcleos, compartida con otros agentes. La carga media estuvo entre 5 y 10 durante las mediciones, así que los tiempos son ruidosos (±20 %). Todo lo que diga «Colab» sin medición es 🟡.

---

## 0. Resumen para quien construye el NB1

1. **La arquitectura planificada funciona tal cual** (API funcional, `padding="valid"`, capas con nombre). Hay 4 combinaciones imposibles en la rejilla del brief (sección 1).
2. **visualkeras 0.2.0 (la última de PyPI, del 14-oct-2025) FALLA con Keras 3.15.1**: `AttributeError: 'InputLayer' object has no attribute 'output_shape'`. Se arregla con un parche de una línea por capa (sección 2). La leyenda nativa sale en inglés y la fuente por defecto no tiene «ó» ni «×». Hay que usar `text_callable` con DejaVu Sans y dejar el plan B de matplotlib dentro de un `try/except`.
3. **Usa `m.input`, no `m.inputs`, para construir submodelos.** Con `m.inputs` (una lista) Keras 3 lanza `UserWarning: The structure of inputs doesn't match the expected structure` en cada llamada.
4. **Grad-CAM sobre la salida de `relu_N` del último bloque**, nunca sobre `conv_N`. Con BatchNorm, `conv_N` da mapas vacíos en 116 de 200 imágenes y `relu_N` en 0 de 200. **Con la clase «Normal» (−logit), el 53 % de los mapas salen vacíos** en el modelo por defecto. Hay que tratarlo con un mensaje y elegir ejemplos que no estén vacíos (sección 4).
5. **La configuración por defecto entrena en unos 3 s en local (5 s con 2 hilos).** El presupuesto de 90 s en Colab sobra, aunque el factor de Colab es 🟡.
6. **BatchNorm con el `momentum=0.99` por defecto rompe la red en 8 épocas**: val_acc se queda clavada en 0,7424, es decir, siempre responde «Neumonía» (test 0,625, especificidad 0). **Con `momentum=0.9` funciona**: val_acc 0,95-0,97 con 3 semillas. **Usa `BatchNormalization(momentum=0.9)`.**
7. **El reto C tal como está escrito no enseña sobreajuste.** Con la red por defecto, `dropout=0` y 20 épocas, la val_loss sigue bajando hasta la época 19-20 en 3 semillas. Sí se ve sin pooling con densa 128: el mínimo cae en la época 16 y luego sube un 30 %, en unos 19 s en local. Ver sección 5.4.
8. **La semilla reproduce exactamente** si se usan la misma máquina y el mismo número de hilos, también al reentrenar en el mismo kernel. Entre semillas, la variación es grande en especificidad (0,50-0,73). Las diferencias entre grupos pueden ser ruido.
9. **`class_weight` funciona** con etiquetas `(n,1)` (float32 o uint8) y salida sigmoide, sin avisos. Sube la especificidad y baja algo la sensibilidad en las 3 semillas probadas.
10. **La estimación previa infravalora un 10-20 %.** Mide 1 lote de calentamiento más 5 lotes cronometrados y multiplica por 1,2. Umbrales propuestos en la sección 8.

---

## 1. Tamaño espacial y combinaciones imposibles

Fórmula, con conv `valid` de stride 1 y `MaxPooling2D(2)` (que también es `valid`):
`s ← s − k + 1` (si `s < 1`, imposible) y, con pooling, `s ← ⌊s/2⌋` (si `s < 1`, imposible).
La he comprobado contra `layer.output.shape` de Keras en las 32 combinaciones. Coincide en todas. Las imposibles lanzan `ValueError: Computed output size would be zero or negative. Received inputs shape=(None, 4, 4, 32), kernel shape=(5, 5, 32, 64)…`. Hay que capturarlo **antes** de construir, porque el mensaje de Keras no le sirve al alumno.

**Imposibles (todas con pooling):**

| res | bloques | kernel | trayectoria tras cada bloque | Motivo |
|---|---|---|---|---|
| 28 | 3 | 5 | 12 → 4 → ✗ | conv 5×5 sobre 4×4 |
| 28 | 4 | 3 | 13 → 5 → 1 → ✗ | conv 3×3 sobre 1×1 (es el caso del brief) |
| 28 | 4 | 5 | 12 → 4 → ✗ | ya falla en el 3.er bloque |
| 64 | 4 | 5 | 30 → 13 → 4 → ✗ | conv 5×5 sobre 4×4 |

Caso límite posible: 28 px, 3 bloques, k=3, con pooling. Tras el último pooling queda en 1×1, y `relu_3` es 3×3, así que el Grad-CAM saldrá de 3×3 y muy grueso. Merece un aviso suave.

**Parámetros (filtros 16 duplicando + densa 64 / filtros 32 duplicando + densa 128), k=3:**

| res | bloques | pooling | f16·d64 | f32·d128 |
|---|---|---|---|---|
| 28 | 1 | sí | 173.345 | 692.801 |
| 28 | 2 | sí (**defecto**) | **56.129** | 223.873 |
| 28 | 3 | sí | 27.521 | 109.313 |
| 28 | 2 | no | 1.184.577 | 4.737.665 |
| 28 | 4 | no | 3.374.081 | 13.495.297 |
| 64 | 2 | sí | 406.337 | 1.624.705 |
| 64 | 4 | sí | 130.049 | 519.169 |
| 64 | 1 | no | 3.936.545 | 15.745.601 |
| 64 | 2 | no | 7.377.729 | 29.510.273 |
| 64 | 3 | no | 13.802.369 | 55.208.705 |
| 64 | 4 | no | 25.787.393 | **103.148.545** (393 MB en float32) |

Con pooling, más bloques supone **menos** parámetros: la capa densa tras aplanar domina. Es una observación buena para el reto A (1 bloque tiene 173 k; 3 bloques, 27 k). La tabla completa de las 32 combinaciones está en la salida de `v1_arquitectura.py`.

```python
def tamanos(res, bloques, kernel, pooling):
    s, tray = res, []
    for _ in range(bloques):
        s = s - kernel + 1
        if s < 1: return None, tray
        if pooling:
            s //= 2
            if s < 1: return None, tray
        tray.append(s)
    return s, tray
# Mensaje: f"Con {bloques} bloques, kernel {k} y pooling la imagen se queda en 0×0 píxeles "
#          f"(tras cada bloque: {' → '.join(map(str, tray))}). Quita un bloque, usa kernel 3 o sube la resolución a 64."
```

## 2. Construcción, tabla por capa y tamaño

```python
import os; os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras import layers

def construir(res=28, bloques=2, filtros=16, duplicar=True, kernel=3, pooling=True,
              batchnorm=False, dropout=0.2, densa=64):
    x = entrada = keras.Input(shape=(res, res, 1), name="entrada")
    f = filtros
    for i in range(1, bloques + 1):
        x = layers.Conv2D(f, kernel, padding="valid", name=f"conv_{i}")(x)
        if batchnorm:
            x = layers.BatchNormalization(momentum=0.9, name=f"bn_{i}")(x)   # ¡0.9, no 0.99! (ver §5.3)
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        if pooling:
            x = layers.MaxPooling2D(2, name=f"pool_{i}")(x)
        if duplicar: f *= 2
    x = layers.Flatten(name="aplanar")(x)
    if dropout > 0: x = layers.Dropout(dropout, name="dropout")(x)
    if densa: x = layers.Dense(densa, activation="relu", name="densa")(x)
    x = layers.Dense(1, name="logit")(x)
    return keras.Model(entrada, layers.Activation("sigmoid", name="probabilidad")(x))
```

- `model.summary()` funciona en Keras 3.15.1 (tabla rich en inglés). Con `print_fn`, la llamada es **una sola**, con toda la tabla. Para la ficha en castellano, conviene montarla a mano:

```python
TIPO = {"InputLayer": "Entrada", "Conv2D": "Convolución", "BatchNormalization": "Normalización (BN)",
        "Activation": "Activación", "MaxPooling2D": "Pooling (máx.)", "Flatten": "Aplanar",
        "Dropout": "Dropout", "Dense": "Densa"}
filas = [dict(capa=l.name, tipo=TIPO.get(l.__class__.__name__, l.__class__.__name__),
              salida=" × ".join(str(d) for d in l.output.shape[1:]), parametros=l.count_params())
         for l in m.layers]
total = m.count_params(); mb = total * 4 / 1024**2     # float32; coincide con summary (219,25 KB)
```

- `l.output.shape` funciona en todas las capas, `InputLayer` incluida: da `(None, 28, 28, 1)`. **Las capas no tienen `output_shape` en Keras 3**; solo lo tiene el `Model`.
- Con BN, `count_params()` incluye los no entrenables (media y varianza móviles: 96 en el defecto con BN). Para distinguirlos: `sum(w.numpy().size for w in m.non_trainable_weights)`.

## 3. visualkeras 0.2.0

**Falla tal cual.** `layered_view` lee `layer.output_shape` (`visualkeras/layered.py:290`), un atributo que las capas de Keras 3 no tienen:
`AttributeError: 'InputLayer' object has no attribute 'output_shape'` (y `'Conv2D'…` si se ignora `InputLayer`). Es el problema de Keras 3 documentado en [keras-team/keras#19052](https://github.com/keras-team/keras/issues/19052). La versión 0.2.0 es la última publicada ([PyPI](https://pypi.org/project/visualkeras/)).

**Parche verificado** (no afecta a `predict`, `summary` ni `fit`):

```python
for l in m.layers:
    if not hasattr(l, "output_shape"):
        object.__setattr__(l, "output_shape", tuple(l.output.shape))
```

Con el parche funciona en los 3 modelos probados (defecto; 64 px, 4 bloques y BN; 1 bloque sin pooling):

- Devuelve un `PIL.Image.Image` en modo RGBA. Se muestra con `display(img)` y se guarda con `img.save(...)`.
- `legend=True` funciona, pero **la leyenda usa el nombre de clase en inglés** (`Conv2D`, `MaxPooling2D`…) y no se puede traducir.
- `show_dimension=True` pone la forma en la leyenda con formato feo (`Conv2D(['26', '26', '16'])`).
- `type_ignore=[layers.InputLayer, layers.Activation, layers.BatchNormalization, layers.Dropout]` oculta capas.
- Emite `UserWarning: The legend_text_spacing_offset parameter is deprecated` en cada llamada, aunque no se pase. Hay que silenciarlo con `warnings.filterwarnings("ignore", module="visualkeras")`.
- **Fuente:** `ImageFont.load_default(size=14)` (Pillow 12) **no tiene «ó» ni «×»** y salen cuadros. La truetype `DejaVuSans.ttf` no se encuentra por nombre en macOS. Solución portable: la fuente que trae matplotlib, `font_manager.findfont("DejaVu Sans")`.

**Receta recomendada** (verificada: `vk_4bl_64_es.png` y `vk_defecto_es.png`; se lee bien a 14 pt):

```python
import warnings, visualkeras
from PIL import ImageFont
from matplotlib import font_manager
warnings.filterwarnings("ignore", module="visualkeras")
fuente = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 14)
NOMBRE = {"Conv2D": "Convolución", "MaxPooling2D": "Pooling", "Flatten": "Aplanar", "Dense": "Densa"}
def etiqueta(i, capa):   # texto encima/debajo alternando, hace de leyenda en castellano
    return f"{NOMBRE.get(capa.__class__.__name__, capa.__class__.__name__)}\n" + \
           "×".join(str(d) for d in capa.output.shape[1:]), i % 2 == 0
try:
    img = visualkeras.layered_view(m, legend=False, font=fuente, text_callable=etiqueta,
        spacing=40, padding=60, scale_xy=4, scale_z=0.5, max_z=60,
        color_map={layers.Conv2D: {"fill": "#3b528b"}, layers.MaxPooling2D: {"fill": "#21918c"},
                   layers.Flatten: {"fill": "#5ec962"}, layers.Dense: {"fill": "#fde725"}},  # viridis
        type_ignore=[layers.InputLayer, layers.Activation, layers.BatchNormalization, layers.Dropout])
    display(img)
except Exception:
    diagrama_matplotlib(m)   # plan B
```

Defecto conocido: con 64 px, alguna etiqueta de pooling se superpone a la caja grande de la primera convolución. Se lee, pero no queda limpio.

**Plan B con matplotlib (verificado, `mpl_4bl_64.png`):** rectángulos horizontales, alto según el tamaño espacial, ancho según `log2(canales)`, forma `a×b×c` encima y leyenda en castellano con colores viridis. Código en `verificacion/b_keras_gradcam/v5_diagrama_mpl.py` (unas 25 líneas). Sugerencia: escala el alto de forma lineal con `forma[0]/res`, para que se vea más cómo encoge la imagen. **Recomendación:** si se quiere cero riesgo en Colab, usar solo el plan B. Si se usa visualkeras, siempre dentro del `try/except`.

## 4. Feature maps, filtros y Grad-CAM

**Feature maps (verificado):**

```python
activaciones = [l.name for l in m.layers if l.name.startswith("relu_")]
extractor = keras.Model(m.input, [m.get_layer(n).output for n in activaciones])   # m.input, NO m.inputs
mapas = extractor(img[None], training=False)          # lista: (1,26,26,16), (1,11,11,32)…
kernels = m.get_layer("conv_1").get_weights()[0]      # (3,3,1,16) → kernels[:, :, 0, i]
```

- Entre el 22 % y el 23 % de los canales están **apagados** (todo ceros tras ReLU) en cada imagen. Hay 2 de 16 muertos en todas las imágenes de `relu_1` y 4 de 32 en `relu_2`. **Muestra los 8 canales con mayor activación media para esa imagen**, no los 8 primeros, o saldrán cuadros lisos. La figura `fmaps_modelo_3bl.png` muestra el problema.
- Con 3 bloques, `relu_3` es de 3×3: se ve como bloques, y cuadra con la frase «de bordes a patrones abstractos».

**Grad-CAM (verificado en 3 modelos entrenados: defecto, BN con momentum 0,9 y 3 bloques):**

```python
import tensorflow as tf
def gradcam(m, img, clase, capa=None):
    """clase: 1=Neumonía (+logit), 0=Normal (−logit). Devuelve (mapa HxW en [0,1], vacio: bool)."""
    capa = capa or [l.name for l in m.layers if l.name.startswith("relu_")][-1]
    gm = keras.Model(m.input, [m.get_layer(capa).output, m.get_layer("logit").output])
    with tf.GradientTape() as tape:
        A, logit = gm(img[None], training=False)
        objetivo = logit[:, 0] if clase == 1 else -logit[:, 0]
    g = tape.gradient(objetivo, A)
    cam = tf.nn.relu(tf.reduce_sum(A * tf.reduce_mean(g, axis=(1, 2))[:, None, None, :], axis=-1))[0]
    mx = float(tf.reduce_max(cam))
    if mx == 0:
        return np.zeros(img.shape[:2]), True
    cam = tf.image.resize((cam / mx)[..., None], img.shape[:2], method="bilinear")[..., 0]
    return cam.numpy(), False
# Superposición: ax.imshow(img[..., 0], cmap="gray"); ax.imshow(cam, cmap="viridis", alpha=0.5, vmin=0, vmax=1)
```

- No hay NaN en ninguna de las 200 × 3 × 2 pruebas. Tarda unos 6 ms por imagen, aunque se reconstruya `gm` en cada llamada. `cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)` también vale.
- **Qué capa usar:** la salida de la `Activation` del último bloque (`relu_N`, después de BN y ReLU y antes del pooling). Es la «activación de la última capa convolucional» del artículo original. Números medidos (mapas vacíos de 200):

| modelo | `conv_N` (antes de BN/ReLU) | `relu_N` |
|---|---|---|
| defecto | 22 | 30 |
| BN (m=0,9) | **116** | **0** |
| 3 bloques | 0 | 0 |

- **Mapas vacíos con la clase «Normal».** En el modelo por defecto, sobre el test completo (624), hay 101 mapas vacíos, y **todos son predicciones «Normal»**: 101 de 190, un 53 %. Hay 84 aciertos y 17 fallos entre ellos, con una confianza media de 0,75 frente a 0,93 en los no vacíos. Si se usa siempre +logit («indicios de neumonía»), bajan a 14. En el modelo con BN y en el de 3 bloques hay 0 vacíos con la clase predicha. **Depende del modelo entrenado.** Recomendación:
  1. Grad-CAM estándar sobre la clase predicha.
  2. Si sale vacío, mostrar la imagen con el texto «Grad-CAM no encuentra ninguna zona que empuje hacia "Normal"». Da juego en el debate: «Normal» se decide a menudo por **ausencia** de indicios.
  3. Al elegir los 6 ejemplos (3 aciertos y 3 fallos), priorizar los que no estén vacíos y, si se puede, mezclar clases.
- 🟡 Observación en 6 imágenes (`gradcam_modelo_defecto.png`): con «Neumonía», el calor se concentra en la franja central (mediastino y columna) más que en los campos pulmonares. Sirve para el debate sobre atajos, **sin afirmar** que sea un atajo. No se ve concentración en los bordes: el marco de 3 px tiene el 38 % del área y recibe el 40 % del calor.
- Cita para el recuadro de atajos, verificada: Zech JR et al., «Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study», *PLOS Medicine* 15(11): e1002683, 2018. Las CNN identificaban el hospital y el departamento de origen, y rendían mejor en validación interna que externa en 3 de 5 comparaciones ([PLOS](https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1002683)).

## 5. Entrenamiento real (PneumoniaMNIST 28, cacheado)

Protocolo: imágenes /255, etiquetas float32 `(n,1)`, `Adam(1e-3)`, `binary_crossentropy`, `metrics=["accuracy"]`, **batch 128**, `fit(verbose=0)`, `keras.utils.set_random_seed(42)` y un proceso nuevo por medición. Splits: train 4.708 (25,8 % normal), val 524 (25,8 %) y test 624 (**37,5 %** normal). **Referencia tonta en test: 390/624 = 62,5 %.**

### 5.1 Tiempos (segundos, locales y ruidosos)

`t_fit` incluye la validación de cada época. «ép1» es la primera época, con el trazado; «ép·» es la mediana del resto.

| Config | Params | todos los hilos: t_fit (ép1 / ép·) | 2 hilos: t_fit (ép1 / ép·) |
|---|---|---|---|
| **Defecto** (2 bl, 16, k3, pool, d0.2, densa 64, 8 ép) | 56.129 | **3,1** (0,94 / 0,30) | **5,2** (1,11 / 0,53) |
| Defecto con batch 64 | 56.129 | 3,8 | 5,5 |
| Defecto con curvas en vivo (Agg + PNG por época) | 56.129 | 4,0 (dibujo 0,53 s en total) | 5,6 |
| Res 64 | 406.337 | 12,9 | 20,8 |
| Sin pooling, 28 | 1.184.577 | 7,5 | 13,4 |
| Sin pooling, 64 | 7.377.729 | 44,1 | 74,9 |
| 3 bloques | 27.521 | 2,6 | — |
| 1 bloque | 173.345 | 2,0 | — |
| Dropout 0, 20 ép | 56.129 | 5,5 | — |
| Dropout 0,5, 20 ép | 56.129 | 6,6 | — |
| BN (momentum 0,99) | 56.321 | 4,0 | — |
| class_weight | 56.129 | 2,6 | 4,3 |
| Sin pooling + densa 128 + dropout 0, 20 ép (reto C propuesto) | 2.364.353 | 18,9 | — |

Se suman unos 5 s de import de TF/Keras por proceso. `predict` sobre test (624) tarda unos 0,1 s. 🟡 Colab gratuito (2 vCPU) será más lento que «2 hilos en M3 Pro». Con un factor de 2 a 3, el defecto rondaría 10-15 s, muy por debajo de 90 s. **Hay que medirlo en Colab.**

**Curvas en vivo en un kernel real** (nbclient, `display(fig, display_id=True)` + `handle.update(fig)`): 2,51 s sin curvas frente a 3,60 s con ellas (8 épocas, unos 130 ms por época). En el notebook ejecutado queda **una sola** salida `display_data` (se actualiza en el sitio). Sin `plt.close(fig)` tras el primer `display`, Jupyter la pinta otra vez al final de la celda. 🟡 Que `display_id` se actualice en Colab lo verifica el frente e.

### 5.2 Métricas (semilla 42, umbral 0,5; sens = recall de Neumonía, espec = recall de Normal)

| Config | val_acc | test_acc | sens | espec | AUC |
|---|---|---|---|---|---|
| Defecto | 0,950 | 0,865 | 0,949 | 0,727 | 0,934 |
| Defecto con batch 64 | 0,960 | 0,862 | 0,967 | 0,688 | 0,933 |
| Res 64 | 0,960 | 0,837 | 0,977 | 0,603 | 0,929 |
| Sin pooling, 28 | 0,954 | 0,865 | 0,980 | 0,675 | 0,925 |
| Sin pooling, 64 | 0,954 | 0,835 | 0,980 | 0,594 | 0,922 |
| 1 bloque | 0,962 | 0,840 | 0,977 | 0,611 | 0,925 |
| 3 bloques | 0,922 | 0,865 | 0,923 | 0,769 | 0,935 |
| Dropout 0, 20 ép | 0,962 | 0,841 | 0,977 | 0,615 | 0,940 |
| Dropout 0,5, 20 ép | 0,970 | 0,856 | 0,977 | 0,654 | 0,942 |
| BN (m=0,99) | **0,742** | **0,625** | 1,000 | **0,000** | 0,926 |
| BN (m=0,9) | 0,971 | 0,862 | 0,992 | 0,645 | 0,962 |
| class_weight | 0,916 | 0,883 | 0,936 | 0,795 | 0,932 |

- **Val (~95 %) frente a test (~85 %) no es sobreajuste.** Según `medmnist.INFO` (paquete instalado), train y val salen del mismo conjunto de entrenamiento original (reparto 9:1) y test sale del conjunto de validación original, con otra proporción de clases. Casi todo el error en test son falsos positivos: la especificidad es baja. La referencia oficial de MedMNIST (ResNet-18 a 28 px) es ACC 0,854 y AUC 0,944 ([medmnist.com](https://medmnist.com/)). Nuestra red pequeña está en ese orden, dicho con prudencia.
- **Semillas 42, 1 y 2** (defecto): test_acc 0,865 / 0,806 / 0,848; sens 0,949 / 0,992 / 0,972; espec 0,727 / 0,496 / 0,641. **El ruido entre semillas es comparable al efecto de muchos cambios de arquitectura.** El guion debe presentarlo como tendencias.

### 5.3 BatchNorm: hallazgo importante

Con `BatchNormalization()` por defecto (`momentum=0.99`), batch 128 y 8 épocas (296 pasos), la val_acc vale **0,7424 en todas las épocas** con las semillas 42, 1 y 2. La val_loss sube de 0,58 a 2,0 y en test responde siempre «Neumonía». Las estadísticas móviles no llegan a converger. La AUC sigue en 0,90-0,93: el ranking es bueno y la calibración en inferencia es la que se rompe. No se arregla con lr 3e-4. Con 15 épocas o batch 64 mejora a medias (val 0,82 / 0,90). **`momentum=0.9` lo arregla**: val_acc 0,971 / 0,952 / 0,968 con las 3 semillas, AUC 0,954-0,962 y 5,2 s. Sin este cambio, el alumno concluiría que «BatchNorm empeora la red», que es falso en general.

### 5.4 Reto C (sobreajuste): no se ve con la red por defecto

| Config (20 épocas) | min val_loss (época) | val_loss final | train_loss final | t_fit |
|---|---|---|---|---|
| Defecto con dropout 0 (s42 / s1) | 0,094 (20) / 0,105 (20) | igual | — / 0,115 | 5,5 |
| Defecto con dropout 0,5 | 0,100 (20) | 0,100 | — | 6,6 |
| Sin pooling, dropout 0 | 0,092 (14) | 0,103 (+12 %) | 0,060 | 14,7 |
| **Sin pooling, densa 128, dropout 0** | **0,093 (16)** | **0,121 (+30 %)** | 0,045 | 18,9 |
| **Sin pooling, densa 128, dropout 0,5** | 0,098 (20) | 0,098 | 0,070 | 20,7 |
| 1 bloque, sin pooling, densa 128, dropout 0 | 0,106 (20) | igual | 0,065 | 8,4 |

**Propuesta:** el reto C con `usar_pooling=False`, `neuronas_capa_densa=128` y 20 épocas, comparando dropout 0 y 0,5. Es la única variante medida con un sobreajuste visible: la curva de validación toca fondo y sube. Cuesta unos 19-21 s en local (2,4 M de parámetros); en Colab 🟡 probablemente más de 60 s. Aunque sea así, el efecto es moderado y depende de la semilla. En el guion, conviene decir «debería empezar a separarse» y no «verás sobreajuste». Si se mantiene el reto C tal cual (red por defecto), el mensaje honesto es «en este dataset y con esta red pequeña, 20 épocas aún no sobreajustan».

### 5.5 Diagnóstico automático (reglas probadas contra las 25 ejecuciones con historial)

```python
def diagnostico(h, base_val=0.742):          # base_val = proporción de Neumonía en validación
    vl, ta, va = h["val_loss"], h["accuracy"][-1], h["val_accuracy"][-1]
    imin, n = vl.index(min(vl)), len(vl)
    if va <= base_val + 0.02:   return "colapso: responde casi siempre lo mismo"
    if (vl[-1] > 1.15 * min(vl) and imin <= n - 3) or ta - va > 0.05: return "sobreajuste"
    if ta < 0.90:               return "infraajuste"
    return "razonable" + (" (aún mejoraba: prueba más épocas)" if imin >= n - 2 else "")
```

Resultado: detecta el colapso de BN con m=0,99 (3 de 3) y el sobreajuste del reto C propuesto. El resto sale «razonable». Nota para el texto visible: al principio, la validación puede ir *mejor* que el entrenamiento, porque el dropout solo actúa al entrenar y la cifra de entrenamiento es la media de la época. Se ve en `curvas_vivo.png`.

## 6. Semillas y reproducibilidad

- `keras.utils.set_random_seed(42)` antes de construir da un resultado **idéntico bit a bit** entre dos procesos (val_loss 0,15234342217445374 en ambos). También es idéntico **en el mismo kernel** al reentrenar 3 veces (nbclient: 0,152343 las tres). La celda es idempotente siempre que se fije la semilla dentro de la función de entrenar.
- Con otro número de hilos cambia en la 4.ª decimal (0,15224 con 2 hilos). **En Colab los números no coincidirán exactamente con los de local.** Los resultados de referencia del guion deben decir «aprox.».
- La estimación previa (sección 8) consume RNG. Hay que reconstruir el modelo y volver a fijar la semilla después (verificado: da resultados idénticos a entrenar sin estimar).

## 7. class_weight

```python
n0, n1 = int((ytr == 0).sum()), int((ytr == 1).sum())
class_weight = {0: len(ytr) / (2 * n0), 1: len(ytr) / (2 * n1)}   # ≈ {0: 1.94, 1: 0.67}
m.fit(xtr, ytr, ..., class_weight=class_weight)
```

Funciona con `ytr` `(n,1)` en float32 o uint8, tal como viene del npz, sin avisos de Python. Efecto a umbral 0,5 (sin ponderar → ponderado):

| semilla | sens | espec | test_acc |
|---|---|---|---|
| 42 | 0,949 → 0,936 | 0,727 → 0,795 | 0,865 → 0,883 |
| 1 | 0,992 → 0,951 | 0,496 → 0,688 | 0,806 → 0,853 |
| 2 | 0,972 → 0,946 | 0,641 → 0,765 | 0,848 → 0,878 |

La dirección es consistente: **sube la especificidad y baja algo la sensibilidad**. Da para el debate del reto D («compensar el desbalanceo no es gratis en salud»).

## 8. Estimación previa del tiempo y umbrales

```python
def estimar_segundos(m, xtr, ytr, xva, epocas, batch=128, n=5):
    t0 = time.perf_counter(); m.train_on_batch(xtr[:batch], ytr[:batch]); t_calent = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(1, n + 1):
        m.train_on_batch(xtr[i*batch:(i+1)*batch], ytr[i*batch:(i+1)*batch])
    t_lote = (time.perf_counter() - t0) / n
    pasos = -(-len(xtr) // batch)
    return 1.2 * (t_calent + epocas * (pasos + len(xva) / batch / 3) * t_lote)   # ×1,2: infravalora un 10-20 %
# Después: keras.backend.clear_session(); set_random_seed(semilla); m = construir(...); m.compile(...)
```

Estimado sin el factor frente a real, en 14 configuraciones: la razón está entre 0,72 y 1,04, con mediana de unos 0,87. Con ×1,2 queda en ±15 %. Ejemplos: defecto 2,7 frente a 3,1; sin pooling 64 con 2 hilos 70 frente a 75; res 64 10,3 frente a 12,9. Coste de la estimación: 0,5 s en el defecto y unos 3 s en un modelo de 13 M.

**Estimaciones de configuraciones extremas (sin entrenar, 8 épocas, local):**

| Config | Params | Estimado | RSS del proceso |
|---|---|---|---|
| 64 px, 4 bl, 32 filtros, densa 128, sin pool | 103 M | **1.858 s** (2.944 s con 2 hilos); 1 lote tarda 6-9,5 s | 3,1 GB |
| 64 px, 4 bl, 16, densa 64, sin pool | 25,8 M | 243 s | 2,3 GB |
| 64 px, 2 bl, 32, densa 128, sin pool | 29,5 M | 141 s | 1,5 GB |
| 28 px, 4 bl, 32, densa 128, sin pool | 13,5 M | 118 s | 1,2 GB |
| 28 px, 4 bl, 16, densa 64, sin pool | 3,4 M | 46 s | 0,8 GB |

**Umbrales propuestos:**

1. Tamaño espacial < 1 → **bloquear**, con el mensaje de la sección 1 y sin construir nada.
2. **Más de 20 M de parámetros → bloquear sin estimar.** Estimar ya cuesta 30-50 s en el caso de 103 M y el entrenamiento se va a 30-50 min en local. Mensaje: «Esta red tiene X millones de parámetros (Y MB): en la CPU de Colab tardaría del orden de una hora. Activa el pooling o baja la resolución a 28».
3. Si no, estimar en la máquina real (Colab): ≤ 90 s, seguir sin avisar; 90-600 s, **avisar** con «Tiempo estimado: unos N minutos» y entrenar; > 600 s, **bloquear**, porque no cabe en el bloque de 25 min del laboratorio.
4. Aviso suave si el último `relu_N` mide ≤ 3×3: «el mapa de Grad-CAM será muy grueso».
5. Durante el entrenamiento, poner en el título de la figura «época k/N · t s · quedan ≈ (N−k)·t_media s».

## 9. Gotchas

- `keras.Model(m.inputs, …)` → `UserWarning` en cada llamada. **Usa `m.input`.**
- visualkeras 0.2.0 con Keras 3: sin el parche de `output_shape`, falla. La leyenda sale en inglés y la fuente por defecto no tiene tildes.
- `BatchNormalization()` por defecto colapsa con esta receta. Usa `momentum=0.9`.
- Grad-CAM sobre `conv_N` con BN es casi siempre vacío. Usa `relu_N`.
- Grad-CAM sobre «Normal» sale vacío con frecuencia (un 53 % en el defecto). Trátalo como un resultado y no dejes que la figura quede en negro sin explicación.
- Feature maps: entre el 20 % y el 25 % de los canales están apagados. Ordena por activación.
- `tf.config.threading.set_*_parallelism_threads` solo funciona antes de inicializar TF. No hace falta en el notebook; aquí solo se usó como aproximación a Colab.
- En macOS no sale ruido por stderr con `TF_CPP_MIN_LOG_LEVEL=2`. 🟡 En Colab habrá mensajes de CUDA/XLA. Conviene fijar la variable antes de importar.
- `summary(print_fn=…)` recibe una única cadena con toda la tabla, no una línea por capa.

## Fuentes

- Ejecución local: `verificacion/b_keras_gradcam/v1…v7*.py`, `resultados.jsonl`, `resultados2.jsonl`, `resultados3.jsonl` y figuras `vk_*.png`, `mpl_*.png`, `gradcam_*.png`, `fmaps_*.png`, `curvas_vivo.png`.
- Código fuente instalado: `visualkeras/layered.py:290` (lee `layer.output_shape`) y `keras/src/models/functional.py:211` (`output_shape` solo existe en `Model`).
- `medmnist.INFO["pneumoniamnist"]` (paquete 3.0.2): descripción de los splits y licencia CC BY 4.0.
- [PyPI visualkeras](https://pypi.org/project/visualkeras/): 0.2.0 es la última (14-oct-2025, según la API JSON de PyPI).
- [keras-team/keras#19052](https://github.com/keras-team/keras/issues/19052): las capas de Keras 3 no tienen `input_shape`/`output_shape`.
- [medmnist.com](https://medmnist.com/): benchmarks de PneumoniaMNIST (ResNet-18 (28): AUC 0,944, ACC 0,854).
- [Zech et al. 2018, PLOS Medicine](https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1002683).

## Pendientes de validar

- 🟡 Factor de velocidad real de la CPU de Colab frente a esta máquina. Hay que medir el defecto y el reto C propuesto en Colab.
- 🟡 Que `display(fig, display_id=True)` + `handle.update(fig)` se actualice en el sitio en Colab (frente e).
- 🟡 Que el `pip install visualkeras` de Colab instale la misma 0.2.0 (y que siga necesitando el parche).
