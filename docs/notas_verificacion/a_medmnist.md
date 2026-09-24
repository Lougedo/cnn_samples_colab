# Frente a) medmnist / PneumoniaMNIST — notas de verificación (fase 1)

Fecha: 24-sep-2026. Todo lo que no lleva 🟡 se ha **ejecutado** contra el venv del repo o se ha leído en fuente primaria.
Scripts y figuras en `verificacion/a_medmnist/` (ignorado por git): `t1_api.py`, `t2_imgs.py`, `t3_download.py`,
`t4_tf_torch.py`, `cargar.py` (cargador propuesto con autocomprobación), `muestras.png`, `esquinas.png`.
Tiempos medidos en un M3 Pro con otros agentes cargando la CPU y la red a la vez: tómalos como orientativos.

## 1. Veredicto rápido

- `medmnist 3.0.2` (la última en PyPI, de 16-sep-2024) funciona. La firma es `PneumoniaMNIST(split, transform=None, target_transform=None, download=False, as_rgb=False, root=DEFAULT_ROOT, size=None, mmap_mode=None)`.
- Referencia tonta en test (responder siempre «Neumonía»): **390/624 = 62,50 %**.
- **Trampa gorda:** la descarga de medmnist usa `torchvision.datasets.utils.download_url`, que llama a `urlopen` **sin timeout**. Hoy, a las 00:44, una descarga del fichero de 64 px desde Zenodo se quedó colgada más de 10 minutos con la conexión TCP abierta y sin recibir datos. En clase eso deja la celda girando para siempre. Solución: descargar con nuestro propio código, con timeout y MD5, y usar medmnist solo para leer (`download=False`). Está en §7.
- Espejo verificado solo para **28 px**: Hugging Face `albertvillanova/medmnist-v2`, idéntico byte a byte (MD5 `28209eda…`). **No es oficial.** No he encontrado espejo para 64 px.

## 2. Versiones (venv local, Python 3.12.7)

medmnist 3.0.2 · torch 2.14.0 · torchvision 0.29.0 · scikit-image 0.26.0 · fire 0.7.1 · tqdm 4.70.1 · Pillow 12.3.0 · pandas 3.0.6 · scikit-learn 1.9.1 · numpy 2.5.3 · tensorflow 2.21.0 · keras 3.15.1.

## 3. API verificada

| Caso | Resultado real |
|---|---|
| `available_sizes` (clases 2D) | `[28, 64, 128, 224]`. `size=None` o `28` usa `pneumoniamnist.npz`; el resto, `pneumoniamnist_{size}.npz` |
| `size=32`, `size=100` | `AssertionError` **sin mensaje** (`assert size in self.available_sizes`) |
| `size="64"` (str) | `AssertionError` sin mensaje. **Haz `int()` del valor del formulario.** `np.int64(64)` sí vale |
| `split="validation"` | `ValueError` sin mensaje. Los únicos valores válidos son `"train"`, `"val"` y `"test"` |
| `root` inexistente | `RuntimeError: Failed to setup the default root directory…`. **medmnist no crea `root`**: hay que hacer `os.makedirs(root, exist_ok=True)` antes |
| `root` existe pero sin fichero y `download=False` | `RuntimeError: Dataset not found. You can set download=True…` |
| `.imgs` | `numpy.ndarray` **uint8**, forma `(N, 28, 28)` o `(N, 64, 64)` (sin canal) |
| `.labels` | `numpy.ndarray` **uint8**, forma **`(N, 1)`**, valores {0, 1}. Hace falta `.ravel()` |
| `INFO["pneumoniamnist"]["label"]` | `{"0": "normal", "1": "pneumonia"}` → 0 = Normal, 1 = Neumonía |
| `ds[i]` | `(PIL.Image modo "L", array([y]) int64)`. **No lo uses con Keras**: trabaja directamente con `ds.imgs` y `ds.labels` |
| Coste de crear el objeto | 1–13 ms en 28 px y 6–55 ms en 64 px (hace `np.load` del npz entero) |
| `import medmnist` | Crea `~/.medmnist` (vacío) como efecto secundario, **siempre** |

El `.npz` también se lee sin medmnist: `np.load(ruta)` → claves `train_images, val_images, test_images, train_labels, val_labels, test_labels`.

## 4. Splits, clases y referencias (medido, idéntico en 28 y 64)

| Split | N | Normal (0) | Neumonía (1) |
|---|---|---|---|
| train | 4.708 | 1.214 (25,8 %) | 3.494 (74,2 %) |
| val | 524 | 135 (25,8 %) | 389 (74,2 %) |
| test | 624 | 234 (37,5 %) | 390 (62,5 %) |

- **Referencia tonta en test**, «siempre Neumonía»: accuracy **0,6250**, sensibilidad 1,0, especificidad 0,0 y AUC 0,5. «Siempre Normal» da 0,375.
- ⚠️ **La prevalencia cambia de validación a test: 74,2 % → 62,5 %.** La referencia tonta en validación sería 74,2 %, no 62,5 %. Si el notebook compara con la referencia, que sea en test, y que el texto no prometa que la accuracy de validación anticipa la de test. El motivo: el test de MedMNIST es el test original de Kermany, con otra proporción de clases (ver §8). Es esperable que el modelo prediga Neumonía de más, con sensibilidad alta y especificidad más baja en test 🟡 (tendencia habitual; se confirma en fase 3).
- `class_weight` «balanced» (sklearn `compute_class_weight`) sobre train: **{0: 1.939, 1: 0.6737}**. Devuelve `np.float64`: conviértelo con `float()` antes de mostrarlo o pasarlo a Keras.
- Las etiquetas de 28 y 64 son **idénticas y van en el mismo orden**, y la imagen de 64 reducida a 28 tiene una correlación media de 0,993 con la de 28. Un «índice concreto» apunta a la misma radiografía en las dos resoluciones.
- Las etiquetas de train vienen barajadas (1.804 cambios de clase en 4.708 imágenes).
- Duplicados exactos (en 28 px): 27 grupos, 29 copias sobrantes (18 dentro de train, 8 entre train y val, 3 dentro de test). **Ninguno tiene etiquetas contradictorias** y no hay ninguno entre train y test. No merece la pena contarlo en clase; basta saberlo si alguien pregunta.
- Referencia «seria» (MedMNIST v2, Tabla 3, test de PneumoniaMNIST): ResNet-18 (28) AUC 0,944 / ACC 0,854; ResNet-18 (224) 0,956 / 0,864; ResNet-50 (28) 0,948 / 0,854; ResNet-50 (224) 0,962 / 0,884; auto-sklearn 0,942 / 0,855; AutoKeras 0,947 / 0,878; Google AutoML Vision 0,991 / 0,946. Sirve para situar lo que saque una CNN pequeña en 90 s.

## 5. Descarga: URLs, tamaños, MD5 y comportamiento

Registro Zenodo **10519652** ([MedMNIST+], versión 3.0, 16-ene-2024, CC BY 4.0, DOI 10.5281/zenodo.10519652). Datos de la API de Zenodo, que coinciden con `INFO`:

| Fichero | Bytes | MD5 |
|---|---|---|
| `pneumoniamnist.npz` (28) | 4.170.669 | `28209eda62fecd6e6a2d98b1501bb15f` |
| `pneumoniamnist_64.npz` | 20.606.998 | `8f4eceb4ccffa70c672198ea285246c6` |
| `pneumoniamnist_128.npz` | 75.506.212 | `05b46931834c231683c68f40c47b2971` |
| `pneumoniamnist_224.npz` | 214.384.716 | `d6a3c71de1b945ea11211b03746c1fe1` |

URL: `https://zenodo.org/records/10519652/files/pneumoniamnist{,_64}.npz?download=1` (responde 200 directo, sin redirección).

- **Tiempos medidos desde Zenodo** (unos 2 MB/s): 28 px entre 2,4 y 5,3 s; 64 px entre 8,8 y 11,4 s. Hubo **un cuelgue de más de 10 minutos** (ver §1).
- **`download=True` con el fichero ya presente:** comprueba el MD5 (`check_integrity`) y **no vuelve a descargar**. Tarda 0,03 s en calcular el MD5 de 20 MB, pero la primera llamada suma ~1,3 s por el `import torchvision.datasets.utils` perezoso. Si el fichero está corrupto, lo descarga de nuevo. No escribe nada por stdout ni stderr.
- **Descarga interrumpida:** torchvision escribe directamente sobre el nombre final. Si se corta, queda un npz corrupto; la siguiente llamada con `download=True` lo detecta por MD5 y descarga otra vez, pero con `download=False` fallaría al leerlo.
- El `except:` desnudo de medmnist convierte **cualquier** fallo en un `RuntimeError` en inglés con instrucciones para descargar a mano.
- **User-Agent:** Zenodo devuelve **403** si la cabecera es `"Mozilla/5.0"` a secas. Con `pytorch/vision`, `Python-urllib/3.12`, `curl/…` o una cadena descriptiva propia, responde 200/206. **No finjas ser un navegador.**
- **Silenciar la barra tqdm** (va a stderr, ~6 KB de texto por descarga):
  - `contextlib.redirect_stderr(io.StringIO())` alrededor de la llamada → 0 bytes en stderr. **Es la opción robusta.**
  - `os.environ["TQDM_DISABLE"]="1"` **solo funciona si se fija antes de importar tqdm.** Fijado después, no tiene efecto (`disable=False`, comprobado). En un notebook re-ejecutable no es fiable.
  - Con el cargador de §7 la pregunta desaparece: no usa tqdm.

## 6. Dependencias e importación

- `importlib.metadata.requires('medmnist')` = `numpy, pandas, scikit-learn, scikit-image, tqdm, Pillow, fire, torch, torchvision`, **sin versiones fijadas**. El wheel pesa 25,8 KB (`py3-none-any`, `requires_python>=3.6`). `fire 0.7.1` pesa 116 KB y depende de `termcolor`.
- **Colab:** según `googlecolab/backend-info/pip-freeze.txt` (commit del 23-sep-2026), el runtime CPU trae **Python 3.13.15**, torch 2.11.0+cpu, torchvision 0.26.0+cpu, scikit-image 0.25.2, pandas 2.2.3, scikit-learn 1.6.1, tqdm 4.67.3, pillow 11.3.0, termcolor 3.3.0, **tensorflow 2.21.0 y keras 3.13.2**. Solo faltan `medmnist` y `fire`, así que `pip install -q medmnist` baja unos 140 KB. 🟡 No lo he ejecutado en Colab: está deducido del listado. Ojo, Colab ya va en 3.13, no en 3.11–3.12 como suponía el BRIEF (dato para el frente e).
- **Fuera de Colab, sin torch:** `pip install medmnist` arrastra torch y torchvision, que en Linux vienen de PyPI con CUDA: varios GB 🟡. Y si torch no está, `import medmnist` **no falla**: imprime `Please install the required packages first…` y luego `medmnist.PneumoniaMNIST` da `AttributeError` (comprobado simulando la ausencia de torch). `medmnist.INFO` sí sigue disponible.
- **Tiempo de `import medmnist`** en local y en caliente: 2,6–3,2 s. Se reparte entre torch (~1,3 s) y `sklearn.metrics`, que lo importa `medmnist.evaluator` (~1,3–1,6 s). En Colab, probablemente más 🟡.
- **torch y TensorFlow en el mismo proceso:** funcionan en los dos órdenes de importación, entrenan una época sin errores y no sacan avisos (`t4_tf_torch.py`). En frío, `import medmnist` tarda 4,6 s y `keras`+`tf` 4,8 s.

## 7. Cargador recomendado (probado: `verificacion/a_medmnist/cargar.py`)

Estrategia: nuestro código asegura el `.npz` (Zenodo con timeout, después el espejo de Hugging Face solo para 28 px, y MD5 siempre); medmnist se limita a leerlo con `download=False`. Así se cumple «medmnist para los datos», torchvision no descarga nunca y no puede colgarse, y no hay barra tqdm.

```python
import hashlib, os, shutil, urllib.request
MD5 = {28: "28209eda62fecd6e6a2d98b1501bb15f", 64: "8f4eceb4ccffa70c672198ea285246c6"}
URLS = {28: ["https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1",
             "https://huggingface.co/datasets/albertvillanova/medmnist-v2/resolve/"
             "f6dd981c7400b3e3738bde12ff948b7cc8f0d623/data/pneumoniamnist.npz"],   # espejo NO oficial
        64: ["https://zenodo.org/records/10519652/files/pneumoniamnist_64.npz?download=1"]}

def _md5(ruta):
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def asegurar_npz(res, raiz, timeout=60):
    os.makedirs(raiz, exist_ok=True)
    ruta = os.path.join(raiz, "pneumoniamnist.npz" if res == 28 else f"pneumoniamnist_{res}.npz")
    if os.path.exists(ruta) and _md5(ruta) == MD5[res]: return ruta
    fallos = []
    for url in URLS[res]:
        parcial = ruta + ".parcial"
        try:
            pet = urllib.request.Request(url, headers={"User-Agent": "laboratorio-cnn-unir/1.0"})  # no "Mozilla/5.0"
            with urllib.request.urlopen(pet, timeout=timeout) as r, open(parcial, "wb") as f:
                shutil.copyfileobj(r, f)
            if _md5(parcial) != MD5[res]: raise ValueError("MD5 distinto: fichero incompleto")
            os.replace(parcial, ruta); return ruta
        except Exception as e:
            fallos.append(f"{url.split('/')[2]}: {e}")
            if os.path.exists(parcial): os.remove(parcial)
    consejo = "Prueba con resolución 28, que tiene servidor de reserva." if res != 28 else "Espera un minuto y vuelve a ejecutar la celda."
    raise RuntimeError(f"No he podido descargar PneumoniaMNIST a {res} px. {consejo}\n"
        f"Si sigue fallando, descarga «{os.path.basename(ruta)}» desde https://doi.org/10.5281/zenodo.10519652 "
        f"y súbelo a {raiz}.\nDetalle técnico: " + " | ".join(fallos))

def cargar_pneumoniamnist(res=28, raiz=os.path.expanduser("~/.medmnist")):
    import medmnist
    res = int(res)
    if res not in (28, 64): raise ValueError(f"Resolución {res} no disponible: elige 28 o 64.")
    asegurar_npz(res, raiz)
    return {s: (lambda d: (d.imgs, d.labels.ravel()))(medmnist.PneumoniaMNIST(split=s, size=res, root=raiz))
            for s in ("train", "val", "test")}
```

Autocomprobación ejecutada (`python cargar.py` → `TODO OK`):

1. Con caché válida: 28 px en 4,1 s (incluye `import medmnist`) y 64 px en 0,13 s. Sin red.
2. Con Zenodo «mudo» (IP no enrutable y timeout=5), cae al espejo de Hugging Face y termina bien en 7,5 s.
3. Con un npz corrupto, lo descarga de nuevo y cuadra el MD5.
4. Con 64 px y Zenodo caído, sale el error en castellano y no deja `.parcial`.
5. Con resolución 32, `ValueError` en castellano.

Contra Zenodo real, 28 px tarda 2,5 s y 64 px 9,2 s; la segunda llamada, que solo comprueba el MD5, 0,03 s.

- El timeout de `urlopen` se aplica **a cada operación de socket**. Un servidor que se calla corta la espera a los 60 s; uno que envía muy despacio pero sin pararse, no.
- `raiz` por defecto es `~/.medmnist`, que medmnist crea igualmente. Para la verificación local, pasa `raiz=".../datos/medmnist"`, donde ya están los dos npz, o lee una variable de entorno. Así no hay red.

## 8. Imágenes: orientación, rango y preprocesado

- Rango global 0–255 (en test de 28 px, máximo 254) y media ≈ 145. **No vienen normalizadas por imagen**: el mínimo por imagen va de 0 a 185 (mediana 41) y el máximo de 130 a 255 (mediana 206). La exposición y el contraste varían mucho.
- **Preprocesado:** `x = imgs.astype("float32")[..., None] / 255.0` → `(N, H, W, 1)` en [0, 1]; `y = labels.ravel().astype("float32")` para `binary_crossentropy`. No hace falta nada más. En memoria, float32 ocupa unos 15 MB (28 px) o 77 MB (64 px) en train.
- **Orientación:** todas las imágenes vistas están derechas y son frontales (48 al azar en `muestras.png` y 16 en `esquinas.png`). Las imágenes medias por clase muestran una anatomía coherente, con la silueta cardiaca a la derecha de la imagen, y ninguna aparece volteada. Hay variación de encuadre, rotación leve, brazos o hombros visibles y recorte central (el recorte usa el lado corto y se pierden laterales). 🟡 No he revisado las 5.856 una a una. **No uses volteo horizontal como aumento de datos**: cambia la anatomía de lado.
- A la vista: las normales tienen más contraste y costillas nítidas; las de neumonía, opacidad difusa. A 28 px apenas se ven costillas; a 64 px sí. Buen gancho para el control `resolucion`.
- He buscado marcas o letras en las esquinas por si servían de ejemplo de «atajo». No lo sostiene: las esquinas brillantes resultaron ser hombros y brazos. No afirmes que el dataset tiene marcadores de texto.

## 9. Licencia y origen (fuentes primarias salvo 🟡)

- **Licencia de PneumoniaMNIST: CC BY 4.0** (README oficial, `INFO["license"]`, Zenodo). El código de medmnist es Apache-2.0. DermaMNIST es la única con CC BY-NC.
- **Aviso clínico (README y artículo):** «this dataset is NOT intended for clinical use». El artículo lo justifica por la reducción de resolución, que puede dejar imágenes insuficientes para representar la patología.
- **Canal oficial:** el README y la ficha de Zenodo dicen «The only official distribution link for the MedMNIST dataset is Zenodo». Por eso el espejo de Hugging Face hay que declararlo **no oficial** en `DECISIONS.md`. Se justifica porque el MD5 coincide con el publicado por los autores, la revisión está fijada (`f6dd981c…`, subida el 30-may-2023, sin acceso restringido, `license: cc-by-4.0`) y CC BY 4.0 permite redistribuir. El espejo es anterior a MedMNIST+ y aun así su MD5 coincide: el fichero de 28 px no cambió entre v2 y v3.
- **Cita de MedMNIST:** Yang J, Shi R, Wei D, Liu Z, Zhao L, Ke B, Pfister H, Ni B. *MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification.* Scientific Data 10, 41 (2023). doi:10.1038/s41597-022-01721-8 (el artículo es CC BY 4.0). Los autores piden citar también la v1 (ISBI 2021, doi:10.1109/ISBI48211.2021.9434062) y el dataset de origen.
- **Cómo se hicieron los splits** (artículo e `INFO`): «We split the source training set with a ratio of 9:1 into training and validation set, and use its source validation set as the test set». Las imágenes originales, en gris y de (384–2.916)×(127–2.713) px, se recortan al centro con ventana del lado corto y se redimensionan. En total son 5.856 radiografías pediátricas.
  - Consecuencias: (a) la validación sale del mismo conjunto que train, con la misma prevalencia; (b) el test es el conjunto de evaluación original, con otra prevalencia; (c) 🟡 el reparto 9:1 parece hecho por imagen y no por paciente, y el dataset de Kermany tiene varias imágenes por paciente, así que la validación puede ser algo optimista. Los 8 duplicados exactos entre train y val apuntan en esa dirección. No lo he comprobado contra el CSV de correspondencias de MedMNIST (está en Google Drive).
- **Origen:** Kermany DS et al., *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, **Cell** 172(5):1122–1131.e9 (2018), doi:10.1016/j.cell.2018.02.010, PMID 29474911. Datos: Kermany, Zhang y Goldbaum, Mendeley Data, doi:10.17632/rscbjbr9sj (v2 y v3), **CC BY 4.0**.
  - 🟡 **Edad y hospital:** «pediatric patients of one to five years old from Guangzhou Women and Children's Medical Center, Guangzhou», proyección anteroposterior y parte de la atención clínica rutinaria. Tras un control de calidad, dos expertos graduaron los diagnósticos y un tercero revisó el conjunto de evaluación. Es la frase de los STAR Methods, pero la he leído en copias secundarias (Kaggle, GitHub, Labelbox) porque cell.com y ScienceDirect devuelven 403 a la descarga automática. Mi confianza es alta, pero conviene que Lou la mire en el PDF.
  - Coherencia verificada: train+val de MedMNIST = 1.349 normales + 3.883 neumonías = 5.232, y test = 234 + 390 = 624. Cuadra con los recuentos que se suelen dar para el conjunto de Kermany (5.232 de entrenamiento y 624 de test) 🟡, que están en fuentes secundarias.
- Para el debate de Grad-CAM: todo PneumoniaMNIST procede de **un solo centro**. El atajo de «reconocer el hospital» que describe Zech et al. (2018) no puede darse *dentro* de este dataset. Sí vale como aviso de por qué un modelo entrenado aquí puede no generalizar a otro hospital. Redáctalo así y no insinúes que este dataset mezcla hospitales.

## 10. Recomendaciones para quien construya el Notebook 1

1. Preparación: `pip install -q medmnist`, con `-q` y redirigiendo la salida. En Colab solo añade medmnist y fire. Imprime `medmnist.__version__`.
2. Usa el cargador de §7 tal cual. **No uses `download=True`**, que no tiene timeout. Cachea en `~/.medmnist`: si Colab reinicia la sesión se pierde, pero se vuelve a bajar en 3–10 s.
3. Pasa `resolucion` por `int()`. Con 64 px sin Zenodo, el mensaje ya sugiere volver a 28.
4. Etiquetas: `labels.ravel()`; 0 = «Normal» y 1 = «Neumonía». La sensibilidad es el recall de la clase 1.
5. Para la referencia tonta, calcula `y_test.mean()` en vivo (0,625) en vez de escribir el número a mano. Así sigue siendo cierta si cambia el dataset.
6. Dile al alumno en una línea que en test hay menos neumonías (62,5 %) que en entrenamiento (74,2 %).
7. `compensar_desbalanceo`: `compute_class_weight("balanced", …)` → `{0: 1.94, 1: 0.67}`, convertido a `float`.
8. Nada de volteos horizontales. Normaliza con `/255`.
9. Avisos visibles: CC BY 4.0, «no apto para uso clínico», cita a MedMNIST v2 y a Kermany 2018. En `DECISIONS.md`, registra el espejo de Hugging Face como no oficial y verificado por MD5, solo para 28 px.
10. En `README` → problemas frecuentes: «Zenodo va lento o no responde → la celda espera como mucho 60 s por intento; con 28 px usa el espejo; con 64 px, vuelve a 28 o sube el fichero a mano».

## 11. Fuentes

- Código instalado: `.venv/lib/python3.12/site-packages/medmnist/{dataset.py,info.py,__init__.py}` y `torchvision/datasets/utils.py` (0.29.0).
- README de MedMNIST: https://github.com/MedMNIST/MedMNIST (raw: https://raw.githubusercontent.com/MedMNIST/MedMNIST/main/README.md)
- Registro Zenodo: https://zenodo.org/records/10519652 · API: https://zenodo.org/api/records/10519652
- MedMNIST v2 (Sci Data 2023), texto completo: https://europepmc.org/article/PMC/PMC9852451 · https://doi.org/10.1038/s41597-022-01721-8
- Kermany 2018 (Cell): https://doi.org/10.1016/j.cell.2018.02.010 (403 a la descarga automática) · metadatos en Europe PMC, PMID 29474911
- Datos de Kermany (Mendeley, CC BY 4.0): https://data.mendeley.com/datasets/rscbjbr9sj/2 · https://data.mendeley.com/datasets/rscbjbr9sj/3
- 🟡 Cita textual de los STAR Methods en copias secundarias: https://github.com/angeanto/Chest-X-Ray-Pneumonia-Detection-with-keras-CNN · https://labelbox.com/datasets/chest-x-ray-images/
- Espejo: https://huggingface.co/datasets/albertvillanova/medmnist-v2 (API: https://huggingface.co/api/datasets/albertvillanova/medmnist-v2)
- Paquetes de Colab: https://github.com/googlecolab/backend-info/blob/main/pip-freeze.txt · `os-info.txt`
- PyPI: https://pypi.org/pypi/medmnist/json · https://pypi.org/pypi/fire/json
