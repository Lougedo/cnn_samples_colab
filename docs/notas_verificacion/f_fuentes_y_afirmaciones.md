# Frente f) Fuentes y afirmaciones que verán los alumnos

Verificado el 24-sep-2026. Método: texto completo de la fuente primaria (XML/PDF/código fuente) descargado y leído, metadatos de Crossref, API de GitHub y ejecución local en el `.venv` (Python 3.12.7). Material de trabajo en `verificacion/f_fuentes/` (ignorado por git).

Leyenda: **✅** comprobado en fuente primaria o ejecutando código · **🟡** fuente secundaria, interpretación o no comprobado.

---

## 1. Zech et al. 2018: aprendizaje por atajos en radiografías

**Cita ✅**
Zech JR, Badgeley MA, Liu M, Costa AB, Titano JJ, Oermann EK. *Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study.* PLOS Medicine 15(11): e1002683, 6-nov-2018. DOI [10.1371/journal.pmed.1002683](https://doi.org/10.1371/journal.pmed.1002683). Licencia del artículo: CC BY 4.0.

**Qué hicieron (✅, texto completo leído)**
- 158.323 radiografías de tórax de adultos de tres sistemas hospitalarios de EE. UU.: NIH (112.120 de 30.805 pacientes), Mount Sinai (42.396 de 12.904) e Indiana University (3.807 de 3.683). Edades medias: 46,9, 63,2 y 49,6 años.
- CNN DenseNet-121 preentrenada en ImageNet, imágenes a 224×224.
- Umbral fijado para lograr **95 % de sensibilidad**, «to simulate model use for a theoretical screening task». Sirve para el punto 7.

**Qué encontraron (✅)**
- El rendimiento interno superó al externo en **3 de 5** comparaciones naturales. Ejemplos: el modelo entrenado en NIH tuvo AUC interna 0,750 y externa 0,695 en Mount Sinai; el de Mount Sinai, 0,802 interna y 0,717 en NIH. El modelo conjunto dio 0,931 interna y 0,815 externa en Indiana.
- Prevalencia de neumonía: 34,2 % en Mount Sinai frente a 1,2 % en NIH y 1,0 % en Indiana. Ordenar los casos **solo por hospital de origen**, sin mirar la imagen, ya daba AUC 0,861 en el conjunto MSH–NIH.
- Una CNN identificó el sistema hospitalario en el 99,95 % de las radiografías de NIH (22.050/22.062), el 99,98 % de Mount Sinai (8.386/8.388) y el 95,59 % de Indiana (737/771). Dentro de Mount Sinai distinguió planta de hospitalización y urgencias en el 100 % de las portátiles, y la prevalencia de neumonía difería entre ambas (41,1 % frente a 32,8 %).
- Los mapas de calor señalaban sobre todo las esquinas. En imágenes concretas, la red había aprendido a detectar **una ficha metálica que los técnicos colocan en la esquina del campo**. Los autores lo cuentan como ejemplo ilustrativo y reconocen que no pueden medir del todo qué otros factores intervienen.

**Texto propuesto para el recuadro de debate (Notebook 1, §1.7)**
> **Aprendizaje por atajos.** En 2018, Zech y colaboradores entrenaron CNN para detectar neumonía en unas 158.000 radiografías de adultos de tres hospitales de EE. UU. Los modelos rendían peor fuera del hospital en el que habían aprendido, y una CNN podía adivinar de qué hospital venía cada placa con más de un 99 % de acierto, en algunos casos fijándose en una marca metálica de la esquina. Como la proporción de neumonías cambiaba mucho entre hospitales, saber de dónde venía la placa ya ayudaba a acertar sin mirar el pulmón. Un buen resultado en el test no garantiza que el modelo mire donde creemos.

Pregunta para el grupo: «Nuestras radiografías son de niños y salen de un único centro. ¿Qué atajos podría estar aprendiendo nuestro modelo que no veríamos en el test?».

**Otros ejemplos bien documentados (opcionales)**
- ✅ Geirhos R, Jacobsen J-H, Michaelis C, Zemel R, Brendel W, Bethge M, Wichmann FA. *Shortcut learning in deep neural networks.* Nature Machine Intelligence 2, 665–673 (2020). DOI [10.1038/s42256-020-00257-z](https://doi.org/10.1038/s42256-020-00257-z) · arXiv [2004.07780](https://arxiv.org/abs/2004.07780). El artículo (versión arXiv v5 leída) pone el ejemplo de la vaca: una red que clasifica vacas y falla cuando aparecen fuera de un prado, porque ha usado la hierba como atajo. También cita el caso de Zech: la red de neumonía se fija en la ficha del hospital y no en el pulmón. Vale como frase de refuerzo: «El problema está documentado y tiene nombre».
- ✅ DeGrave AJ, Janizek JD, Lee S-I. *AI for radiographic COVID-19 detection selects shortcuts over signal.* Nature Machine Intelligence 3, 610–619 (2021). DOI [10.1038/s42256-021-00338-7](https://doi.org/10.1038/s42256-021-00338-7). Según el resumen del preprint en medRxiv ([10.1101/2020.09.13.20193565](https://doi.org/10.1101/2020.09.13.20193565)), los detectores de COVID-19 se apoyaban en factores de confusión y no en la patología, parecían precisos y fallaban en hospitales nuevos. 🟡 El detalle de qué atajos concretos encontraron (marcas, texto, posición) no lo he leído en el artículo final porque nature.com exige sesión. Si se usa, no añadir detalles.
- ⚠️ **No usar la historia de los tanques** (red que distinguía tanques por el tiempo que hacía). Geirhos la cita como «the neural net tank urban legend»: es una leyenda urbana.

---

## 2. CNN Explainer

**Cita ✅**
Wang ZJ, Turko R, Shaikh O, Park H, Das N, Hohman F, Kahng M, Chau DH. *CNN Explainer: Learning Convolutional Neural Networks with Interactive Visualization.* IEEE Transactions on Visualization and Computer Graphics 27(2): 1396–1406 (2021; presentado en IEEE VIS 2020). DOI [10.1109/TVCG.2020.3030418](https://doi.org/10.1109/TVCG.2020.3030418) · arXiv [2004.15004](https://arxiv.org/abs/2004.15004).
Web: <https://poloclub.github.io/cnn-explainer/> (HTTP 200 el 24-sep-2026). Código: <https://github.com/poloclub/cnn-explainer>. **Licencia MIT** (© 2020 Polo Club of Data Science). Autoría: Georgia Tech y Oregon State.

**Modelo que muestra ✅** (verificado en el `model.json` que sirve la web en producción y en `tiny-vgg/tiny-vgg.py`)
- **Tiny VGG**, inspirada en la CNN de demostración del curso CS231n de Stanford.
- Entrada **64×64×3 (RGB)**.
- Cuatro convoluciones de **10 filtros 3×3**, paso 1 y **sin relleno** (`padding='valid'`), cada una seguida de ReLU.
- Dos max pooling 2×2 con paso 2, un flatten y una densa de 10 neuronas con softmax.
- Tamaños espaciales: 64 → 62 → 60 → 30 → 28 → 26 → 13. El flatten da 13×13×10 = 1.690 valores.
- **10 clases** de Tiny ImageNet: lifeboat, ladybug, pizza, bell pepper, bus, koala, espresso, red panda, orange, sport car. La imagen por defecto es *espresso*.
- Corre en el navegador con TensorFlow.js, sin instalar nada.

**Controles comprobados** en el código fuente y en el `bundle.js` desplegado. Todos los textos de interfaz citados abajo aparecen en el bundle.
- Arriba a la izquierda, las miniaturas de las 10 imágenes y un «+» con el título *Add new input image*, que abre un modal para pegar una URL o subir una imagen.
- Arriba a la derecha, el botón **«Show detail»** (icono de ojo). Muestra dimensiones de capa y escalas de color. A su lado está el selector de escala de color: *Unit / Module / Global*.
- **Pasar el ratón** sobre un mapa de activación de una capa convolucional resalta los núcleos que lo producen. En la primera convolución son 3, uno por canal.
- **Clic en una neurona convolucional** abre la vista **«Convolution»** (la *Interactive Formula View*): el núcleo 3×3 se desliza con animación. Tiene botón de pausa y reproducción, y al pasar el ratón por las matrices se mueve el núcleo. Otro clic en la misma neurona o en un espacio vacío la cierra.
- **Clic en una neurona ReLU** abre la vista **«ReLU Activation»**.
- **Clic en una neurona de max pooling** abre la vista **«Max Pooling»** (2×2).
- **Clic en una clase de la capa de salida** muestra el flatten y la vista **«Softmax Score for "…"»**, con logits y probabilidades.
- En el artículo inferior, sección *Understanding Hyperparameters*, hay un widget con **Input Size, Padding, Kernel Size y Stride**. Sirve para enseñar qué hace el relleno.

🟡 El recorrido está verificado en el código desplegado (rama `gh-pages`, último commit del 6-oct-2023), **no clicando en un navegador**. Ensayarlo una vez antes de clase.

**Recorrido para el profesor (9 min)**

| Min | Qué hacer | Qué decir (idea) |
|---|---|---|
| 0:00–1:00 | Abrir la web compartiendo pantalla. Recorrer las columnas de izquierda a derecha y pulsar **Show detail** | «Cada columna es una capa. Entra una foto de 64×64 en color y salen 10 probabilidades» |
| 1:00–4:00 | Pasar el ratón por el primer mapa de `conv_1_1` (aparecen 3 enlaces) y hacer **clic** para abrir *Convolution*. Pausar, mover el ratón por la matriz de entrada | «Es un filtro de 3×3 que recorre toda la imagen con los mismos 9 pesos. Esta capa tiene 10 filtros y por eso da 10 mapas» |
| 4:00–5:00 | Cerrar y hacer clic en una neurona de `relu_1_1` | «Los negativos pasan a 0 y el resto se queda igual. Sin esto, apilar capas no serviría de nada» |
| 5:00–6:30 | Clic en una neurona de `max_pool_1` | «De cada 2×2 se queda el máximo: la imagen pasa de 60 a 30 de lado y se descarta el 75 % de los valores» |
| 6:30–8:00 | Clic en la clase *espresso* de la salida | «El flatten estira todo en una fila y el softmax lo convierte en probabilidades que suman 1» |
| 8:00–9:00 | Cambiar a otra miniatura (koala, red panda) y ver cómo cambian los mapas | «La red es pequeña y se equivoca. En el cuaderno vais a construir una parecida» |
| (+1 min si sobra) | Widget de hiperparámetros: subir *Padding* a 1 | «Con relleno la imagen no encoge. Aquí no lo usan y por eso pierde 2 píxeles en cada convolución» |

**Tres preguntas para un grupo no técnico**
1. «El mismo filtro de 3×3 recorre toda la foto. ¿Qué ganamos frente a aprender un peso distinto para cada píxel?» Respuesta esperada: muchos menos parámetros, y el patrón se detecta esté donde esté.
2. «Después del pooling queda una cuarta parte de la imagen. ¿Qué ganamos y qué podemos perder?» Se gana cálculo y tolerancia a pequeños desplazamientos. Se puede perder detalle fino y la posición exacta. Enlaza con el reto B.
3. «La red dice "espresso, 90 %". Si le enseñamos una foto de un perro, que no está entre sus 10 clases, ¿qué contestará?» El softmax siempre reparte el 100 % entre las clases que conoce. Enlaza con los atajos y con Grad-CAM.

**Avisos**
- La web está **solo en inglés**. El profesor traduce de palabra: *Show detail* es «mostrar detalle» y *activation map*, «mapa de activación».
- **Diferencia con nuestro cuaderno.** CNN Explainer usa convoluciones sin relleno y la imagen encoge. Si el Notebook 1 usa `padding='same'` (lo decide el frente b), hay que decirlo para que las dimensiones de la ficha no confundan.
- 🟡 **Seguridad.** El HTML en producción carga `https://polyfill.io/v3/polyfill.min.js`, un dominio que sufrió un ataque de cadena de suministro en junio de 2024 ([Sansec](https://sansec.io/research/polyfill-supply-chain-attack)). Hoy el dominio resuelve a IP de Cloudflare y no sé qué sirve. Recomendación: que la abra solo el profesor, con navegador actualizado y compartiendo pantalla, y no pedir a los alumnos que la abran. No lo he comprobado más allá de la resolución DNS.

---

## 3. Grad-CAM

**Cita ✅**
Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* ICCV 2017, pp. 618–626. DOI [10.1109/ICCV.2017.74](https://doi.org/10.1109/ICCV.2017.74). Versión ampliada en IJCV 128(2): 336–359 (2020), DOI [10.1007/s11263-019-01228-7](https://doi.org/10.1007/s11263-019-01228-7). arXiv [1610.02391](https://arxiv.org/abs/1610.02391).

**Descripción prudente (una frase)**
> Grad-CAM usa los gradientes de la clase predicha que llegan a la última capa convolucional para dibujar un mapa de calor **aproximado** de las zonas de la imagen que más han empujado esa predicción.

El resumen del artículo lo llama «coarse localization map» ✅.

**Limitaciones que conviene decir**
- **Resolución gruesa ✅.** El mapa tiene el tamaño de la última capa convolucional y luego se amplía. Tamaños medidos en Keras 3.15.1 con `padding='same'` y pooling 2×2 tras cada bloque:

  | Bloques | Lado de la última conv (res. 28) | (res. 64) |
  |---|---|---|
  | 1 | 28 | 64 |
  | 2 | 14 | 32 |
  | 3 | 7 | 16 |
  | 4 | **3** | 8 |

  Con 4 bloques a resolución 28, el mapa es de 3×3 y apenas localiza nada. Conviene decirlo en el cuaderno.
- **Correlación, no causa.** Señala dónde era sensible el modelo, no por qué, y no demuestra que esa zona sea la causa clínica. Un mapa bonito puede estar mirando un atajo, como en el caso de Zech.
- **Evidencia en radiografía de tórax ✅.** Saporta A, et al. *Benchmarking saliency methods for chest X-ray interpretation.* Nature Machine Intelligence 4, 867–878 (2022). DOI [10.1038/s42256-022-00536-x](https://doi.org/10.1038/s42256-022-00536-x). De los siete métodos evaluados, Grad-CAM fue el que mejor localizó, pero todos rindieron significativamente peor que la referencia humana, sobre todo con lesiones pequeñas o de forma compleja.

**Texto propuesto (2-3 líneas junto a la figura)**
> El color indica en qué zonas se ha fijado la red para esta predicción. Es un mapa aproximado, del tamaño de la última capa convolucional ampliado, y enseña dónde miró, no por qué acertó o falló. En radiografías reales estos mapas localizan peor que un radiólogo.

---

## 4. MedMNIST v2 y PneumoniaMNIST

**Citas ✅**
- Yang J, Shi R, Wei D, Liu Z, Zhao L, Ke B, Pfister H, Ni B. *MedMNIST v2 – A large-scale lightweight benchmark for 2D and 3D biomedical image classification.* Scientific Data 10: 41 (2023). DOI [10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8). Artículo con licencia CC BY 4.0.
- Datos de las versiones 28/64/128/224 («MedMNIST+»): Zenodo, registro v3.0 del 16-ene-2024, **licencia cc-by-4.0**, DOI [10.5281/zenodo.10519652](https://doi.org/10.5281/zenodo.10519652). Es lo que descarga `medmnist` 3.0.2.
- Fuente original: Kermany DS, et al. *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell 172(5): 1122–1131.e9 (2018). DOI [10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010). Datos: Kermany, Zhang, Goldbaum, Mendeley Data v2 y v3, DOI [10.17632/rscbjbr9sj.3](https://doi.org/10.17632/rscbjbr9sj.3), **CC BY 4.0** ✅.

**Datos verificados ✅**
- `medmnist.INFO['pneumoniamnist']`: `license = "CC BY 4.0"`, etiquetas `0 = normal` y `1 = pneumonia`, reparto 4.708 / 524 / 624.
- El artículo MedMNIST v2 dice: «5,856 pediatric chest X-Ray images». El entrenamiento original se dividió 9:1 en entrenamiento y validación, y la validación original se usa como test.
- Recuento ejecutado sobre los `.npz` locales (idéntico en 28 y 64):

  | Split | Normal | Neumonía | % neumonía |
  |---|---|---|---|
  | train | 1.214 | 3.494 | 74,2 % |
  | val | 135 | 389 | 74,2 % |
  | test | 234 | 390 | **62,5 %** |

  ⚠️ **El test tiene otra proporción que el entrenamiento.** La referencia tonta («siempre neumonía») da **62,5 %** de accuracy en test, con sensibilidad del 100 % y especificidad del 0 %. No calcularla con el % de entrenamiento.
- Solo DermaMNIST es CC BY-NC 4.0 (`INFO`). Todo lo demás es CC BY 4.0. Importa solo si alguien cambia de subconjunto.

**Origen del conjunto 🟡** (el artículo de Cell devuelve 403 y no lo he podido leer)
Radiografías anteroposteriores de pacientes pediátricos de **1 a 5 años** del **Guangzhou Women and Children's Medical Center** (Cantón, China), de cohortes retrospectivas, con cribado de calidad y etiquetas de dos médicos expertos. Lo repiten fuentes secundarias, p. ej. [PMC10245274](https://pmc.ncbi.nlm.nih.gov/articles/PMC10245274/). En Mendeley solo figura que colaboraron la UCSD y ese centro. Ponerlo como «según los autores originales» y sin cifras de pacientes.

**Aviso de uso no clínico ✅**
MedMNIST v2, *Usage Notes*: «Please note that this dataset is NOT intended for clinical use, as substantially reducing the resolution of medical images might result in images that are insufficient to represent and capture different disease pathologies».

Texto propuesto para la portada:
> ⚠️ **No apto para uso clínico.** PneumoniaMNIST es un conjunto educativo: radiografías de tórax de niños reducidas a 28×28 o 64×64 píxeles. Sus autores advierten que a esa resolución se pierde información necesaria para diagnosticar. Nada de lo que hagamos aquí sirve para diagnosticar a nadie.

Atribución (CC BY 4.0 obliga a citar, enlazar la licencia e indicar cambios):
> Datos: PneumoniaMNIST (MedMNIST v2; Yang et al., *Scientific Data*, 2023), derivado de Kermany et al. (*Cell*, 2018). Licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [Aquí, una línea con los cambios reales que aplique el cuaderno, p. ej. normalización o reescalado.]

---

## 5. Ultralytics YOLO26 y licencia AGPL-3.0

**Fechas y cita ✅**
- Publicado el **14-ene-2026** con `ultralytics` 8.4.0 (release de GitHub «v8.4.0 – YOLO26 Models Release», API de GitHub, `published_at 2026-01-14`).
- Los pesos `yolo26n.pt` que se descargaron llevan dentro `date 2025-12-15`, `version 8.3.222` y `license 'AGPL-3.0 (https://ultralytics.com/license)'` (ejecutado).
- Artículo: Jocher et al. *Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models.* arXiv [2606.03748](https://arxiv.org/abs/2606.03748), v1 del 2-jun-2026.

**Novedades ✅** (release, docs y resumen de arXiv)
- Inferencia de extremo a extremo **sin NMS**, con doble cabeza: una uno-a-uno para inferir y otra uno-a-muchos para entrenar.
- **Sin DFL** (Distribution Focal Loss).
- Optimizador MuSGD (SGD con Muon), ProgLoss y STAL (mejor asignación para objetos pequeños).
- Hasta un 43 % más rápido en CPU (ONNX) que YOLO11n, según Ultralytics.
- Cinco tamaños (n/s/m/l/x). COCO mAP 40,9 (n) a 57,5 (x).

Comprobado ejecutando: la cabeza `Detect` tiene `reg_max = 1` (sin DFL) y las ramas `one2one_cv2/cv3`, y el yaml trae `end2end: True`. 🟡 Si la predicción usa por defecto la rama sin NMS lo confirma el frente c.

**Arquitectura ✅** (`cfg/models/26/yolo26.yaml` y `yolo26n.pt` inspeccionados)
- 2.572.280 parámetros, 6,1 GFLOPs según el yaml.
- Backbone de `Conv` + `C3k2` + `SPPF`, más **un bloque `C2PSA`** (atención posicional) al final del backbone y **un `C3k2` con atención** en la cabeza P5.
- En total, 2 módulos `Attention` que suman ~102.656 parámetros (**~4 %**). Las propias proyecciones de la atención son convoluciones 1×1: el 99,2 % de los parámetros son pesos de `Conv2d`.
- Preentrenado en COCO: 80 clases, `names[0] == 'person'` ✅.

**Descripción para alumnos**
> YOLO26 nano es un detector preentrenado con 80 tipos de objeto; «persona» es la clase 0. Por dentro es mayoritariamente convolucional, como vuestra red del Notebook 1 pero con unos 2,6 millones de parámetros y un par de bloques de atención. Da las cajas directamente, sin el filtrado final de cajas repetidas (NMS) que usaban versiones anteriores.

**AGPL-3.0 en llano ✅** (texto de la licencia en [gnu.org](https://www.gnu.org/licenses/agpl-3.0.txt) y [ultralytics.com/license](https://www.ultralytics.com/license)). Divulgativo, no es asesoramiento jurídico.
- **Usarlo en clase está bien.** La AGPL permite ejecutar el programa sin condiciones (sección 2: «You may make, run and propagate covered works that you do not convey, without conditions»), y Ultralytics incluye expresamente el trabajo académico y el aprendizaje personal.
- **Si modificas el código y lo ofreces como servicio en red** (web o API), debes ofrecer a sus usuarios el código fuente completo de tu versión (sección 13).
- **Si distribuyes software que lo incluye**, tiene que ir bajo AGPL-3.0 y con el código. Ultralytics interpreta que eso alcanza a la aplicación completa y a los modelos entrenados con su código. 🟡 Es la interpretación del licenciante, no un criterio pacífico.
- **Uso comercial cerrado** (herramientas internas, SaaS, dispositivos): Ultralytics vende una **licencia Enterprise**.

Texto propuesto (Notebook 2, preparación):
> Este cuaderno usa Ultralytics YOLO, con licencia AGPL-3.0. Puedes usarlo para aprender y experimentar. Si algún día lo metes en un producto, lo ofreces como servicio o distribuyes una versión modificada, la AGPL te obliga a publicar tu código. Para uso comercial cerrado existe una licencia Enterprise de pago.

---

## 6. Privacidad (España/UE, divulgativo)

**Fuentes ✅**
- **RGPD**, art. 4.1 y 4.2 ([EUR-Lex, ES](https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32016R0679)).
  - Dato personal: «toda información sobre una persona física identificada o identificable».
  - Tratamiento: «cualquier operación… como la recogida, registro… consulta, utilización…».
- **Considerando 26 RGPD.** Los principios de protección de datos no se aplican a la información anónima, «inclusive con fines estadísticos». Para decidir si alguien es identificable cuentan «todos los medios… que razonablemente pueda utilizar» el responsable o un tercero.
- **EDPB, Guidelines 3/2019 on processing of personal data through video devices**, v2.0 adoptada el 29-ene-2020 (v2.1 del 26-feb-2020 corrige una errata). [Página](https://www.edpb.europa.eu/our-work-tools/our-documents/guidelines/guidelines-32019-processing-personal-data-through-video_en) · [PDF](https://www.edpb.europa.eu/system/files/documents/files/file1/edpb_guidelines_201903_video_devices_en_0.pdf).
  - La introducción distingue técnicas «more intrusive (e.g. complex biometric technologies) or less intrusive (e.g. simple counting algorithms)».
  - Apartado 2.1: el Reglamento no se aplica si la persona no puede identificarse ni directa ni indirectamente.
  - Sobre expectativas razonables: la gente puede esperar no ser vigilada en zonas de ocio y descanso, «such as sitting areas, tables in restaurants». Muy relevante para el ejemplo de la cafetería.
- **AEPD, «Guía sobre el uso de videocámaras para seguridad y otras finalidades»** (edición de febrero de 2025, [PDF](https://www.aepd.es/guias/guia-videovigilancia.pdf)). Dice: «La imagen de una persona en la medida que identifique o pueda identificar a la misma constituye un dato de carácter personal». También excluye el RGPD solo cuando las imágenes no afectan a personas identificadas o identificables (ejemplo de una panorámica de playa en la que no se distingue a nadie). Cita el «estudio de hábitos de uso o consumo» como finalidad que implica personas identificables.

**Texto propuesto (portada del Notebook 2)**
> **Contar no es identificar, pero grabar sí es tratar datos.** La imagen de una persona reconocible es un dato personal (RGPD, art. 4), y grabarla o analizarla es un tratamiento aunque al final solo queramos un número. El Comité Europeo de Protección de Datos considera el conteo simple menos intrusivo que la biometría, pero sigue haciendo falta una base legal, informar y guardar lo mínimo, y en sitios como las mesas de un bar la expectativa de no ser grabado pesa más. Por eso de este cuaderno solo salen agregados: personas por zona y por intervalo. Esto es divulgación, no asesoramiento jurídico.

Matiz para el profesor 🟡 (principio general de anonimización, no verificado en una fuente concreta para este caso): un agregado solo deja de ser dato personal si de verdad no permite señalar a nadie. Con muy pocas personas, franjas muy cortas o datos que se puedan cruzar (por ejemplo, «a las 9:02 entró 1 persona» junto con el cuadrante de turnos), el agregado puede volver a identificar. El análisis en el momento de la captura sigue siendo tratamiento aunque se descarte la imagen.

---

## 7. Sensibilidad frente a especificidad en cribado

**Texto propuesto (Notebook 1, §1.5)**
> En un cribado, dejar pasar una neumonía (falso negativo) suele salir más caro que una falsa alarma (falso positivo): el paciente se va a casa sin tratamiento, mientras que la falsa alarma se aclara con otra prueba. Por eso en cribado se suele priorizar la **sensibilidad**, aceptando más falsas alarmas (menos **especificidad**). El equilibrio correcto depende del contexto clínico (prevalencia, coste y riesgo de la prueba siguiente, recursos) y lo decide el equipo clínico, no el modelo.

Apoyos:
- ✅ Zech et al. fijaron el umbral para un **95 % de sensibilidad** «para simular un cribado». Es un ejemplo real de esa decisión.
- ✅ Trevethan R. *Sensitivity, Specificity, and Predictive Values: Foundations, Pliabilities, and Pitfalls in Research and Practice.* Frontiers in Public Health 5: 307 (2017). DOI [10.3389/fpubh.2017.00307](https://doi.org/10.3389/fpubh.2017.00307). Referencia general de definiciones (solo metadatos verificados).
- Definiciones para el cuaderno, con la clase positiva = 1 = neumonía:
  - sensibilidad = VP / (VP + FN), «de los enfermos, cuántos detecta»;
  - especificidad = VN / (VN + FP), «de los sanos, cuántos deja tranquilos».

---

## 8. supervision y vídeos de demo

- **supervision 0.30.5: licencia MIT ✅** (`License-Expression: MIT` y fichero `licenses/LICENSE` en el paquete instalado).
- **Vídeos de demo.** `supervision.assets.VideoAssets` tiene 10 vídeos alojados en `https://media.roboflow.com/supervision/video-examples/`: vehicles, milk-bottling-plant, vehicles-2, grocery-store, subway, market-square, people-walking, beach-1, basketball-1 y skiing.
  - 🟡 **Ni el paquete ni la documentación ([Assets](https://supervision.roboflow.com/latest/assets/)) indican origen ni licencia de los clips.**
  - Solo `skiing.mp4` tiene origen documentado: PR [#1657](https://github.com/roboflow/supervision/pull/1657), «Video by Adrien JACTA», Pexels ✅. El PR de `beach-1.mp4` (#1107) no dice nada.
- **Licencia de Pexels ✅** ([pexels.com/license](https://www.pexels.com/license/)), para el clip de cafetería que pueda subir el profesor. Es una licencia propia, no CC0.
  - Uso gratuito, sin atribución obligatoria, y se puede modificar.
  - No se puede mostrar a personas identificables de forma negativa u ofensiva, vender copias sin modificar ni sugerir respaldo.

Texto propuesto:
> Vídeos de demostración: distribuidos por Roboflow con la librería `supervision` (MIT). La librería no indica la licencia de cada clip, así que se usan solo para la demostración en clase y no se redistribuyen. Si subes tu propio clip (por ejemplo, de Pexels), respeta su licencia y no uses vídeos con personas reconocibles sin base legal.

---

## Recomendaciones para quien construya los notebooks

1. Referencia tonta en test = **62,5 %**. Calcularla siempre sobre `test_labels`, nunca sobre el entrenamiento.
2. Clase positiva = 1 = neumonía. Sensibilidad = recall de la clase 1.
3. Explicar que las radiografías son **pediátricas y de un solo centro** (🟡 origen): conecta directamente con el debate de Zech (adultos, tres hospitales).
4. Grad-CAM: avisar de la resolución cuando la última convolución sea de 3×3 o menos (4 bloques a res. 28) y enseñar en pantalla el tamaño del mapa.
5. Si el Notebook 1 usa `padding='same'`, una línea que lo compare con CNN Explainer, que no usa relleno.
6. YOLO26: decir «mayoritariamente convolucional con un par de bloques de atención». No decir «100 % convolucional».
7. Poner los avisos de licencia visibles: CC BY 4.0 con atribución, AGPL-3.0, supervision MIT y clips sin licencia declarada.
8. 🟡 **Decisión para `DECISIONS.md`:** como el Notebook 2 importa Ultralytics (AGPL) y el repo se publicará con badge de Colab, lo más sencillo y prudente es publicar el código del repo abierto y declarar AGPL-3.0 al menos para el Notebook 2 y `src/build_nb2.py`.
9. No usar la leyenda de los tanques como ejemplo real.
10. CNN Explainer lo abre solo el profesor (aviso de polyfill.io, 🟡).

## Qué se instaló

Nada en el `.venv`. Para leer los PDF de EDPB, AEPD y Geirhos instalé `pypdf` 6.19.0 con `uv pip install --target <scratchpad>/pylib`, fuera del proyecto y del venv, porque no había `pdftotext` ni `pdftoppm` y la herramienta de lectura de PDF los necesita.

## Pendientes de validar

- 🟡 Edad (1–5 años) y hospital de origen de las radiografías de Kermany: falta leerlo en el artículo de Cell (403 desde aquí).
- 🟡 Detalle de los atajos concretos de DeGrave et al. (artículo final tras sesión de Nature).
- 🟡 Licencia de los vídeos de `supervision` salvo `skiing.mp4`.
- 🟡 Recorrido de CNN Explainer clicado en navegador real: ensayarlo antes de clase.
- 🟡 Estado actual de `polyfill.io`.

## Fuentes utilizadas

- Zech et al. 2018, PLOS Med: <https://doi.org/10.1371/journal.pmed.1002683> (XML completo desde journals.plos.org)
- Geirhos et al. 2020: <https://doi.org/10.1038/s42256-020-00257-z> · <https://arxiv.org/abs/2004.07780>
- DeGrave et al. 2021: <https://doi.org/10.1038/s42256-021-00338-7> · preprint <https://doi.org/10.1101/2020.09.13.20193565>
- CNN Explainer: <https://poloclub.github.io/cnn-explainer/> · <https://github.com/poloclub/cnn-explainer> · <https://doi.org/10.1109/TVCG.2020.3030418> · `assets/data/model.json` y `bundle.js` en producción
- Grad-CAM: <https://doi.org/10.1109/ICCV.2017.74> · <https://doi.org/10.1007/s11263-019-01228-7> · <https://arxiv.org/abs/1610.02391>
- Saporta et al. 2022: <https://doi.org/10.1038/s42256-022-00536-x>
- MedMNIST v2: <https://doi.org/10.1038/s41597-022-01721-8> (texto completo en Europe PMC, PMC9852451) · Zenodo <https://doi.org/10.5281/zenodo.10519652>
- Kermany et al. 2018: <https://doi.org/10.1016/j.cell.2018.02.010> · datos <https://data.mendeley.com/datasets/rscbjbr9sj/3>
- YOLO26: <https://docs.ultralytics.com/models/yolo26/> · <https://github.com/ultralytics/ultralytics/releases/tag/v8.4.0> · <https://arxiv.org/abs/2606.03748>
- Licencias: <https://www.gnu.org/licenses/agpl-3.0.txt> · <https://www.ultralytics.com/license> · <https://www.pexels.com/license/>
- RGPD: <https://eur-lex.europa.eu/legal-content/ES/TXT/HTML/?uri=CELEX:32016R0679>
- EDPB 3/2019: <https://www.edpb.europa.eu/system/files/documents/files/file1/edpb_guidelines_201903_video_devices_en_0.pdf>
- AEPD: <https://www.aepd.es/guias/guia-videovigilancia.pdf>
- Trevethan 2017: <https://doi.org/10.3389/fpubh.2017.00307>
- supervision: <https://supervision.roboflow.com/latest/assets/> · <https://github.com/roboflow/supervision/pull/1657>
- polyfill.io: <https://sansec.io/research/polyfill-supply-chain-attack>
