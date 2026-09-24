# Guion docente · Una CNN por dentro y contar personas en vídeo

IASP · última sesión. Dos versiones del mismo guion:

- **Versión S7 · 40 min** (jueves 24-sep-2026, online): bloques 8 y 9 del esqueleto, de 20:30 a 21:10. Es la que se da esta noche.
- **Versión completa · 60 min**: la agenda del brief (§2). Para una edición con la hora entera.

Leyenda: sin marca = medido o comprobado · 🟡 inferido o sin comprobar en Colab · [🔴 Pendiente de confirmar] = no hay fuente.
Todas las cifras salen de `verificacion/final/hechos_para_docs.md` y de las vistas de alumno de las ejecuciones finales.

**Índice**
0. Las dos versiones de un vistazo · 1. Antes de clase · 2. Tarjetas de consulta (CNN Explainer, retos) ·
3. Versión S7 · 40 min · 4. Versión completa · 60 min · 5. Plan B si algo falla · 6. Resultados de referencia medidos ·
Fuentes utilizadas · Pendientes de validar

---

## 0. Las dos versiones de un vistazo

### Versión S7 · 40 min (20:30–21:10)

| Bloque | Hora | Min | Qué |
|---|---|---|---|
| 8a | 20:30–20:32 | 2 | Mensaje del chat; arrancan 1.1 y 1.2 |
| 8b | 20:32–20:37 | 5 | CNN Explainer: convolución, ReLU, pooling |
| 8c | 20:37–20:39 | 2 | Checkpoint de datos y cómo se hace el reto |
| 8d | 20:39–20:50 | 11 | Retos individuales |
| 8e | 20:50–20:57 | 7 | Grad-CAM, atajos y debate |
| 8f | 20:57–20:58 | 1 | La hoja común, leída por mí |
| | **Bloque 8** | **28** | 2 + 5 + 2 + 11 + 7 + 1 = 28 |
| 9a | 20:58–21:00 | 2 | Del cuaderno 1 al 2; detector y umbral |
| 9b | 21:00–21:05 | 5 | Aforo por zonas y entradas por la línea |
| 9c | 21:05–21:07 | 2 | Qué sale del cuaderno: CSV y JSON |
| 9d | 21:07–21:10 | 3 | POC ≠ producción, otros sectores, privacidad |
| | **Bloque 9** | **12** | 2 + 5 + 2 + 3 = 12 |
| | **Total** | **40** | 28 + 12 = 40 ✅ |

Cabe en los 28 + 12 minutos que reserva `sesiones/S7/01-esqueleto.md`, sin mover nada más.

### Versión completa · 60 min

| Bloque | Reloj | Min | Qué |
|---|---|---|---|
| 1a | 00:00–02:00 | 2 | Mensaje del chat; arrancan 1.1 y 1.2 |
| 1b | 02:00–05:00 | 3 | CNN Explainer: la red y la convolución |
| 1c | 05:00–08:30 | 3,5 | CNN Explainer: ReLU, pooling y salida |
| 1d | 08:30–10:00 | 1,5 | Checkpoint de datos |
| | **Bloque 1** | **10** | 2 + 3 + 3,5 + 1,5 = 10 |
| 2a | 10:00–16:00 | 6 | Sígueme en pantalla: la red por defecto |
| 2b | 16:00–31:00 | 15 | Retos individuales |
| 2c | 31:00–35:00 | 4 | La hoja común, entre todos |
| | **Bloque 2** | **25** | 6 + 15 + 4 = 25 |
| 3a | 35:00–38:00 | 3 | Grad-CAM en su cuaderno |
| 3b | 38:00–40:00 | 2 | Atajos: Zech et al. (2018) |
| 3c | 40:00–45:00 | 5 | Debate |
| | **Bloque 3** | **10** | 3 + 2 + 5 = 10 |
| 4a | 45:00–46:30 | 1,5 | Del cuaderno 1 al 2 |
| 4b | 46:30–50:00 | 3,5 | El detector y el umbral |
| 4c | 50:00–54:00 | 4 | Aforo por zonas y entradas por la línea |
| 4d | 54:00–57:00 | 3 | Exportar y puente opcional a n8n |
| 4e | 57:00–60:00 | 3 | POC ≠ producción, otros sectores, privacidad |
| | **Bloque 4** | **15** | 1,5 + 3,5 + 4 + 3 + 3 = 15 |
| | **Total** | **60** | 10 + 25 + 10 + 15 = 60 ✅ |

Reloj relativo: 00:00 es el minuto en que empieza la hora.

### Qué se corta en la versión de 40 y qué hago yo solo

| En la de 60 | En la de 40 (S7) |
|---|---|
| CNN Explainer 6,5 min con la salida (softmax) y tres preguntas | 5 min: convolución, ReLU y pooling, con una pregunta. **Se corta** la salida y la pregunta del perro |
| «Sígueme»: todos entrenan juntos la red por defecto (6 min) | **Se corta.** La red por defecto ya está en la fila 2 de la hoja (la entreno antes de clase). La vuelvo a entrenar **yo solo** a las 20:40 para enseñar mi Grad-CAM |
| Retos 15 min | 11 min |
| Reto C con dos entrenamientos (dropout 0 y 0,5) | **Solo dropout 0.** El 0,5 queda de extra para quien acabe antes de las 20:45; su fila de referencia la pego yo en la hoja a las 20:45 |
| Lectura de la hoja entre todos, con pregunta (4 min) | 1 min, **la leo yo** |
| Grad-CAM + Zech + debate (10 min) | 7 min |
| Cuaderno 2 ejecutado en directo (15 min) | 12 min. **Lo ejecuto yo solo** durante los retos (20:41) y en el bloque 9 enseño los resultados |
| Pregunta del umbral en el cuaderno 2 | **Se corta.** En su lugar, 30 s: cuentan en el chat las personas del primer fotograma (9a) |
| Puente a n8n en directo (opcional) | **Se corta y no lo nombro.** En pantalla solo se ve lo que pone el cuaderno en 2.8 |

### Techo de teoría

Cuento como teoría los minutos en los que explico un concepto y el grupo solo escucha. No cuentan las instrucciones de uso,
la espera mientras se ejecuta una celda, la lectura de sus propios resultados ni el debate con micrófono abierto.

| Versión | Dónde | Min de teoría | Total |
|---|---|---|---|
| 60 min | 1b 3 · 1c 3,5 · 2a 1 · 2c 0,5 · 3a 1 · 3b 2 · 3c 0,5 · 4a 1 · 4b 0,5 · 4c 1 · 4e 0,5 | 3 + 3,5 + 1 + 0,5 + 1 + 2 + 0,5 + 1 + 0,5 + 1 + 0,5 | **14,5 ≤ 15 ✅** |
| 40 min | 8b 5 · 8e 2,5 · 9a 1 · 9b 1 · 9d 0,5 | 5 + 2,5 + 1 + 1 + 0,5 | **10 ≤ 15 ✅** |

⚠️ En la S7 el techo de la sesión entera ya está roto por la actualidad (26 min, aprobado por Lou, `05-checklist.md`).
Estos 40 minutos le suman 10 de explicación.

⚠️ El bloque 9 es una demo: el alumno mira y contesta en el chat (9a y 9d), pero no toca su cuaderno. Si el coordinador
cuenta toda la demo como escucha, la versión de 40 sube a 5 + 2,5 + 12 = 19,5 min, por encima de 15.

⚠️ Solape: la S4 v2 tenía una slide de 2 min sobre cómo «ve» una CNN, con atajos (`sesiones/S4/03-v2-lente-social.md`,
slide 27; 🟡 no consta si se llegó a dar). Si se dio, en 8e/3b basta con recordarla en una frase.

**Entregable del alumno:** su cuaderno 1 ejecutado, con la arquitectura de su reto y su Grad-CAM, guardado en su Drive.

---

## 1. Antes de clase (en orden)

1. **Enlace de los cuadernos.** Publicados en https://github.com/Lougedo/cnn_samples_colab (licencia AGPL-3.0, D50) y copiados en Drive. El enlace para el chat
   es el badge de la portada (Colab desde GitHub) o el de la copia de Drive compartida como lector.
2. **Probar el enlace en una ventana privada**, como alumno: que se ve la vista de formulario sin código, los rótulos que
   van en el mensaje del chat («Ejecutar de todos modos» y «Archivo → Guardar una copia en Drive») 🟡, cuánto tardan 1.1 y 1.2
   la primera vez [🔴 no medido] y que no pide reiniciar la sesión (🟡 sin comprobar con Python 3.13; la resolución de paquetes
   de `e_colab.md` dice que no hace falta).
   - **Criterio:** si 1.1 + 1.2 tardan más de 3 min en esta prueba, el mensaje del chat se pega a las 20:20 (punto 11), para que
     instalen mientras hablo del bloque 7.
3. **Tu cuaderno 1 en Colab**, en este orden:
   - 1.1 y 1.2. En 1.3, sin tocar nada más, `nombre_experimento` = `ref_defecto`. Luego 1.4, 1.5 y 1.8.
     Apunta el tiempo real de 1.4 (proyección: ~20–30 s 🟡) y copia la línea de 1.8.
   - **1.10 Plan B**: marca `ejecutar_plan_b` y ejecuta. Proyección en Colab: ~5–7 min 🟡. Entrena las 7 redes de referencia.
   - Haz captura de la tabla y del gráfico de 1.8 y descarga el CSV. La tabla de 1.8 vive en memoria y no se vuelve a leer
     del CSV: si la sesión de Colab se corta 🟡, te quedan la captura y el CSV.
   - Abre el CSV del Plan B (en Excel se abre en columnas, D19) y copia la fila de `ref_C_dropout_05_20_epocas` desde la
     columna `experimento` hasta el final, sin la columna `id`. Se pega en la hoja común a las 20:45, no antes, para no
     destripar el reto C (8d). 🟡 Que se reparta en columnas al pegarla.
   - Comprueba que los parámetros coinciden exactamente con la tabla del §6. Las métricas pueden moverse unos decimales.
4. **Crear la hoja común** (Google Sheets), p. ej. «IASP S7 · Retos CNN». [🔴 Pendiente: sin crear]
   - Fila 1: pega la **primera** línea de tu 1.8 (cabecera). Son 22 columnas: `experimento reto hipotesis resolucion bloques
     filtros duplicar kernel pooling batchnorm dropout densa compensar epocas parametros tiempo_entrenamiento_s acc_validacion
     accuracy sensibilidad especificidad auc diagnostico`.
   - Fila 2: pega la **segunda** línea de tu `ref_defecto`. Es la referencia con la que comparan B y D.
   - **Desde la fila 3, una fila por alumno**, en el orden del reparto (punto 5); A y C, dos filas seguidas. La fila siguiente
     a la última asignada queda para `ref_C_dropout_05_20_epocas`. Quien llegue tarde o haga experimentos extra, debajo.
     Así nadie pega a la vez en la misma «primera fila libre», que es lo que dice el cuaderno: en Sheets, dos que pegan en la
     misma fila se pisan y la del primero se pierde sin que nadie lo note 🟡.
   - Inmoviliza la fila 1, configuración regional España (para la coma decimal) y «cualquiera con el enlace puede editar» 🟡 (rótulos de menú sin comprobar).
   - Si en clase desaparece una fila: Archivo → Historial de versiones 🟡, o que el alumno la vuelva a pegar desde su 1.8.
   - Prueba a pegar una línea copiada desde Colab: debe repartirse en columnas (va separada por tabuladores, D19). 🟡
5. **Preparar el reparto de retos.** Lista de matriculados por apellido, y A, B, C, D en ciclo (1.º A, 2.º B, 3.º C, 4.º D, 5.º A…),
   cada uno con su fila de la hoja (punto 4). Quien no esté o llegue tarde, hace el D, que es el más rápido.
   [🔴 La lista no consta en el repositorio]
6. **Dejar escrito el mensaje del chat** (se pega a las 20:30, o en el 00:00 de la versión de 60). Lleva los pasos: en la S7
   es lo único que tienen delante, porque la hoja del alumno se sube después de clase (punto 12).

   ```
   Cuaderno 1 · Una CNN por dentro: <ENLACE_NB1> (mejor en Chrome)
   1) Ábrelo y pulsa ▶ en 1.1 y después en 1.2. Si Colab avisa de que no lo ha creado Google: «Ejecutar de todos modos».
   2) Tu reto y tu fila de la hoja: A → <nombre> (filas 3 y 4) · B → <nombre> (fila 5) · C → … · D → …
      Si no estás en la lista, haz el D y pega debajo de la última fila. Reto C: hoy solo dropout 0; el 0,5, si acabas antes de las 20:45.
   3) Después de 1.2, ve directo a 1.9 (reto e hipótesis) y luego a 1.3 → 1.4 → 1.5 → 1.8. No ejecutes todo de arriba abajo.
      En nombre_experimento: tus iniciales y el reto (p. ej. ALM_A). La celda 1.10 es del profesor: no la ejecutes.
   4) En 1.8 salen dos líneas: copia la segunda y pégala en tu fila de la hoja común: <ENLACE_HOJA>
      1.8 también descarga un CSV: es tu copia. Si el navegador pregunta, permite la descarga.
   5) Después, 1.7 (Grad-CAM). Antes de cambiar de cuaderno: Archivo → Guardar una copia en Drive. Es lo que te llevas hoy.
   ```

   En la versión de 60, quita del punto 2 la frase del C: allí hacen los dos entrenamientos.

7. **Ensayar CNN Explainer** una vez con la tarjeta del §2.1, a pantalla completa. El enlace **no** va al chat (D51).
8. **Abrir el cuaderno 2 una vez en Colab** con la demo 1: 2.1 → 2.8. Comprueba:
   - que los vídeos de 2.5 y 2.6 se reproducen (Chrome y Safari) 🟡;
   - el tiempo de 2.3 + 2.5 + 2.6 (proyección ~60–90 s 🟡; presupuesto ≤ 2 min) y el de la primera instalación [🔴 no medido];
   - que el navegador deja descargar los dos archivos de 2.8.
   - Graba 20 s de pantalla de cada vídeo anotado (Cmd+Mayús+5): es el Plan B si en clase el vídeo no se ve.
9. **Solo versión de 60 min, opcional:** en n8n, un flujo con un nodo Webhook en **POST** y su URL de prueba copiada.
   Los rótulos de los botones dependen de la versión de n8n (D41). 🟡
10. **Pestañas, en este orden:** chat · CNN Explainer · tu cuaderno 1 (con el Plan B hecho) · hoja común · cuaderno 2 · (n8n).
11. **Al empezar el bloque 7 (20:20)**, sin dejar de hablar:
    - Si en la prueba del punto 2 la instalación pasó de 3 min: pego ya el mensaje del chat y digo «pulsad 1.1 y 1.2 y seguid
      escuchando». En 8a solo recuerdo el reto, la fila y la hoja.
    - Miro la pestaña de mi cuaderno 1. Si Colab la ha desconectado 🟡: 1.1 → 1.2 → 1.10 (instalación sin medir 🔴 más ~5–7 min
      🟡). Así a las 20:40 la sesión sigue viva y la tabla del Plan B vuelve a estar en 1.8 para 8f. Si está viva, no toco nada.
12. **Después de clase:** los dos cuadernos y `docs/hoja_alumno.md` al aula virtual, con el badge ya corregido (checklist de la S7, §4).
    En la S7 la hoja del alumno es para repasar en casa: en clase, los pasos van en el mensaje del chat (punto 6). Si quieres
    que la tengan en directo, expórtala a PDF (a 10 pt y márgenes de 1,5 cm cabe en un A4), compártela desde Drive como lector y añade al mensaje
    `Hoja con los pasos: <ENLACE_HOJA_ALUMNO>`.

---

## 2. Tarjetas de consulta

### 2.1 CNN Explainer: recorrido verificado en navegador (24-sep-2026)

Web: <https://poloclub.github.io/cnn-explainer/> (Georgia Tech, licencia MIT). Red Tiny VGG: entrada 64×64×3, cuatro convoluciones
de 10 filtros 3×3 sin relleno, cada una con su ReLU, dos max pooling 2×2 y 10 clases. Tamaños: 64 → 62 → 60 → 30 → 28 → 26 → 13.

| Paso | Clic | Qué aparece | Qué digo |
|---|---|---|---|
| 0 | Abrir a pantalla completa. Cerrar el recuadro *You might also like* (abajo a la izquierda). Pulsar **Show detail** | La red entera: `input`, `conv_1_1`, `relu_1_1`, `conv_1_2`, `relu_1_2`, `max_pool_1`, `conv_2_1` … `max_pool_2`, `output`. Imagen por defecto: *espresso* | «Cada columna es una capa. Entra una foto de 64×64 en color y salen 10 probabilidades» |
| 1 | Una neurona de `conv_1_1` | **Vista intermedia**, todavía no la fórmula: canales R, G y B, un resultado por canal y la suma con el sesgo | «Así se fabrica un mapa: un resultado por color, y se suman» |
| 2 | **Segundo clic**, en uno de los cuadros *intermediate* | Ventana **Convolution**: *Input (64, 64)* → *Output (62, 62)*, núcleo 3×3 con sus pesos. Pasar el ratón mueve el núcleo | «Nueve números que recorren toda la imagen» |
| 3 | La × de la ventana y luego un espacio vacío | Vuelve la red completa | — |
| 4 | Una neurona de `relu_1_1` | **ReLU Activation**: *Input (62, 62)* → *Output (62, 62)*, `max(0, x)`. Otro clic en la misma neurona la cierra | «Lo negativo pasa a cero» |
| 5 | Una neurona de `max_pool_1` | **Max Pooling**: *Input (60, 60)* → *Output (30, 30)*, máximo de cada 2×2 | «Se queda una cuarta parte» |
| 6 | La clase *espresso* en `output` | `max_pool_2` → flatten → softmax. En el ensayo, *Output value 0.8606*: lee lo que diga la pantalla, no es un dato fijo | «Todo en una fila, y probabilidades que suman 1» |

Avisos: solo está en inglés (traduce de palabra). Con la ventana a 800 px la red se ve rota. Carga un script de `polyfill.io`,
dominio comprometido en 2024; hoy responde 403 y la página funciona sin él 🟡 (puede cambiar). Nuestro cuaderno usa el mismo
relleno que CNN Explainer: cada convolución 3×3 quita 2 píxeles de lado (28 → 26).

**Las tres preguntas**

1. «El mismo filtro de 3×3 recorre toda la foto. ¿Qué ganamos frente a aprender un peso distinto para cada píxel?»
   → Muchos menos parámetros, y el patrón se detecta esté donde esté.
2. «Después del pooling queda una cuarta parte de la imagen. ¿Qué ganamos y qué podemos perder?»
   → Se gana cálculo y tolerancia a pequeños desplazamientos; se puede perder detalle fino y la posición exacta. Enlaza con el reto B.
3. «Si le enseñamos la foto de un perro, que no está entre sus 10 clases, ¿qué contesta?»
   → Alguna de sus 10 clases: el softmax siempre reparte el 100 %. Enlaza con atajos y Grad-CAM.

### 2.2 Los retos: reparto y qué esperar

**Reparto online.** Individual (sin parejas ni salas). Por orden de la lista de matriculados, A-B-C-D en ciclo, cada uno con
su fila de la hoja, pegado en el chat con el enlace (§1, puntos 4 a 6). Quien no aparezca, D. Si una letra se queda sin nadie,
la comento con la fila del Plan B.

**Ciclo de cada alumno:** 1.1 → 1.2 → 1.9 (letra e hipótesis, antes de entrenar) → 1.3 (cambia lo que pide el reto; nombre
`iniciales_reto`, p. ej. `ALM_A`) → 1.4 → 1.5 → 1.8 → pega la **segunda** línea en **su fila** de la hoja. Todos parten de la
red por defecto (valores listados encima de 1.3). La portada del cuaderno dice lo mismo que el chat: tras 1.2, quien tenga reto
salta a 1.9. Quien siga en orden de arriba abajo entrena antes `grupo_1` con la red por defecto (20–30 s 🟡): no pasa nada, pero que no pegue esa fila.

| Reto | Qué cambian en 1.3 | Entrenos | Entrenamiento en Colab 🟡 | Qué deberían ver (medido, semilla 42) | Frase para comentarlo |
|---|---|---|---|---|---|
| **A · Profundidad** | `bloques_convolucionales` en 1; luego en 3 | 2 | ~14–19 s y ~19–26 s | 1 bloque: **173.345** parámetros, sensibilidad 97,9 %, especificidad 61,1 %. 3 bloques: **27.521**, parecida a la red por defecto (86,2 % de acierto en test) | «Con pooling, más bloques dan menos parámetros: cada bloque encoge la imagen antes de la capa densa, que en la red por defecto y en la de 1 bloque se lleva casi todos. Con 3 bloques la densa ya es pequeña y pesan más las convoluciones» |
| **B · Sin pooling** | Desmarca `usar_pooling` | 1 | ~41–58 s | **1.184.577** parámetros, 21 veces más; el doble de tiempo; test 86,5 %, sin mejora clara | «Veintiún veces más parámetros y el test no mejora claramente» |
| **C · Dropout 0 frente a 0,5** | Desmarca `usar_pooling`, `epocas` en 20; `dropout` en 0 y luego en 0,5. **En la S7, solo dropout 0**: el 0,5 es extra si acaba antes de las 20:45, y su fila de referencia la pego yo | 2 (S7: 1) | ~1,6–2,2 min cada uno; ~3,2–4,5 min en total | Dropout 0: diagnóstico **sobreajuste**, las curvas se separan. Dropout 0,5: **razonable**. El test cambia 1 punto o menos | «La lección del C está en las curvas de 1.4, no en la cifra de test» |
| **D · Compensar el desbalanceo** | Marca `compensar_desbalanceo` | 1 | ~17–24 s | **56.129** parámetros, como la red por defecto. Especificidad 77,8 % (defecto 75,6 %), sensibilidad 93,8 % (defecto 94,9 %) | «La red dice "Normal" algo más a menudo. Es un efecto pequeño, del tamaño del ruido: una tendencia» |

Los tiempos en Colab son 🟡: los calculo multiplicando el tiempo medido con 2 hilos por el factor 2,5–3,5 de `hechos_para_docs.md`. Nadie los ha medido en Colab.

**Dónde se atascan**
- **A y C:** 1.8 solo da la línea del último entrenamiento. Que peguen la primera **antes** de entrenar la segunda.
- **A con 3 bloques:** 1.3 avisa de que el último mapa mide 3×3 px y el Grad-CAM de 1.7 tendrá solo 3×3 cuadros. Es normal: que siga.
- **C:** tiene que quitar también el pooling. El deslizador enseña `0.5`, con punto. Con más de 90 s estimados, 1.3 avisa
  de que tardará (D16): es normal, que sigan. 🟡
- **Todos:** no volvieron a los valores por defecto antes de empezar; no cambiaron el nombre; 1.9 sin hipótesis (la celda lo avisa);
  ejecutaron de arriba abajo y tienen una fila `grupo_1` (que no la peguen); pegaron en una fila que no es la suya.
- **Descargas:** cada vez que el registro cambia, 1.8 descarga `experimentos.csv` (A y C, dos veces). Chrome puede preguntar
  por descargas múltiples; en Safari y Firefox puede fallar (README §3). No hace falta para la hoja común.
- **Quien pruebe 4 bloques:** sale «⛔ Con 4 bloques, kernel 3×3 y pooling, la imagen de 28×28 se queda en 0×0 píxeles…». El mensaje da la solución.
- **Los que acaben pronto:** 1.6 (qué ve cada capa) o un experimento «Libre», con su hipótesis.

**Cómo leer los resultados (tendencias, no garantías)**
- **Ruido entre semillas:** con otra semilla, la misma red por defecto pasó de 75,6 % a 58,5 % de especificidad. Diferencias
  de 1 a 3 puntos entre dos entrenamientos pueden ser azar. Lo que sí se ve claro: parámetros, tiempo y diagnóstico.
- **Validación ~95 % frente a test ~87 % no es sobreajuste:** el test es otro lote, con menos neumonías (62,5 % frente a 74,2 %),
  y en él cae sobre todo la especificidad (88,1 % → 75,6 % con la red por defecto). El cuaderno lo calcula en vivo en 1.5.
- **C en Colab:** el sobreajuste con dropout 0 salió en 3 de 3 semillas en la calibración, pero en Colab cambian los decimales.
  Di «deberían separarse», no «se separan».
- **Tamaño del mapa de Grad-CAM** (sale de la última capa de convolución, antes del pooling): red por defecto y D, 11×11 (lo dice el
  cuaderno); A con 3 bloques, 3×3 (lo avisa 1.3); A con 1 bloque, 26×26, y B y C, 24×24 🟡 (calculados, no vistos en pantalla).

---

## 3. Versión S7 · 40 min (jueves 24-sep-2026, 20:30–21:10)

Viene de «Lo mismo de una sentada» (bloque 7). Chat y micrófono abierto. Pregunto por nombre.

### 8a · Arranque · 20:30–20:32 · 2 min · teoría 0

**En pantalla**
El chat. Pego el mensaje preparado (§1, punto 6); si ya lo pegué a las 20:20 (§1, punto 11), solo lo señalo. Luego, mi cuaderno 1 en 1.1.

**Lo que digo**
Ahora os toca a vosotros: en el chat tenéis el enlace al cuaderno 1. Abridlo ya y
pulsad el triángulo de 1.1; cuando acabe, el de 1.2. Si Colab dice que el cuaderno no lo ha creado Google, dadle a «Ejecutar
de todos modos». La primera vez tarda porque instala, así que no os quedéis mirando la barra: mientras tanto os enseño cómo
funciona una red por dentro. En el mismo mensaje está vuestro reto, A, B, C o D, vuestra fila y el enlace a la hoja común. Si
no os veis en la lista, haced el D. La celda 1.10 es mía: no la ejecutéis.

**Frase clave:** «Pulsad 1.1 y 1.2 ya; mientras se instala, miramos una red por dentro.»

**Checkpoint:** no hay; el siguiente es a las 20:37.

**Si voy justo de tiempo:** pego el mensaje y no lo leo en voz alta.

### 8b · CNN Explainer exprés · 20:32–20:37 · 5 min · teoría 5

**En pantalla**
CNN Explainer a pantalla completa, pasos 0 a 5 de la tarjeta §2.1. El paso 6 (la salida) se corta.

**Lo que digo**
Esto es CNN Explainer, una web de Georgia Tech para ver una red convolucional por dentro. Está en inglés; os voy traduciendo.
La red es pequeña y clasifica fotos en diez clases. Cada columna es una capa: entra una foto de 64 por 64 en color y salen diez
probabilidades.
Pincho en un cuadrito de la primera capa. Cada cuadrito es un mapa, y aquí veis cómo se fabrica: un resultado por cada color,
rojo, verde y azul, y la suma. Pincho ahora en uno de estos cuadros intermedios. Esto es una convolución: un filtro de 3 por 3, nueve números, que recorre toda
la imagen. Muevo el ratón y el filtro se mueve; en cada posición multiplica, suma y da un número. Fijaos arriba: entra 64 y
sale 62, porque el filtro no se sale del borde. En vuestro cuaderno pasa igual: entra 28 y sale 26.
Cierro y pincho en la ReLU. Hace una sola cosa: lo negativo pasa a cero y lo positivo se queda igual.
Y el pooling. De cada cuadrado de 2 por 2 se queda con el número más alto. Entra una imagen de 60 de lado y sale una de 30.

**Frase clave:** «Nueve números que recorren toda la imagen. Y el pooling se queda con una cuarta parte.»

**Pregunta al grupo**
Después del pooling queda una cuarta parte de la imagen. ¿Qué ganamos y qué podemos perder?
(Se gana cálculo y tolerancia a pequeños desplazamientos; se pierde detalle fino. «Los del B vais a quitarlo y veréis qué pasa.»)

**Checkpoint:** no hay; el siguiente es a las 20:37.

**Si voy justo de tiempo:** la ReLU en una frase sin abrirla y la pregunta la contesto yo (3 min, −2).

### 8c · Checkpoint de datos y cómo se hace el reto · 20:37–20:39 · 2 min · teoría 0

**En pantalla**
Mi cuaderno 1: la salida de 1.2, luego 1.9 y los controles de 1.3. Después bajo unos 40 s por las salidas ya ejecutadas de mi
`ref_defecto` (curvas de 1.4, métricas de 1.5 y las dos líneas de 1.8) y enseño la fila 2 de la hoja común. 🟡 Si la sesión se
reconectó y las salidas no están, enseño la captura (§1, punto 3).

**Lo que digo**
Volvemos a Colab. Son radiografías de tórax de niños, a 28 por 28 píxeles, y no sirven para diagnosticar a nadie. Los pasos
están en el chat. Cuando acabe 1.2, id directos a 1.9, como dice la portada para quien tiene reto: elegid la letra y escribid
qué creéis que va a pasar, antes de entrenar. Luego subís a 1.3, cambiáis lo que pide el reto y en el nombre ponéis vuestras
iniciales y la letra, por ejemplo ALM_A. Ejecutáis 1.3, 1.4, 1.5 y 1.8. Esto es lo que sale con la red por defecto: las curvas,
las métricas y, en 1.8, dos líneas. La de arriba son los títulos, que ya están en la hoja; copiad la de abajo y pegadla en
vuestra fila, la que os da el chat. Al ejecutar 1.8 se os descarga un CSV: es vuestra copia, no hace falta abrirlo, y si el
navegador pregunta, permitid. Los del A entrenáis dos veces: pegad la primera línea antes de entrenar la segunda. Los del C,
hoy solo con dropout 0.

**Checkpoint**
«Todo el mundo debería ver ahora en 1.2: "Datos listos a 28×28 px: 4.708 radiografías de entrenamiento, 524 de validación y
624 de test", y debajo, que responder siempre "Neumonía" acierta el 62,5 %.»
- Si 1.1 sigue girando: que espere, sin tocar nada.
- Si 1.2 da un mensaje rojo de descarga: que la vuelva a ejecutar en un minuto (Plan B, §5). Que no suba a 64 px.
- **Variante simplificada:** quien no tenga datos a las 20:42 se olvida del reto, entrena la red por defecto sin tocar nada
  (1.3 → 1.4 → 1.5) y hace su Grad-CAM en 8e. Se lleva igualmente el entregable.

**Si voy justo de tiempo:** no leo el checkpoint; lo escribo en el chat.

### 8d · Retos individuales · 20:39–20:50 · 11 min · teoría 0

**En pantalla**
La hoja común. Alterno con el chat.

**Qué hago yo mientras (solo)**
- 20:40: en mi cuaderno 1, `nombre_experimento` = `profesor_defecto`, red por defecto, 1.3 → 1.4 → 1.5 (~20–30 s 🟡). Si la
  sesión se ha desconectado pese al punto 11 del §1: 1.1 → 1.2 antes (instalación sin medir 🔴). Es la red de mi Grad-CAM en 8e.
- 20:41: en la pestaña del cuaderno 2 (demo 1, 10 s, salto 2, confianza 0,35), pulso ▶ en 2.1, 2.2, 2.3, 2.4, 2.5 y 2.6; Colab
  las pone en cola 🟡. Tarda ~60–90 s más la instalación si la sesión se cortó 🟡. En el bloque 9 solo enseño resultados.
  🟡 Sin comprobar que la cuenta gratuita sostenga dos entornos a la vez (cuaderno 1 y cuaderno 2).
- 20:45: pego en la hoja, en la fila que le reservé, la línea de `ref_C_dropout_05_20_epocas` que traigo copiada (§1, punto 3).
- Leo la hoja según entran líneas y elijo qué comentar en 8f.

**Lo que digo**
Adelante, once minutos. Estoy en el chat y con el micro abierto. Los del C: vuestra red es la más lenta, uno o dos minutos por
entrenamiento, y 1.3 os puede avisar de que tarda. Es normal, seguid. Hoy entrenáis solo con dropout 0; la de 0,5 la pongo yo
en la hoja a las 20:45. Si tenéis la primera pegada antes de esa hora, entrenad también la de 0,5. Los que acabéis antes, abrid
1.6 y mirad qué ve cada capa.
Los del A, abrid 1.6 después de cada entrenamiento: cambia el tamaño de los mapas.

**Checkpoint (20:45)**
«Todo el mundo debería tener ya una línea en su fila de la hoja. Mirad la columna de parámetros: con 1 bloque, 173.345; con 3
bloques, 27.521; sin pooling, el B y el C, 1.184.577; el D, 56.129, igual que la red por defecto. Los del C: acabo de pegar la
fila con dropout 0,5 para que comparéis.»
- Si alguien no la tiene: en el chat, «¿en qué celda estás y qué mensaje ves?». Cada celda dice qué paso falta.
- Si la línea no se reparte en columnas: que escriba en el chat reto, parámetros, sensibilidad, especificidad y diagnóstico, y lo copio yo.
- **Variante simplificada:** el A hace solo el de 1 bloque; quien vaya perdido, red por defecto sin tocar nada y 1.7.

**Si voy justo de tiempo:** a las 20:48 digo «quien no haya acabado, que pegue lo que tenga y vaya a 1.7».

### 8e · Grad-CAM, atajos y debate · 20:50–20:57 · 7 min · teoría 2,5

**En pantalla**
Mi 1.7: seis parejas de imágenes, aciertos arriba y fallos abajo. Después, el recuadro «Para el debate» que hay debajo.

**Lo que digo**
Todos a 1.7 y pulsad. Grad-CAM pinta encima de cada radiografía las zonas que más han empujado a la red hacia su respuesta:
amarillo, mucho; morado, poco. Arriba, tres aciertos; abajo, tres fallos. En el título, lo que es de verdad, lo que dice la red
y con qué probabilidad. Cerca del 50 %, la red dudaba. El mapa sale de la última capa de convolución, que en la red por defecto
mide 11 por 11, y se amplía a 28: por eso es tan grueso. Enseña dónde miró la red, no por qué acertó o falló.
En 2018, Zech y su equipo entrenaron redes para detectar neumonía con unas 158.000 radiografías
de adultos de tres sistemas hospitalarios de Estados Unidos. En 3 de 5 comparaciones, los modelos rendían peor fuera del
hospital donde habían aprendido. Y una red adivinaba de qué hospital venía cada placa en más del 95 % de los casos, a veces
fijándose en una marca metálica de la esquina. Como en un hospital había muchas más neumonías que en otro, saber de dónde venía
la placa ya ayudaba a acertar sin mirar el pulmón. Nuestras radiografías son de niños y, según sus autores, de un único centro 🟡:
ese atajo concreto aquí no debería darse. Otros, no lo sabemos.

**Frase clave:** «Un buen resultado en el test no garantiza que el modelo mire donde creemos.»

**Pregunta al grupo**
Mirad vuestros mapas. ¿Se fijan en los pulmones o en otras zonas? ¿Os fiaríais de este modelo en otro hospital?
(En mi ejecución de referencia, 4 de los 6 mapas se concentran en el centro del tórax, entre los pulmones y no sobre ellos, y el
acierto «Normal · 50 %» sale casi vacío 🟡: en Colab puede cambiar. Pregunto si eso es mirar la neumonía o mirar otra cosa.
Pregunto por nombre a uno del A con 3 bloques, que tiene el mapa más grueso, y a uno del B o el C, que lo tiene más fino 🟡.)

**Checkpoint (20:52)**
«Todo el mundo debería ver ahora seis radiografías con su mapa de color: tres aciertos arriba y tres fallos abajo.»
- Si 1.7 dice que falta un paso: 1.4 y 1.5 primero (la celda dice cuál).
- **Variante simplificada:** quien no tenga red entrenada, red por defecto sin tocar nada (1.3 → 1.4 → 1.5, ~20–30 s 🟡) y 1.7.

**Al cerrar (20:57), la frase literal:** «Antes de cambiar de cuaderno: Archivo → Guardar una copia en Drive 🟡. Es lo que os
lleváis hoy.» Un cuaderno abierto desde GitHub no se guarda solo: quien cierre la pestaña sin la copia pierde el entregable.

**Si voy justo de tiempo:** Zech en 30 segundos (solo la marca metálica) y dos intervenciones. La frase de la copia en Drive no se corta.

### 8f · La hoja común, leída por mí · 20:57–20:58 · 1 min · teoría 0

**En pantalla**
La hoja común. Si tiene pocas filas, la tabla y el gráfico de mi 1.8 con el Plan B; si la sesión se reconectó y no rehíce el
Plan B (§1, punto 11), la captura.

**Lo que digo**
Un minuto con la hoja. [Leo lo que haya, empezando por los parámetros.] Cuando lo preparé salió esto: con pooling, más bloques,
menos parámetros; sin pooling, 21 veces más parámetros sin mejora clara en test; sin pooling y 20 épocas, dropout 0 sobreajusta.
Y ojo con las diferencias de uno a tres puntos: con otra semilla, la misma red pasó de 75,6 a 58,5 % de especificidad.

**Checkpoint:** no hay; el siguiente es a las 20:59.

**Si voy justo de tiempo:** se corta; la hoja queda en el aula virtual.

### 9a · Del cuaderno 1 al 2: detector y umbral · 20:58–21:00 · 2 min · teoría 1

**En pantalla**
Cuaderno 2 ya ejecutado (8d): primero el fotograma sin cajas de 2.2; luego 2.3, el mismo fotograma con las cajas y la gráfica de
personas según el umbral.

**Lo que digo**
Cambio de cuaderno. Este lo manejo yo; lo tendréis en el aula virtual al acabar la clase, para probarlo en casa. El detector se
llama YOLO26 nano. Por dentro es casi todo convolucional, como vuestra red, pero con unos 2,6 millones de parámetros frente a
los 56.129 de la red por defecto, y viene entrenado con fotos de 80 tipos de objeto. Aquí no se entrena nada: solo se usa, y
solo para personas. [Checkpoint: cuentan en el chat.] Con una confianza mínima de 0,35 ve 17 personas. A simple vista hay más
de 40: se le escapan las pequeñas y las que van en grupo. Con 0,10 vería 39, pero podría poner cajas donde no hay nadie. Con 0,70, 4.

**Frase clave:** «17 cajas donde hay más de 40 personas: el aforo que salga se quedará corto.»

**Checkpoint (20:59, 30 s)**
«Contad las personas de este fotograma y escribid el número en el chat.» Con el fotograma de 2.2 en pantalla, antes de enseñar
las cajas de 2.3. Leo dos o tres números y los comparo con las 17 cajas.
- Si nadie contesta en 30 s: lo cuento yo por encima y sigo.

**Si voy justo de tiempo:** solo la frase de los 2,6 millones y la de las 17 cajas; el recuento en el chat se queda.

### 9b · Aforo por zonas y entradas por la línea · 21:00–21:05 · 5 min · teoría 1

**En pantalla**
2.4 (vista previa: dos zonas y la línea de puerta morada) → 2.5 (vídeo anotado, serie temporal, tabla) → 2.6 (vídeo con contadores).

**Lo que digo**
Las zonas se escriben en porcentaje de la imagen, así valen para cualquier resolución. Aquí son la mitad izquierda y la derecha;
en una cafetería serían la barra y las mesas. En 2.5 cada persona lleva una caja y un número. En la izquierda hay como máximo
14 personas y 11 de media; en la derecha, 13 y 9,5. Son cifras cortas, por lo que acabamos de ver del umbral.
Para saber si alguien cruza la puerta hay que seguirlo de un fotograma al siguiente. Eso es el seguimiento, el tracking: cada
persona conserva su número mientras la red la ve. Si alguien queda tapado y reaparece con otro número, puede contar dos veces.
En estos 10 segundos, la línea cuenta 8 entradas y 8 salidas.

**Frase clave:** «Sin seguimiento no se distingue a quien cruza de quien aparece al otro lado.»

**Checkpoint (21:03)**
«¿Veis moverse el vídeo en mi pantalla? Si se os queda congelado, decidlo en el chat.»
- Si se congela: lo paro, enseño la serie temporal de 2.5 y leo las cifras. Si no se reproduce en Colab: la grabación del ensayo (§1, punto 8).
- **Variante simplificada:** sin vídeos; solo la tabla de 2.5 y el resultado de 2.6.

**Si voy justo de tiempo:** solo 2.5; de 2.6, la cifra.

### 9c · Qué sale del cuaderno · 21:05–21:07 · 2 min · teoría 0

**En pantalla**
Ejecuto en directo las dos celdas de 2.8 (tardan segundos; el navegador puede pedir permiso para descargar dos archivos):
las primeras filas del CSV y el JSON de resumen, con `webhook_n8n` vacío.

**Lo que digo**
Del cuaderno solo salen números: un CSV con las personas de cada zona en cada instante y este resumen, sin imágenes ni
identificadores. Este resumen puede entrar en un flujo de automatización; debajo, el cuaderno describe uno que avisa si una zona
pasa del aforo. Un detalle: el máximo es el de un solo fotograma y puede ser un parpadeo del detector; un sistema real usaría el
máximo que se mantiene unos segundos.

**Checkpoint:** no hay.

**Si voy justo de tiempo:** se corta.

### 9d · POC ≠ producción, otros sectores, privacidad · 21:07–21:10 · 3 min · teoría 0,5

**En pantalla**
2.9 Cierre: límites, «POC ≠ producción» y otros sectores.

**Lo que digo**
Lo que habéis visto hoy son dos prototipos. Para llevar el contador a un local de verdad hay que medir el error con vídeo propio,
decidir dónde se procesa y qué se guarda, cumplir la protección de datos y mirar la licencia: Ultralytics es AGPL, y para un
producto cerrado vende una licencia de pago. Ese salto es trabajo de consultoría, como el que hacéis en las actividades.
La lógica de contar por zonas se mueve bien a otros sectores. En retail, abrir otra caja cuando se forma cola. En transporte,
ver cómo se llena un andén. En eventos, el aforo de los accesos. En industria, avisar si alguien entra en la zona de una máquina,
siempre como apoyo a los sistemas de seguridad certificados.
Quien no haya guardado todavía la copia de su cuaderno 1 en Drive, que lo haga ahora: es lo que os lleváis hoy.

**Frase clave:** «Contar no es identificar, pero grabar a una persona reconocible es tratar sus datos aunque solo queramos un número.»

**Pregunta al grupo**
«A las 9:02 entró 1 persona» es un número. ¿Podría servir para saber quién era? ¿Cómo lo evitaríais?

**Checkpoint:** no hay.

**Si voy justo de tiempo:** solo POC ≠ producción y la copia en Drive; la pregunta queda en el cuaderno.

### Colchón de la versión de 40

- **Si sobra tiempo:** en mi pantalla, 1.6 con una «Neumonía aleatoria», o la salida de CNN Explainer (paso 6).
- **Si falta, por orden:** (1) el bloque 9 pasa a demo de 5 min, como prevé el esqueleto: 20:58–20:59 puente y 17 cajas,
  20:59–21:01:30 vídeo de 2.5 y cifra de 2.6, 21:01:30–21:03 POC ≠ producción, sectores y recordatorio de la copia en Drive
  (1 + 2,5 + 1,5 = 5; −7 min);
  (2) CNN Explainer a 3 min (−2); (3) 8f se corta (−1).
- **No se toca:** 8d y 8e. Son el entregable.
- Si el bloque 8 empieza tarde, se mantiene el orden y se aplican los recortes; las horas se desplazan.

---

## 4. Versión completa · 60 min

### 1a · Arranque · 00:00–02:00 · 2 min · teoría 0

**En pantalla**
El chat con el mensaje preparado (§1, punto 6). Luego, mi cuaderno 1 en 1.1.

**Lo que digo**
Vamos con la última parte práctica. En el chat tenéis el enlace al cuaderno 1. Abridlo y pulsad el triángulo de 1.1; cuando
acabe, el de 1.2. Si Colab dice que el cuaderno no lo ha creado Google, dadle a «Ejecutar de todos modos». La primera vez tarda
porque instala; mientras tanto os enseño cómo funciona una red por dentro. En el mismo mensaje está vuestro reto y el enlace a la
hoja común, que usaremos dentro de un cuarto de hora. La celda 1.10 es mía: no la ejecutéis.

**Frase clave:** «Pulsad 1.1 y 1.2 ya; mientras se instala, miramos una red por dentro.»

**Checkpoint:** no hay; el siguiente es a las 10:00.

**Si voy justo de tiempo:** pego el mensaje sin leerlo.

### 1b · CNN Explainer: la red y la convolución · 02:00–05:00 · 3 min · teoría 3

**En pantalla**
CNN Explainer a pantalla completa, pasos 0 a 2 de la tarjeta §2.1.

**Lo que digo**
Esto es CNN Explainer, una web de Georgia Tech para ver una red convolucional por dentro. Está en inglés; os voy traduciendo.
La red clasifica fotos en diez clases, como pizza, koala o espresso. Cada columna es una capa: entra una foto de 64 por 64 en
color y salen diez probabilidades. Pulso «Show detail» y aparece el tamaño de cada capa.
Pincho en un cuadrito de la primera capa. Cada cuadrito es un mapa, y aquí veis cómo se fabrica: un resultado por cada color y
la suma. Pincho ahora en uno de estos cuadros intermedios. Esto es una convolución: un filtro de 3 por 3, nueve números, que recorre toda la imagen. Muevo el
ratón y se mueve; en cada posición multiplica, suma y da un número. Esta capa tiene diez filtros y por eso da diez mapas. Fijaos
arriba: entra 64 y sale 62, porque el filtro no se sale del borde. Vuestro cuaderno hace lo mismo: entra 28 y sale 26.

**Frase clave:** «Nueve números que recorren toda la imagen.»

**Pregunta al grupo**
El mismo filtro de 3 por 3 recorre toda la foto. ¿Qué ganamos frente a aprender un número distinto para cada píxel?
(Muchos menos parámetros, y el patrón se detecta esté donde esté.)

**Checkpoint:** no hay; el siguiente es a las 10:00.

**Si voy justo de tiempo:** no me paro en la vista intermedia; paso al segundo clic.

### 1c · CNN Explainer: ReLU, pooling y salida · 05:00–08:30 · 3,5 min · teoría 3,5

**En pantalla**
Pasos 3 a 6 de la tarjeta §2.1.

**Lo que digo**
Cierro y pincho en la ReLU, la capa de al lado. Hace una sola cosa: lo negativo pasa a cero y lo positivo se queda igual. Sin
ese paso, apilar capas serviría de poco.
Ahora el pooling. De cada cuadrado de 2 por 2 se queda con el número más alto. Entra una imagen de 60 de lado y sale una de 30.
[Pregunta.]
Última parada: pincho en la clase espresso. Todo lo anterior se estira en una fila y una función, la softmax, lo convierte en
probabilidades que suman 1. Aquí dice espresso con [lo que marque la pantalla]. Os dejo una pregunta para dentro de media hora:
si le enseño la foto de un perro, que no está entre sus diez clases, ¿qué contesta?

**Frase clave:** «El pooling se queda con el máximo de cada 2 por 2: de 60 de lado pasa a 30.»

**Pregunta al grupo**
Después del pooling queda una cuarta parte de la imagen. ¿Qué ganamos y qué podemos perder?
(Cálculo y tolerancia a pequeños desplazamientos; se pierde detalle fino y la posición exacta. «Los del B lo vais a quitar.»)

**Checkpoint:** no hay; el siguiente es a las 10:00.

**Si voy justo de tiempo:** me salto la salida y la pregunta del perro (−1 min).

### 1d · Checkpoint de datos · 08:30–10:00 · 1,5 min · teoría 0

**En pantalla**
Mi cuaderno 1: la salida de 1.2 (rejilla de ejemplos, gráfico de clases y la referencia tonta).

**Lo que digo**
Volvemos a Colab. Vuestra red va a hacer lo mismo con radiografías de tórax de niños, en gris y a 28 por 28 píxeles. No sirven
para diagnosticar a nadie: son datos educativos. Mirad la última línea de 1.2. En entrenamiento hay casi tres neumonías por cada
radiografía normal. Una red que dijera siempre «Neumonía» acertaría el 62,5 % en test sin mirar nada. Ese es el listón.

**Checkpoint**
«Todo el mundo debería ver ahora en 1.2: "Datos listos a 28×28 px: 4.708 radiografías de entrenamiento, 524 de validación y
624 de test", y debajo la referencia tonta del 62,5 %.»
- Si 1.1 sigue girando: que espere.
- Si 1.2 da un mensaje rojo de descarga: que la repita en un minuto (Plan B, §5). Que no suba a 64 px.
- **Variante simplificada:** quien no tenga datos a las 11:00 sigue mi pantalla en 2a y se engancha en 2b.

**Si voy justo de tiempo:** no explico el listón; está escrito en la celda.

### 2a · Sígueme en pantalla: la red por defecto · 10:00–16:00 · 6 min · teoría 1

**En pantalla**
1.3 (diagrama, tabla por capa, 56.129 parámetros) → 1.4 (curvas en vivo y diagnóstico) → 1.5 (métricas y matriz de confusión).

**Lo que digo**
Primero todos la misma red, la que viene por defecto, para que veáis qué sale en cada celda. Bajad a 1.3 y, sin tocar nada,
pulsad el triángulo. Esta es la ficha de la red: dos bloques de convolución, ReLU y pooling, y al final una capa densa. Mirad los
tamaños: 28, 26, 13, 11, 5. Encoge igual que en CNN Explainer: cada convolución quita 2 píxeles y el pooling deja la mitad de
lado. Tiene 56.129 parámetros, los números que la red ajusta al entrenar, y
51.264 están en la capa densa del final. Acordaos para el reto A.
Ahora 1.4. Las curvas se dibujan época a época: azul, entrenamiento; naranja, validación, que son imágenes que la red no usa para
aprender. Si van juntas, bien. Si la pérdida de validación se separa hacia arriba, la red está memorizando. Debajo sale el
diagnóstico y la regla que lo decide.
Y 1.5, el examen: 624 radiografías que no ha visto nunca. Aquí importan dos números. La sensibilidad: de las neumonías, cuántas
detecta. La especificidad: de las normales, cuántas deja tranquilas. En un cribado se suele priorizar la sensibilidad, porque una
neumonía que se escapa suele ser peor que una falsa alarma, pero ese equilibrio lo decide el equipo clínico. Veréis que en
validación acierta más que en test. No tiene por qué ser memorizar: el test viene de otro lote, con menos neumonías, y ahí falla
más con las normales. El cuaderno lo explica debajo.

**Frase clave:** «Sensibilidad: de las neumonías, cuántas detecta. Especificidad: de las normales, cuántas deja tranquilas.»

**Checkpoint (16:00)**
«Todo el mundo debería ver ahora 56.129 parámetros en la ficha, "razonable (aún mejoraba)" en el diagnóstico de 1.4 y, en 1.5,
un acierto en test de alrededor del 87 %, unos 25 puntos por encima de responder siempre "Neumonía".»
- Si 1.4 dice que falta un paso: la celda dice cuál, normalmente 1.3.
- Si le salen otros parámetros: ha tocado algo en 1.3. Que vuelva a los valores por defecto (listados encima de 1.3).
- **Variante simplificada:** quien vaya lento se salta 1.5 ahora; la hará en su reto.

**Si voy justo de tiempo:** no espero a que todos acaben 1.5; explico la sensibilidad en mi pantalla y paso a 2b.

### 2b · Retos individuales · 16:00–31:00 · 15 min · teoría 0

**En pantalla**
1.9 y 1.3 en mi cuaderno (1,5 min). Después, la hoja común; alterno con el chat.

**Lo que digo**
Ahora cada uno su reto; la letra está en el chat. Primero 1.9: elegid vuestra letra y escribid qué creéis que va a pasar. Antes
de entrenar, no después. Luego subís a 1.3, cambiáis lo que pide el reto y en el nombre ponéis vuestras iniciales y la letra, por
ejemplo ALM_A. La secuencia es 1.3, 1.4, 1.5 y 1.8. En 1.8 salen dos líneas: la de arriba son los títulos, que ya están en la
hoja; copiad la de abajo y pegadla en vuestra fila, la que os da el chat. Al ejecutar 1.8 se os descarga un CSV: es vuestra
copia; si el navegador pregunta, permitid. Los del A y los del C entrenáis dos veces: pegad la primera línea antes de entrenar la segunda. Los del C,
vuestra red es la más lenta, uno o dos minutos por entrenamiento, y 1.3 os puede avisar de que tarda. Es normal, seguid. Tenéis
quince minutos. Estoy en el chat y con el micro abierto.

**Qué hago yo mientras:** leo la hoja según entra, contesto en el chat y elijo qué filas comentar en 2c. A quien acabe, 1.6 o un experimento «Libre».

**Frase clave:** «Escribid la hipótesis antes de pulsar 1.4.»

**Checkpoint (24:00)**
«Todo el mundo debería tener ya al menos una línea en su fila de la hoja. Mirad la columna de parámetros: con 1 bloque, 173.345; con 3
bloques, 27.521; sin pooling, el B y el C, 1.184.577; el D, 56.129, igual que la red por defecto.»
- Si alguien no la tiene: «¿en qué celda estás y qué mensaje ves?». Cada celda dice qué paso falta.
- Si la línea no se reparte en columnas: que escriba en el chat reto, parámetros, sensibilidad, especificidad y diagnóstico, y lo copio yo.
- **Variante simplificada:** el C, solo dropout 0, comparando con la fila `ref_C_dropout_05_20_epocas`, que pego yo en la hoja
  (§1, punto 3); el A, solo 1 bloque; quien esté perdido, la red por defecto y 1.7.

**Si voy justo de tiempo:** a las 28:00, «quien no haya acabado, que pegue lo que tenga».

### 2c · La hoja común, entre todos · 31:00–35:00 · 4 min · teoría 0,5

**En pantalla**
La hoja común, ordenada por reto 🟡. Si tiene pocas filas, la tabla y el gráfico de mi 1.8 con el Plan B.

**Lo que digo**
Vamos a mirar la hoja. [Leo lo que haya, reto por reto.] Cuando lo preparé salió esto. Con pooling, más bloques dan menos
parámetros, porque cada bloque encoge la imagen antes de la capa densa: con un bloque, 173.345; con tres, 27.521, seis veces menos.
Sin pooling, 21 veces más parámetros y el doble de tiempo, y el test no mejora claramente. Sin pooling, con 20 épocas y dropout 0,
el diagnóstico dice sobreajuste y las curvas se separan; con 0,5, razonable, y el test apenas cambia. Compensar el desbalanceo sube la
especificidad unos dos puntos y baja la sensibilidad uno. Eso es poco: con otra semilla, la misma red por defecto pasó de 75,6 a
58,5 % de especificidad. Diferencias de uno a tres puntos pueden ser azar.

**Frase clave:** «Diferencias de uno a tres puntos pueden ser azar; los parámetros, el tiempo y el diagnóstico, no.»

**Pregunta al grupo**
¿Más parámetros han dado siempre mejores resultados en test? (No: el B tiene 21 veces más y no mejora claramente.)

**Checkpoint:** no hay; el siguiente es a las 38:00.

**Si voy justo de tiempo:** la leo yo en un minuto, sin pregunta (−3).

### 3a · Grad-CAM en su cuaderno · 35:00–38:00 · 3 min · teoría 1

**En pantalla**
Mi 1.7 (red por defecto): seis parejas de imágenes, aciertos arriba y fallos abajo.

**Lo que digo**
Todos a 1.7 y pulsad. Grad-CAM pinta encima de cada radiografía las zonas que más han empujado a la red hacia su respuesta:
amarillo, mucho; morado, poco. Arriba, tres aciertos; abajo, tres fallos. En el título, lo que es de verdad, lo que dice la red y
con qué probabilidad; cerca del 50 %, la red dudaba. El mapa sale de la última capa de convolución, que en la red por defecto mide
11 por 11, y se amplía a 28: por eso es grueso y tiene un marco sin color. Los del A con tres bloques veréis un mapa todavía más
grueso; los del B y el C, uno más fino.

**Frase clave:** «Enseña dónde miró, no por qué acertó.»

**Checkpoint (38:00)**
«Todo el mundo debería ver ahora seis radiografías con su mapa de color: tres aciertos arriba y tres fallos abajo.»
- Si 1.7 dice que falta un paso: 1.4 y 1.5 primero.
- **Variante simplificada:** quien no tenga red entrenada, red por defecto (1.3 → 1.4 → 1.5, ~20–30 s 🟡) y 1.7.

**Si voy justo de tiempo:** no explico el tamaño del mapa.

### 3b · Atajos: Zech et al. (2018) · 38:00–40:00 · 2 min · teoría 2

**En pantalla**
El recuadro «Para el debate: aprendizaje por atajos», debajo de 1.7.

**Lo que digo**
En 2018, Zech y su equipo entrenaron redes para detectar neumonía con unas 158.000 radiografías de
adultos de tres sistemas hospitalarios de Estados Unidos. En 3 de 5 comparaciones, los modelos rendían peor fuera del hospital
donde habían aprendido. Y una red adivinaba de qué hospital venía cada placa en más del 95 % de los casos, a veces fijándose en una
marca metálica que los técnicos ponen en la esquina. Como en un hospital había muchas más neumonías que en otro, saber de dónde
venía la placa ya ayudaba a acertar sin mirar el pulmón. Nuestras radiografías son de niños y, según sus autores, de un único
centro 🟡: ese atajo concreto aquí no debería darse. Otros, no lo sabemos.
(Si en la S4 se contó el caso de la regla en las fotos de lunares, lo enlazo en una frase.)

**Frase clave:** «Un buen resultado en el test no garantiza que el modelo mire donde creemos.»

**Checkpoint:** no hay.

**Si voy justo de tiempo:** solo la marca metálica, en 30 segundos.

### 3c · Debate · 40:00–45:00 · 5 min · teoría 0,5

**En pantalla**
Mi Grad-CAM y las dos preguntas del recuadro.

**Qué hago yo mientras:** al empezar (40:00), en otra pestaña, pulso ▶ en 2.1 y 2.2 del cuaderno 2 para que instale y descargue
(instalación sin medir 🔴). No lo enseño; así 4b no espera a la instalación si la sesión abierta antes de clase se ha reciclado 🟡.

**Lo que digo**
Ahora vosotros, con el micro o por el chat. [Pregunto por nombre.] … Y la pregunta del perro: la red contestaría una de sus diez
clases, porque la softmax siempre reparte el 100 %. No sabe decir «no sé». Por eso conviene mirar dónde se fija.

**Pregunta al grupo**
Mirad vuestros mapas. ¿Se fijan en los pulmones o en otras zonas? ¿Os fiaríais de este modelo en otro hospital?
(En mi ejecución local, varios mapas se concentran en el centro del tórax, entre los pulmones. Uno del A con 3 bloques y uno
del B o el C comparan el grosor de sus mapas 🟡.)

**Checkpoint:** no hay; el siguiente es a las 51:00.

**Si voy justo de tiempo:** dos intervenciones y cierro con el perro.

### 4a · Del cuaderno 1 al 2 · 45:00–46:30 · 1,5 min · teoría 1

**En pantalla**
1.11 (puente) y, en otra pestaña, la portada del cuaderno 2 con el aviso de privacidad.

**Lo que digo**
El detector de personas del cuaderno 2 se llama YOLO26 nano. Es casi todo convolucional, como vuestra red, pero con unos 2,6
millones de parámetros frente a los 56.129 de la red por defecto, y un par de bloques de atención. Viene entrenado con fotos de 80
tipos de objeto; aquí no se entrena nada y solo contamos personas. Este cuaderno lo manejo yo; lo tendréis en el aula virtual al
acabar la clase. Una
advertencia antes: grabar a personas reconocibles es tratar datos personales, aunque al final solo queramos un número.

**Frase clave:** «La misma idea que vuestra red, 2,6 millones de parámetros y ya entrenada.»

**Checkpoint:** no hay.

**Si voy justo de tiempo:** solo la primera frase.

### 4b · El detector y el umbral · 46:30–50:00 · 3,5 min · teoría 0,5

**En pantalla**
2.2 (demo 1: primer fotograma; proceso a 1280×720, 25 fps, 125 fotogramas) → 2.3 en directo: cajas y gráfica de personas según el umbral.

**Lo que digo**
Vídeo de demo: gente caminando en un interior, con la cámara alta. Proceso 10 segundos y el detector mira uno de cada dos
fotogramas, 125 imágenes. [2.3] Cada caja es una persona que el detector ve con una confianza mínima de 0,35. Ve 17. Contad
vosotros: hay más de 40. Se le escapan las pequeñas, las del fondo y las que van en grupo. La gráfica es el mismo fotograma con
otros umbrales: con 0,10 ve 39, pero puede poner cajas donde no hay nadie; con 0,70, 4.
(No lo dejes por debajo de 0,25: el seguimiento de 2.5 y 2.6 no empieza a seguir cajas tan dudosas y el cuaderno lo avisa.)

**Frase clave:** «17 cajas donde hay más de 40 personas: el aforo que salga se quedará corto.»

**Pregunta al grupo**
Si esto controla el aforo de un local, ¿qué preferís: que cuente de menos o que se invente gente?

**Checkpoint:** no hay; el siguiente es a las 51:00.

**Si voy justo de tiempo:** no muevo el umbral; leo las tres cifras de la gráfica.

### 4c · Aforo por zonas y entradas por la línea · 50:00–54:00 · 4 min · teoría 1

**En pantalla**
2.4 (vista previa de zonas y línea morada) → 2.5 en directo (vídeo, serie temporal, tabla) → 2.6 en directo (vídeo con contadores).
Proyección en Colab: 2.5 + 2.6 caben en ~1–1,5 min 🟡; lanzo 2.5 y hablo de las zonas mientras procesa.

**Lo que digo**
Las zonas se escriben en porcentaje de la imagen, así valen para cualquier resolución. Aquí, mitad izquierda y mitad derecha; en
una cafetería las llamaríais barra y mesas. En 2.5, en la izquierda hay como máximo 14 personas y 11 de media; en la derecha, 13 y
9,5. Son cifras cortas, por lo que acabamos de ver del umbral.
Para saber si alguien cruza la puerta hay que seguirlo de un fotograma al siguiente. Eso es el seguimiento: cada persona conserva
su número mientras la red la ve. Si alguien queda tapado y reaparece con otro número, puede contar dos veces. En estos 10 segundos,
la línea cuenta 8 entradas y 8 salidas.

**Frase clave:** «Sin seguimiento no se distingue a quien cruza de quien aparece al otro lado.»

**Checkpoint (51:00)**
«¿Veis moverse el vídeo en mi pantalla? Si se os queda congelado, decidlo en el chat.»
- Si se congela: lo paro y enseño la serie temporal. Si no se reproduce en Colab: la grabación del ensayo.
- **Variante simplificada:** 2.5 y 2.6 ejecutadas antes de clase; solo enseño resultados.

**Si voy justo de tiempo:** 2.6 solo con la cifra.

### 4d · Exportar y puente opcional a n8n · 54:00–57:00 · 3 min · teoría 0

**En pantalla**
2.8: primeras filas del CSV y el JSON. **Opcional:** n8n con el Webhook en escucha; pego su URL en `webhook_n8n`, ejecuto la celda
y enseño el JSON que llega dentro de `body`.

**Lo que digo**
Del cuaderno solo salen números: un CSV con las personas de cada zona en cada instante y un resumen, sin imágenes ni
identificadores. [Con n8n] Pego la dirección del webhook, ejecuto y aquí llega: el máximo por zona, las entradas y las salidas.
Desde aquí, un nodo IF compararía el máximo con el aforo permitido y mandaría un aviso. No lo montamos hoy. [Sin n8n] Esto es lo
que recibiría un flujo de automatización; debajo, el cuaderno describe uno de alerta de aforo en cuatro pasos. En los dos casos:
el máximo es el de un solo fotograma y puede ser un parpadeo del detector.

**Frase clave:** «Del cuaderno solo salen recuentos.»

**Checkpoint:** no hay.

**Si voy justo de tiempo:** sin n8n (−2). Si no abro n8n en pantalla, no lo nombro más allá de lo que pone el cuaderno.
Si da 404: método del Webhook en POST y la escucha de prueba activada (D41) 🟡.

### 4e · POC ≠ producción, otros sectores, privacidad · 57:00–60:00 · 3 min · teoría 0,5

**En pantalla**
2.9 Cierre.

**Lo que digo**
Lo que habéis visto hoy son dos prototipos. Para llevar el contador a un local de verdad hay que medir el error con vídeo propio,
decidir dónde se procesa y qué se guarda, cumplir la protección de datos y mirar la licencia: Ultralytics es AGPL, y para un
producto cerrado vende una licencia de pago. Ese salto es trabajo de consultoría, como el de vuestras actividades.
La lógica de contar por zonas se mueve bien a otros sectores. En retail, abrir otra caja cuando se forma cola. En transporte, ver
cómo se llena un andén. En eventos, el aforo de los accesos. En industria, avisar si alguien entra en la zona de una máquina,
siempre como apoyo a los sistemas de seguridad certificados.
Guardad una copia de vuestro cuaderno 1 en Drive: es lo que os lleváis hoy.

**Frase clave:** «Contar no es identificar, pero grabar a una persona reconocible es tratar sus datos aunque solo queramos un número.»

**Pregunta al grupo**
«A las 9:02 entró 1 persona» es un número. ¿Podría servir para saber quién era? ¿Cómo lo evitaríais?

**Checkpoint:** cierre. Compruebo en el chat que los que han hecho el reto tienen su línea en la hoja.

**Si voy justo de tiempo:** solo POC ≠ producción y la copia en Drive.

### Colchón de la versión de 60

- **Si sobra tiempo:** 1.6 en mi pantalla con una «Neumonía aleatoria»; más experimentos «Libre»; 2.7 permanencia (5,6 s y 5,7 s
  de media, pero 30 de 35 estancias recortadas por los bordes del tramo: el cuaderno avisa de que 10 s dicen poco); la demo 2
  (andén), solo si ya está descargada (128 MB, ~80–110 s en Colab 🟡).
- **Si falta, por orden:** (1) 4d sin n8n (−2); (2) 2c la leo yo en un minuto (−3); (3) salida y perro de CNN Explainer (−1);
  (4) en 2a solo 1.3 en mi pantalla y cada uno entrena ya su reto (−3).
- **No se toca:** 2b y 3a. Son el entregable.

---

## 5. Plan B si algo falla

| Qué falla | Cómo se nota | Qué hago | Qué hace el alumno |
|---|---|---|---|
| **Zenodo lento o caído** (1.2) | 1.2 tarda más de un minuto o sale «No he podido descargar PneumoniaMNIST…» | Nada que tocar: la celda prueba Zenodo (se rinde si no responde en 60 s) y luego un espejo para 28 px con comprobación MD5 (D14). Que la repitan en un minuto | Mantener 28 px: 64 px no tiene espejo. Si sigue, mi pantalla y la variante simplificada |
| **Colab lento o no conecta** | ▶ girando varios minutos en 1.1 | Por eso se lanza antes de CNN Explainer. Si no conecta en 2 min: otra pestaña o Chrome | Si nada funciona, sigue mi pantalla y hace el cuaderno después (aula virtual) |
| **Entrenamiento lento** | 1.3 avisa de más de 90 s | En el C es normal (~1,6–2,2 min por entrenamiento 🟡). En otros: ¿ha quitado el pooling sin querer o subido a 64 px? | C: solo dropout 0 y compara con la fila `ref_C_dropout_05_20_epocas` que pego en la hoja |
| **Sin GPU** | 1.1 dice «Tarjeta gráfica (GPU): no, y no hace falta» | Normal: todo está pensado para CPU. No cambiar el tipo de entorno en clase: reinicia la sesión y se pierde lo hecho 🟡 | Nada |
| **Configuración imposible** | «⛔ Con 4 bloques, kernel 3×3 y pooling, la imagen de 28×28 se queda en 0×0 píxeles…» | Que lea el mensaje: 3 bloques como máximo | Vuelve a la red de su reto |
| **Alumno perdido** | «No sé dónde estoy» en el chat | «¿En qué celda estás y qué mensaje ves?». Cada celda dice qué paso falta. Las celdas se pueden repetir: si hace falta, desde 1.1 en orden | Variante: red por defecto sin tocar nada y 1.7 |
| **Cuaderno roto o sesión reiniciada** | Error inesperado, «reconectando», variables perdidas | Volver a ejecutar 1.1 → 1.2 → su reto. Si falla a muchos a la vez: **mi Plan B de 1.10**, ya ejecutado antes de clase: enseño la tabla y el gráfico de 1.8 (o la captura y el CSV) y mi Grad-CAM, y la clase sigue en mi pantalla | Hace el cuaderno después; el entregable se entrega igual |
| **Colab pide reiniciar tras instalar** | Aviso de reinicio en 1.1 o 2.1 | 🟡 No debería pasar (resolución de paquetes en `e_colab.md`), sin comprobar con Python 3.13. Aceptar y volver a ejecutar desde 1.1 (o 2.1). Si la instalación falla o lo sigue pidiendo: paleta de comandos → «Change runtime version» → 2026.07, que es Python 3.12 (README §3) 🟡; no se mantiene entre sesiones | Igual |
| **La línea no se reparte en la hoja** | Toda la línea cae en la columna A | «Dividir texto en columnas» en el menú Datos 🟡, o que escriban en el chat reto, parámetros, sensibilidad, especificidad y diagnóstico | Escribirlo en el chat |
| **Una fila de la hoja desaparece o la pisa otro** | Un alumno dice que su línea no está | Archivo → Historial de versiones 🟡, o que la vuelva a pegar desde su 1.8 en su fila | Volver a pegar |
| **Mapa de Grad-CAM vacío** | Un mapa sin color en 1.7 | El cuaderno lo explica (pasó en 2 de 197 predicciones de «Normal», D18) | Nada |
| **CNN Explainer no carga o se ve rota** | Página en blanco o capas descolocadas | Pantalla completa (a 800 px se rompe). Si no carga, explico sobre la ficha de 1.3: 28 → 26 → 13 → 11 → 5 | — |
| **El vídeo no se reproduce** (2.5, 2.6) | Recuadro negro o sin controles | Probar en Chrome. Si no, la serie temporal de 2.5, las cifras y la grabación del ensayo | — |
| **El vídeo se congela al compartir pantalla** | Lo dicen en el chat | Pausar, narrar y enseñar la serie temporal | — |
| **Cuaderno 2 lento en Colab** | La 2.3 avisa: estimación de más de 100 s para 2.5 + 2.6 | Subir `saltar_fotogramas` a 3 en 2.2 (o bajar los segundos) y volver a ejecutar 2.2, 2.3 y 2.4, como propone el aviso, o tirar de la grabación del ensayo. Si la sesión se recicló, 2.1 y 2.2 primero: en la versión de 60 ya van lanzadas en 3c; en la de 40, en 8d | — |
| **Webhook de n8n da 404** | Mensaje de 2.8 con las causas típicas | Webhook en POST, escucha de prueba activada o flujo activo (D41) 🟡. Si no sale en un minuto, enseño el JSON impreso | — |

---

## 6. Resultados de referencia medidos

Ejecución final del 24-sep-2026 con `scripts/verificar.py`: Apple M3 Pro, Python 3.12.7, TF 2.21.0, Keras 3.15.1, ultralytics
8.4.161. 0 errores. El cuaderno 1 también pasó con las versiones de Colab (Python 3.13, Keras 3.13.2) con métricas idénticas.

### Cuaderno 1 · Plan B (semilla 42, 28 px)

Con todos los hilos; entre paréntesis, con 2 hilos (cambian los decimales, no las tendencias).

| Experimento | Reto | Parámetros | Tiempo | Acierto val. | Acierto test | Sensibilidad | Especificidad | AUC | Diagnóstico |
|---|---|---|---|---|---|---|---|---|---|
| ref_defecto | — | 56.129 | 4,0 s (6,7) | 94,5 % | 87,7 % (86,9) | 94,9 % (94,9) | 75,6 % (73,5) | 0,936 | razonable (aún mejoraba) |
| ref_A_1_bloque | A | 173.345 | 3,0 s (5,5) | 95,2 % | 84,1 % (84,5) | 97,9 % (97,9) | 61,1 % (62,0) | 0,924 | razonable (aún mejoraba) |
| ref_A_3_bloques | A | 27.521 | 4,2 s (7,4) | 92,6 % | 86,2 % (85,7) | 94,6 % (94,6) | 72,2 % (70,9) | 0,936 | razonable (aún mejoraba) |
| ref_B_sin_pooling | B | 1.184.577 | 7,7 s (16,5) | 95,6 % | 86,5 % (87,3) | 97,9 % (97,9) | 67,5 % (69,7) | 0,927 | razonable |
| ref_C_dropout_0_20_epocas | C | 1.184.577 | 15,0 s (38,3) | 95,8 % | 87,5 % (86,9) | 98,2 % (98,5) | 69,7 % (67,5) | 0,939 | **sobreajuste** |
| ref_C_dropout_05_20_epocas | C | 1.184.577 | 14,8 s (37,6) | 96,0 % | 88,1 % (87,8) | 97,9 % (97,9) | 71,8 % (70,9) | 0,946 | razonable |
| ref_D_compensar_desbalanceo | D | 56.129 | 4,1 s (6,9) | 92,6 % | 87,8 % (88,1) | 93,8 % (94,4) | 77,8 % (77,8) | 0,931 | razonable (aún mejoraba) |

Referencia tonta (siempre «Neumonía»): 62,5 % de acierto en test, sensibilidad 100 %, especificidad 0 %.
Datos: 4.708 de entrenamiento, 524 de validación y 624 de test; 74,2 % de neumonías en entrenamiento y 62,5 % en test.

### Cuaderno 2 (confianza 0,35, salto 2, 10 s)

| | Demo 1 · personas caminando (0–10 s) | Demo 2 · andén de metro (22–32 s) |
|---|---|---|
| Fotogramas procesados · resolución | 125 · 1280×720 | 150 · 720×1280 (vertical) |
| Personas en el primer fotograma (2.3) | 17 con 0,35 · 39 con 0,10 · 4 con 0,70 (a simple vista, más de 40) | 9 con 0,35 · 16 con 0,10 · 2 con 0,70 |
| Zonas (máximo / media) | Izquierda 14 / 11,0 · Derecha 13 / 9,5 | Izquierda 6 / 2,9 · Derecha 7 / 4,4 |
| Línea de puerta «auto» | Horizontal al 50 %: **8 entradas / 8 salidas** | Diagonal `62,15; 22,100`: **9 entradas (se alejan del tren) / 3 salidas (van hacia el tren)** |
| Permanencia media | 5,6 s / 5,7 s (35 estancias, 30 recortadas por los bordes del tramo) | 2,3 s / 2,1 s (29 estancias, 8 recortadas) |

En la demo 2 la línea no cuenta «subir al tren»: cuenta movimiento que se aleja del tren o va hacia él.

### Tiempos

| Qué | Todos los hilos | Limitado (proxy de Colab) | Proyección Colab gratuito 🟡 |
|---|---|---|---|
| Cuaderno 1 · 1.4 con la red por defecto (8 épocas) | 5,5 s (entrenamiento 4,0 s) | 8,2 s con 2 hilos | ~20–30 s (presupuesto ≤ 90 s ✅) |
| Cuaderno 1 completo por defecto | 26 s | 30 s | ~1–1,5 min |
| Cuaderno 1 · Plan B (7 entrenamientos) | 54 s | 121 s con 2 hilos | ~5–7 min |
| Cuaderno 1 · primera instalación en Colab | — | — | [🔴 no medido] |
| Cuaderno 2 · 2.3 + 2.5 + 2.6, demo 1 | 18,6 s | 24,7 s con 1 hilo | ~60–90 s (presupuesto ≤ 2 min ✅) |
| Cuaderno 2 · 2.3 + 2.5 + 2.6, demo 2 | 25,2 s | 31,3 s con 1 hilo | ~80–110 s (al límite) |
| Cuaderno 2 completo, demo 1 | 26 s | 31 s | ~1,5–2 min + primera instalación [🔴 no medida] |

Colab gratuito tiene 2 vCPU (🟡, fuentes comunitarias) y Ultralytics usa 1 hilo ahí. El factor 2,5–3,5 compara un núcleo de M3
Pro con un Xeon de 2,2 GHz (🟡, no medido en Colab).

---

## Test de calidad (gate del CLAUDE.md, §9)

| Criterio | 60 min | 40 min (S7) |
|---|---|---|
| Techo de teoría (≤ 15) | ✅ 14,5 | ✅ 10 (la S7 entera ya lo rompe por la actualidad, aprobado). ⚠️ 19,5 si la demo del bloque 9 cuenta entera como escucha |
| Herramienta usada en pantalla | ✅ CNN Explainer, Colab, hoja común; n8n solo si se abre | ✅ CNN Explainer, Colab, hoja común |
| Entregable tangible | ✅ cuaderno 1 con su reto y su Grad-CAM | ✅ igual; la copia en Drive se pide a las 20:57 |
| Ninguna herramienta nombrada sin usar | ⚠️ n8n: si no se abre en 4d, no se nombra | ✅ n8n no se nombra; solo aparece en el texto del cuaderno (2.8) |
| Sin solapes | 🟡 S4 v2 tuvo 2 min de intuición CNN y atajos (si se dio) | 🟡 igual |
| Bloques suman | ✅ 60 | ✅ 40 (28 + 12) |
| Checkpoints cada 15-20 min como máximo | ✅ 10, 16, 24, 38 y 51 | ✅ 20:37, 20:45, 20:52, 20:59 y 21:03 |
| Variante simplificada y colchón | ✅ | ✅ |
| Datos no verificables marcados | ✅ 🟡 y 🔴 | ✅ |

---

## Fuentes utilizadas

- `BRIEF.md` §2 (agenda de 60 min), §7 (contenido del guion) y §9 (criterios).
- `verificacion/final/hechos_para_docs.md`: tiempos, resultados de referencia de los dos cuadernos, contexto de la S7, recorrido de
  CNN Explainer y pendientes en Colab. Detalle en `verificacion/final/resumen_medidas.txt`.
- Vistas de alumno de las ejecuciones finales: `verificacion/final/01_laboratorio_cnn_radiografias_vista_alumno.txt`,
  `01_plan_b_vista_alumno.txt`, `01_imposible_vista_alumno.txt`, `02_conteo_personas_cafeteria_vista_alumno.txt`,
  `02_demo2_vista_alumno.txt`, y las figuras `png/01_laboratorio_cnn_radiografias/c08_0.png` (curvas) y `c14_1.png` (Grad-CAM)
  y `png/02_conteo_personas_cafeteria/c04_1.png` (primer fotograma sin cajas).
  Ojo: `verificacion/` está fuera de git (`.gitignore`), y `scripts/verificar.py` no genera `hechos_para_docs.md`,
  `resumen_medidas.txt` ni las vistas del alumno: se compilaron a mano el 24-sep.
- `src/build_nb1.py` (aviso de 1.3 cuando el último mapa mide 3×3 o menos; descarga de `experimentos.csv` en 1.8) y
  `src/build_nb2.py` (aviso de 2.3 cuando la estimación pasa de 100 s).
- `docs/notas_verificacion/e_colab.md` (reinicio tras instalar, «Change runtime version») y `README.md` §3.
- `DECISIONS.md`: D10, D14, D16, D17, D18, D19, D20, D24, D25, D32, D38, D40, D41, D50, D51 y D52.
- `docs/notas_verificacion/f_fuentes_y_afirmaciones.md` (Zech et al. 2018, CNN Explainer, Grad-CAM, MedMNIST, YOLO26, AGPL, RGPD y
  EDPB) y `h_ensayo_cnn_explainer.md` (recorrido de clics ensayado en navegador).
- `sesiones/S7/01-esqueleto.md` (bloques 8 y 9 y colchón) y `sesiones/S7/05-checklist.md`.
- `docs/hoja_alumno.md` (para que los pasos coincidan) y `sesiones/S4/03-v2-lente-social.md` (posible solape).
- `Presentaciones/IAaSP/CLAUDE.md`: clase online, sin parejas, cierre con POC ≠ producción.

## Pendientes de validar

- ✅ Enlace de los cuadernos: https://github.com/Lougedo/cnn_samples_colab (AGPL-3.0, D50).
- 🔴 Hoja común: sin crear. Probar que la línea pegada desde Colab se reparte en columnas 🟡.
- 🔴 Lista de matriculados para el reparto A-B-C-D: no consta en el repositorio.
- 🔴 Tiempo de la primera instalación en Colab (1.1 y 2.1).
- 🟡 Que la instalación no pida reiniciar la sesión con Python 3.13 (la resolución de paquetes de `e_colab.md` dice que no).
- 🟡 Todos los tiempos de Colab son proyecciones; los de cada reto (§2.2) los he calculado yo con el factor de `hechos_para_docs.md`.
- 🟡 Que 1.3 avise de tiempo en el reto C (depende de lo que mida en Colab).
- 🟡 Tamaños del mapa de Grad-CAM de A con 1 bloque (26×26) y de B y C (24×24): calculados, no vistos. El de 3×3 lo avisa 1.3.
- 🟡 Origen de un solo centro de las radiografías (Kermany et al.): solo fuentes secundarias; el artículo de Cell dio 403
  (`f_fuentes_y_afirmaciones.md` §4).
- 🟡 Lectura de los mapas de Grad-CAM de referencia (4 de 6 en el centro del tórax): una ejecución local; en Colab pueden cambiar.
- 🟡 Rótulos de Colab en español («Ejecutar de todos modos», «Guardar una copia en Drive», reinicio de sesión) y de Google Sheets
  («Dividir texto en columnas», «Historial de versiones»).
- 🟡 Que una fila copiada del CSV del Plan B (abierto en Excel) se reparta en columnas al pegarla en la hoja común.
- 🟡 Reproducción del vídeo H.264 en Chrome y Safari, y `private_outputs` del cuaderno 2 (D38).
- 🟡 Que Colab ejecute en cola las celdas del cuaderno 2 pulsadas seguidas (8d) y que no corte una sesión inactiva antes de las 20:40.
- 🟡 Dos entornos de Colab a la vez en la cuenta del profesor (cuaderno 1 y cuaderno 2).
- 🟡 Estado de `polyfill.io` el día de clase (hoy responde 403).
- 🟡 Si la slide 27 de la S4 v2 (CNN y atajos) se llegó a dar.
- 🟡 n8n: rótulos de los botones del Webhook según la versión (D41), solo para la versión de 60.
- 🔴 Decidir si `verificacion/final/` (hechos, medidas y vistas del alumno) se versiona en git: hoy `.gitignore` lo excluye y,
  si se publica el repositorio, las cifras de referencia se quedan sin fuente rastreable.
- 🔴 D23: rótulos con n.º en el gráfico del registro, desviación del brief pendiente de aceptar.
