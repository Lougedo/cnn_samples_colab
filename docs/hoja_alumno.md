# Hoja del alumno · Una CNN por dentro con radiografías de tórax

Trabajas por tu cuenta; dudas, al chat o al micrófono. Te llevas tu cuaderno ejecutado, con tu red y su Grad-CAM.

> ⚠️ **No apto para uso clínico.** Son radiografías de niños reducidas a 28×28 píxeles, con fines educativos: nada de esto
> sirve para diagnosticar. Es una prueba de concepto (POC), no un sistema en producción.

## Pasos

1. Abre el enlace del chat, mejor en Chrome. Si Colab dice que el cuaderno no es de Google: **Ejecutar de todos modos**.
2. Ejecuta **1.1** (la primera vez tarda algo más) y **1.2** (deja 28 px).
3. Después de 1.2, baja directamente a **1.9** (sáltate 1.3-1.8 por ahora): elige tu reto, **escribe tu hipótesis antes de entrenar** y ejecuta.
4. En **1.3** cambia lo que pide tu reto y pon `nombre_experimento` = `iniciales_reto` (p. ej. `ALM_A`).
5. Ejecuta **1.3 → 1.4 → 1.5 → 1.8**. En 1.8 salen dos líneas: pega **solo la segunda** en **tu fila** de la hoja común (te la da el chat). 1.8 también descarga un CSV con tus experimentos: si el navegador pregunta, permite la descarga.
6. A y C: repite 4 y 5 con la otra variante. Luego, **1.6** (qué ve cada capa) y **1.7** (Grad-CAM). No ejecutes 1.10: es del profesor. Para guardar tu cuaderno: **Archivo → Guardar una copia en Drive**.

## Los cuatro retos (parte de la red por defecto: sus valores están encima de 1.3)

| Reto | Qué cambias en 1.3 | En qué te fijas |
|---|---|---|
| **A · Profundidad** | `bloques_convolucionales` en 1 y luego en 3 | Parámetros, mapas de 1.6 (míralos después de cada entrenamiento), sensibilidad, especificidad |
| **B · Sin pooling** | Desmarca `usar_pooling` | Parámetros, tiempo y métricas* |
| **C · Dropout 0 frente a 0,5** | Desmarca `usar_pooling`, `epocas` en 20; `dropout` en 0 y luego en 0,5 (sale 0.5) | ¿Se separan las curvas de 1.4? ¿Y el diagnóstico? |
| **D · Compensar el desbalanceo** | Marca `compensar_desbalanceo` | Sensibilidad y especificidad* |

**Palabras.** *Parámetros*: números que la red ajusta. *Pooling*: deja la imagen a la mitad de lado (una cuarta parte de los píxeles).
*Sensibilidad*: neumonías que detecta. *Especificidad*: normales que deja tranquilas. *Diagnóstico* (1.4): razonable, sobreajuste (memoriza, falla con nuevas), infraajuste o colapso.

## Mi hipótesis y mis resultados (salen en la tabla de 1.8)

**Si cambio** ……………………………… **creo que** ……………………………… **porque** ………………………………

| Experimento | Parámetros | Acierto test | Sensibilidad | Especificidad | Diagnóstico | ¿Se cumplió mi hipótesis? |
|---|---|---|---|---|---|---|
| Por defecto* | | | | | | — |
| Mío 1 | | | | | | |
| Mío 2 (A y C) | | | | | | |

\* B y D se comparan con la red por defecto: cópiala de la hoja común o entrénala tú (1.3 sin tocar nada).

## Para pensar

1. En la hoja común, ¿qué reto movió más la sensibilidad y la especificidad, y lo bastante para descartar el azar? ¿Más parámetros dieron mejor test?
2. En tu Grad-CAM (1.7), ¿la red mira los pulmones o otras zonas? ¿Te fiarías de ella en otro hospital?
3. Para cribar, ¿prefieres más falsas alarmas o más neumonías que se escapan? ¿Quién debería decidirlo?

<small>**Fuentes.** Datos: PneumoniaMNIST (MedMNIST v2; Yang et al., *Scientific Data*, 2023), de Kermany et al. (*Cell*, 2018). CC BY 4.0. Pasos y cifras: el cuaderno 1.</small>
