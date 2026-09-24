# Ensayo de CNN Explainer en un navegador real (24-sep-2026)

Complementa la nota f §2, que verificó el recorrido en el código desplegado pero no en un navegador.
Ensayado por el coordinador en el navegador integrado de la app (Chromium), ventana de 1440×900.

## Qué se comprobó

| Paso | Clic | Qué aparece | Resultado |
|---|---|---|---|
| 0 | Cargar <https://poloclub.github.io/cnn-explainer/> | Red completa: `input`, `conv_1_1`, `relu_1_1`, `conv_1_2`, `relu_1_2`, `max_pool_1`, `conv_2_1`, `relu_2_1`, `conv_2_2`, `relu_2_2`, `max_pool_2`, `output`. Imagen por defecto: *espresso* | ✅ |
| 1 | Una neurona de `conv_1_1` | **No abre la fórmula todavía.** Abre una vista intermedia: los 3 canales de entrada (rojo, verde, azul), un resultado intermedio por canal y la suma con el sesgo | ✅ Corrige la nota f: son **dos clics** |
| 2 | Uno de los cuadros *intermediate* (o el núcleo) | Ventana **«Convolution»**: *Input (64, 64)* → *Output (62, 62)*, el núcleo 3×3 con sus pesos y *Hover over the matrices to change kernel position* | ✅ |
| 3 | La × de la ventana y luego un espacio vacío | Vuelve a la red completa | ✅ |
| 4 | Una neurona de `relu_1_1` | Ventana **«ReLU Activation»**: *Input (62, 62)* → *Output (62, 62)*, `max(0, x)` | ✅ Un clic. **Otro clic en la misma neurona la cierra** |
| 5 | Una neurona de `max_pool_1` | Ventana **«Max Pooling»**: *Input (60, 60)* → *Output (30, 30)*, máximo de cada 2×2 | ✅ |
| 6 | La clase *espresso* en `output` | Vista de `max_pool_2` → `flatten` → softmax, con *Output value: 0.8606*. El cuadro *softmax* dice *Click to learn more* | ✅ |

## Avisos para el profesor

- Abajo a la izquierda sale un recuadro *You might also like* (otras herramientas del mismo laboratorio). Ciérralo con la × al empezar para que no tape la red.
- Con la ventana estrecha (800 px) la red no cabe y se ve rota: usa el navegador a pantalla completa.
- 🟡 La web carga `https://polyfill.io/v3/polyfill.min.js`. Hoy ese dominio responde **403 de Cloudflare** («Attention Required»), así que el script no se ejecuta y la página funciona sin él (comprobado con `curl` y en el navegador). El dominio sufrió un ataque de cadena de suministro en 2024: se mantiene la recomendación de que solo la abra el profesor, compartiendo pantalla.
- El valor de *espresso* (0,86) es el que salió en este ensayo; no lo cites como dato fijo.
