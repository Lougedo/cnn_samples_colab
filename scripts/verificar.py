"""Ejecuta los notebooks sin interfaz (headless) y resume los resultados.

Simula lo que haría un alumno con los formularios: sustituye el valor de las
líneas `nombre = valor  # @param ...` antes de ejecutar. No toca los .ipynb
del repositorio; guarda las copias ejecutadas en `verificacion/<modo>/`.

Uso (desde la raíz del repo, con el entorno de verificación):
    .venv/bin/python scripts/verificar.py                 # modos rapido + defecto, ambos notebooks
    .venv/bin/python scripts/verificar.py --modos defecto --nb 1
    .venv/bin/python scripts/verificar.py --modos referencias --hilos 2
    .venv/bin/python scripts/verificar.py --modos rapido --set epocas=3 --set usar_pooling=False

Modos:
    rapido       NB1 con 2 épocas; NB2 con 3 s de vídeo.
    defecto      Valores por defecto de los formularios.
    referencias  NB1 con el Plan B activado (entrena las configuraciones de los retos).
    imposible    NB1 con 4 bloques a 28 px: debe salir el aviso, sin excepciones.

--hilos N limita TensorFlow y PyTorch a N hilos, como aproximación a los
2 vCPU de Colab gratuito. Es una aproximación: la CPU de Colab es más lenta
por núcleo que un portátil moderno.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parent.parent
NOTEBOOKS = {
    "1": RAIZ / "notebooks" / "01_laboratorio_cnn_radiografias.ipynb",
    "2": RAIZ / "notebooks" / "02_conteo_personas_cafeteria.ipynb",
}
SALIDAS = {"1": RAIZ / "salidas" / "nb1", "2": RAIZ / "salidas" / "nb2"}

MODOS = {
    "rapido": {"1": {"epocas": 2}, "2": {"segundos_a_procesar": 3}},
    "defecto": {"1": {}, "2": {}},
    "referencias": {"1": {"ejecutar_plan_b": True}},
    "imposible": {"1": {"bloques_convolucionales": 4, "resolucion": 28}},
}
# Texto que debe aparecer en la salida del modo "imposible".
PATRON_IMPOSIBLE = re.compile(r"0\s*[×x]\s*0|no cabe|se queda", re.IGNORECASE)

LINEA_PARAM = r"^(?P<ind>\s*){nombre}(?P<eq>\s*=\s*)(?P<valor>.*?)(?P<resto>\s*#\s*@param.*)$"


def fijar_parametros(nb, valores: dict) -> None:
    """Cambia el valor de cada `nombre = ...  # @param` como si se tocara el formulario."""
    pendientes = dict(valores)
    for celda in nb.cells:
        if celda.cell_type != "code":
            continue
        for nombre in list(pendientes):
            patron = re.compile(LINEA_PARAM.format(nombre=re.escape(nombre)), re.MULTILINE)
            nuevo, n = patron.subn(
                lambda m: f"{m['ind']}{nombre}{m['eq']}{pendientes[nombre]!r}{m['resto']}", celda.source
            )
            if n:
                celda.source = nuevo
                del pendientes[nombre]
    if pendientes:
        raise SystemExit(f"No encuentro estos parámetros de formulario: {sorted(pendientes)}")


def comprobar_estructura(ruta: Path) -> list[str]:
    """Errores de forma del notebook tal como está en el repo (sin ejecutar)."""
    nb = nbformat.read(ruta, as_version=4)
    problemas = []
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        primera = c.source.lstrip().splitlines()[0] if c.source.strip() else ""
        if c.metadata.get("cellView") != "form":
            problemas.append(f"celda {i}: falta metadata cellView='form'")
        if not re.match(r"#\s*@title", primera):
            problemas.append(f"celda {i}: la primera línea no es '# @title ...'")
        if c.get("outputs") or c.get("execution_count"):
            problemas.append(f"celda {i}: el notebook del repo tiene salidas guardadas")
    return problemas


def duracion_celda(celda) -> float | None:
    ex = celda.metadata.get("execution", {})
    ini, fin = ex.get("iopub.execute_input"), ex.get("shell.execute_reply")
    if not (ini and fin):
        return None
    f = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
    return (f(fin) - f(ini)).total_seconds()


def texto_salidas(celda) -> str:
    partes = []
    for o in celda.get("outputs", []):
        if o.output_type == "stream":
            partes.append(o.text)
        elif o.output_type in ("display_data", "execute_result"):
            partes.append(o.get("data", {}).get("text/plain", ""))
            partes.append(o.get("data", {}).get("text/html", ""))
        elif o.output_type == "error":
            partes.append(f"{o.ename}: {o.evalue}")
    return "\n".join(partes)


def ejecutar(nb_id: str, modo: str, extra: dict, hilos: int | None, timeout: int) -> dict:
    ruta = NOTEBOOKS[nb_id]
    nb = nbformat.read(ruta, as_version=4)
    valores = {**MODOS[modo].get(nb_id, {}), **extra}
    fijar_parametros(nb, valores)

    destino = RAIZ / "verificacion" / modo
    destino.mkdir(parents=True, exist_ok=True)
    if SALIDAS[nb_id].exists():
        shutil.rmtree(SALIDAS[nb_id])

    entorno_previo = dict(os.environ)
    if hilos:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "TF_NUM_INTRAOP_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[var] = str(hilos)
        os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["MPLBACKEND"] = "Agg"

    cliente = NotebookClient(nb, timeout=timeout, kernel_name="python3", allow_errors=True,
                             record_timing=True, resources={"metadata": {"path": str(RAIZ)}})
    t0 = time.perf_counter()
    try:
        cliente.execute()
    finally:
        os.environ.clear()
        os.environ.update(entorno_previo)
    total = time.perf_counter() - t0
    nbformat.write(nb, destino / ruta.name)

    celdas, errores, avisos = [], [], []
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        titulo = re.sub(r"^#\s*@title\s*", "", c.source.lstrip().splitlines()[0])
        titulo = re.sub(r"\{.*\}\s*$", "", titulo).strip()
        celdas.append({"celda": i, "titulo": titulo, "segundos": duracion_celda(c)})
        for o in c.get("outputs", []):
            if o.output_type == "error":
                errores.append(f"[{i}] {titulo}: {o.ename}: {o.evalue}")
            elif o.output_type == "stream" and o.name == "stderr" and o.text.strip():
                avisos.append(f"[{i}] {titulo}: {o.text.strip()[:300]}")

    resultado = {"notebook": ruta.name, "modo": modo, "parametros": valores, "hilos": hilos,
                 "segundos_total": round(total, 1), "celdas": celdas,
                 "errores": errores, "stderr": avisos}

    if modo == "imposible":
        salida = "\n".join(texto_salidas(c) for c in nb.cells if c.cell_type == "code")
        resultado["aviso_imposible_detectado"] = bool(PATRON_IMPOSIBLE.search(salida))

    # Copia los ficheros exportados por el notebook junto al notebook ejecutado.
    if SALIDAS[nb_id].exists():
        copia = destino / f"salidas_nb{nb_id}"
        shutil.rmtree(copia, ignore_errors=True)
        shutil.copytree(SALIDAS[nb_id], copia)
        resultado["ficheros"] = sorted(p.name for p in copia.iterdir())
        resultado.update(leer_exportaciones(nb_id, copia))
    return resultado


def leer_exportaciones(nb_id: str, carpeta: Path) -> dict:
    datos = {}
    if nb_id == "1":
        csv = next(carpeta.glob("experimentos*.csv"), None)
        if csv:
            import pandas as pd
            datos["experimentos"] = pd.read_csv(csv, sep=";", decimal=",").to_dict("records")
    else:
        js = carpeta / "resumen.json"
        if js.exists():
            datos["resumen_video"] = json.loads(js.read_text(encoding="utf-8"))
    return datos


def informe_md(resultados: list[dict]) -> str:
    lineas = [f"# Verificación headless — {datetime.now():%Y-%m-%d %H:%M}", ""]
    for r in resultados:
        estado = "✅" if not r["errores"] else "❌"
        lineas += [f"## {estado} {r['notebook']} · modo `{r['modo']}` · hilos {r['hilos'] or 'todos'}", "",
                   f"- Parámetros: `{r['parametros']}`",
                   f"- Tiempo total (incluye arranque del kernel): **{r['segundos_total']} s**"]
        if "aviso_imposible_detectado" in r:
            lineas.append(f"- Aviso de configuración imposible detectado: **{r['aviso_imposible_detectado']}**")
        lineas += ["", "| Celda | Título | s |", "|---|---|---|"]
        lineas += [f"| {c['celda']} | {c['titulo']} | {c['segundos'] if c['segundos'] is None else round(c['segundos'], 1)} |"
                   for c in r["celdas"]]
        if r["errores"]:
            lineas += ["", "**Errores**", *[f"- {e}" for e in r["errores"]]]
        if r["stderr"]:
            lineas += ["", "**Salida por stderr (revisar si es relevante)**", *[f"- {a}" for a in r["stderr"]]]
        if r.get("experimentos"):
            cols = list(r["experimentos"][0])
            lineas += ["", "**Experimentos registrados**", "", "| " + " | ".join(cols) + " |",
                       "|" + "---|" * len(cols),
                       *["| " + " | ".join(str(e.get(k, "")) for k in cols) + " |" for e in r["experimentos"]]]
        if r.get("resumen_video"):
            lineas += ["", "**Resumen del vídeo**", "", "```json",
                       json.dumps(r["resumen_video"], ensure_ascii=False, indent=2), "```"]
        lineas.append("")
    return "\n".join(lineas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modos", nargs="+", default=["rapido", "defecto"], choices=list(MODOS))
    ap.add_argument("--nb", choices=["1", "2", "todos"], default="todos")
    ap.add_argument("--hilos", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=1800, help="segundos máximos por celda")
    ap.add_argument("--set", action="append", default=[], metavar="NOMBRE=VALOR",
                    help="fija un parámetro de formulario (valor en sintaxis Python)")
    ap.add_argument("--construir", action="store_true", help="regenera los notebooks desde src/ antes")
    args = ap.parse_args()

    if args.construir:
        import subprocess
        for script in ("build_nb1.py", "build_nb2.py"):
            subprocess.run([sys.executable, str(RAIZ / "src" / script)], check=True)

    extra = {}
    for s in args.set:
        nombre, _, valor = s.partition("=")
        try:
            extra[nombre.strip()] = ast.literal_eval(valor)
        except (ValueError, SyntaxError):
            extra[nombre.strip()] = valor

    ids = ["1", "2"] if args.nb == "todos" else [args.nb]
    problemas = {NOTEBOOKS[i].name: comprobar_estructura(NOTEBOOKS[i]) for i in ids}
    resultados = []
    for modo in args.modos:
        for i in ids:
            if i not in MODOS[modo]:
                continue
            print(f"▶ {NOTEBOOKS[i].name} · {modo} …", flush=True)
            r = ejecutar(i, modo, extra, args.hilos, args.timeout)
            print(f"  {'OK' if not r['errores'] else 'ERRORES'} en {r['segundos_total']} s", flush=True)
            resultados.append(r)

    marca = datetime.now().strftime("%Y%m%d-%H%M%S")
    carpeta = RAIZ / "verificacion"
    carpeta.mkdir(exist_ok=True)
    md = informe_md(resultados)
    if any(problemas.values()):
        md = "## ❌ Problemas de estructura\n\n" + "\n".join(
            f"- {nb}: {p}" for nb, ps in problemas.items() for p in ps) + "\n\n" + md
    (carpeta / f"informe_{marca}.md").write_text(md, encoding="utf-8")
    (carpeta / f"resultados_{marca}.json").write_text(
        json.dumps({"estructura": problemas, "ejecuciones": resultados}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print(md)
    fallos = any(r["errores"] for r in resultados) or any(problemas.values()) or any(
        r.get("aviso_imposible_detectado") is False for r in resultados)
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
