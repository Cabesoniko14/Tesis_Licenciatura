# ======================  Frozen Lake: estudio del formato del prompt (réplica de la recompensa)  ======================
# Mide qué tan bien replica el LLM la recompensa de Gym (1 si llega a la meta, 0 si no) según:
#   - formato del prompt: con líneas numeradas (00:, 01:, ...) o sin numeración
#   - formato de respuesta: número (0/1) o categoría (SI/NO)
# más un control con el dato "Llegó a la meta" explícito. Todas las variantes ven los mismos episodios.
# Uso:
#   python estudio_prompts.py --config ../configs/A2.json [--n 30] [--salida DIR]
import os
import re
import json
import time
import math
import random
import argparse
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import make_env, semillas_de_mapas, idx_to_rc, tile_at, desc_a_filas, ACCION_NOMBRE
from prompts import construir_prompt, DIRECTIVAS
from llm_reward import crear_cliente

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
SEMILLA_EPISODIOS = 2026
CLASES = ["META", "HOYO", "TRUNCADO"]
MAPAS = {
    "fijo": {"tamano": 4, "modo": "fijo", "nombre_gym": "4x4", "resbaloso": False, "max_pasos": 50},
    "aleatorio": {"tamano": 4, "modo": "aleatorio", "resbaloso": False, "max_pasos": 50},
}

# --------------------  1. Prompt original (con numeración) --------------------
# Primera versión de prompts.py, con la que se corrió A2 el 19 nov y el LLM castigó todas las victorias.
_PROMPT_NUMERADO = """Eres una función de recompensas para un episodio de FrozenLake.

REGLAS DEL ENTORNO:
- Mapa de {H}x{W}. Celdas: 'S' inicio, 'F' hielo seguro, 'H' hoyo, 'G' meta.
- Acciones: 0=Left, 1=Down, 2=Right, 3=Up.
- El episodio termina si el agente llega a 'G' o cae en un 'H'. Si se acaban los pasos, se trunca.

MAPA (por filas):
{mapa}

ÍNDICES DE CADA CELDA (por filas):
{indices}

CAMINO DEL AGENTE:
{camino}

FIN DEL EPISODIO:
- terminated={terminated}, truncated={truncated}
- casilla final: '{casilla_final}'

DIRECTIVA:
{directiva}

Responde solo con un número entre 0 y 1, sin texto extra.
"""

def prompt_numerado(desc, estados, acciones, terminated, truncated):
    H, W = desc.shape
    mapa = "\n".join(f"{r:02d}: {fila}" for r, fila in enumerate(desc_a_filas(desc)))
    indices = "\n".join(" ".join(f"{r * W + c:02d}" for c in range(W)) for r in range(H))
    r0, c0 = idx_to_rc(estados[0], W)
    camino = [f"00: inicio -> idx={estados[0]:02d} (r={r0},c={c0}) '{tile_at(desc, estados[0])}'"]
    for i, (a, s) in enumerate(zip(acciones, estados[1:]), start=1):
        r, c = idx_to_rc(s, W)
        camino.append(f"{i:02d}: {ACCION_NOMBRE[int(a)]} -> idx={s:02d} (r={r},c={c}) '{tile_at(desc, s)}'")
    return _PROMPT_NUMERADO.format(H=H, W=W, mapa=mapa, indices=indices, camino="\n".join(camino),
                                   terminated=bool(terminated), truncated=bool(truncated),
                                   casilla_final=tile_at(desc, estados[-1]), directiva=DIRECTIVAS["replica"])

# --------------------  2. Variantes --------------------
FINAL_NUM = "Responde solo con un número entre 0 y 1, sin texto extra."
DIRECTIVA_CAT = ("Responde SI si el agente llegó a la meta 'G'. "
                 "Responde NO en cualquier otro caso (cayó en un hoyo o se acabaron los pasos).")
FINAL_CAT = "Responde solo con una palabra: SI o NO."

def a_categoria(p):
    return p.replace(DIRECTIVAS["replica"], DIRECTIVA_CAT).replace(FINAL_NUM, FINAL_CAT)

def con_dato_meta(p, gano):
    return p.replace("- Casilla final:", f"- Llegó a la meta: {'sí' if gano else 'no'}\n- Casilla final:")

def leer_numero(t):
    m = re.search(r"[-+]?\d*\.?\d+", t)
    return max(0.0, min(1.0, float(m.group(0)))) if m else None

def leer_categoria(t):
    t = re.sub(r"[^A-Z]", "", t.upper().replace("Í", "I"))
    return 1.0 if t.startswith("SI") else (0.0 if t.startswith("NO") else None)

# ep = (desc, estados, acciones, terminated, truncated, clase)
VARIANTES = {
    "V1 numerado + número":          (lambda ep: prompt_numerado(*ep[:5]), leer_numero),
    "V2 sin numeración + número":     (lambda ep: construir_prompt(*ep[:5], "replica"), leer_numero),
    "V3 numerado + categoría":        (lambda ep: a_categoria(prompt_numerado(*ep[:5])), leer_categoria),
    "V4 sin numeración + categoría":  (lambda ep: a_categoria(construir_prompt(*ep[:5], "replica")), leer_categoria),
    "V5 sin numeración + número + dato de meta":
                                      (lambda ep: con_dato_meta(construir_prompt(*ep[:5], "replica"), ep[5] == "META"), leer_numero),
}

# --------------------  3. Episodios de prueba --------------------
def destinos(desc, s):
    H, W = desc.shape
    r, c = idx_to_rc(s, W)
    return {0: (r, max(c - 1, 0)), 1: (min(r + 1, H - 1), c), 2: (r, min(c + 1, W - 1)), 3: (max(r - 1, 0), c)}

def acciones_permitidas(desc, s, clase):
    """META: no pisar hoyos. HOYO: cualquiera. TRUNCADO: no pisar hoyos ni la meta."""
    if clase == "HOYO":
        return [0, 1, 2, 3]
    prohibidas = (b'H',) if clase == "META" else (b'H', b'G')
    return [a for a, (r, c) in destinos(desc, s).items() if desc[r, c] not in prohibidas] or [0, 1, 2, 3]

def jugar(env, desc, clase, rng):
    s, _ = env.reset()
    estados, acciones = [int(s)], []
    term = trunc = False
    r = 0.0
    while not (term or trunc):
        a = rng.choice(acciones_permitidas(desc, estados[-1], clase))
        s, r, term, trunc, _ = env.step(a)
        estados.append(int(s)); acciones.append(int(a))
    clase_real = "META" if (term and r == 1.0) else ("HOYO" if term else "TRUNCADO")
    return (desc, estados, acciones, term, trunc, clase_real)

def generar_episodios(cfg_mapa, n, semilla):
    """n episodios distintos por clase. En mapas aleatorios cada episodio usa un mapa distinto."""
    rng = random.Random(semilla)
    semillas = iter(semillas_de_mapas(semilla, 1_000_000))
    episodios, vistos = {c: [] for c in CLASES}, set()
    for clase in CLASES:
        intentos = 0
        while len(episodios[clase]) < n and intentos < 200_000:
            intentos += 1
            env, desc = make_env(cfg_mapa, next(semillas) if cfg_mapa["modo"] == "aleatorio" else None)
            ep = jugar(env, desc, clase, rng)
            clave = ("|".join(desc_a_filas(desc)), tuple(ep[2]))
            if ep[5] == clase and clave not in vistos:
                vistos.add(clave)
                episodios[clase].append(ep)
        if len(episodios[clase]) < n:
            print(f"[WARN] solo se generaron {len(episodios[clase])} episodios distintos de {clase} en mapa {cfg_mapa['modo']}")
    return episodios

# --------------------  4. Intervalo de confianza --------------------
def wilson(aciertos, n, z=1.96):
    """Intervalo de confianza al 95% para una proporción (Wilson), en %."""
    if n == 0:
        return (None, None)
    p = aciertos / n
    centro = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    margen = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return (round(max(0.0, 100 * (centro - margen)), 1) + 0.0, round(min(100.0, 100 * (centro + margen)), 1))

def resumir(df, por):
    filas = []
    for llave, g in df.groupby(por):
        llave = llave if isinstance(llave, tuple) else (llave,)
        aciertos, n = int(g["correcto"].sum()), len(g)
        bajo, alto = wilson(aciertos, n)
        filas.append({**dict(zip(por, llave)), "n": n, "aciertos": aciertos,
                      "aciertos_%": round(100 * aciertos / n, 1), "ic95_bajo_%": bajo, "ic95_alto_%": alto,
                      "sin_respuesta_valida": int(g["valor"].isna().sum()),
                      "tokens_salida_prom": round(g["tokens_salida"].mean(), 2),
                      "latencia_prom_s": round(g["latencia_s"].mean(), 3)})
    return pd.DataFrame(filas)

# --------------------  5. Estudio --------------------
def correr_estudio(cfg, n, run_dir):
    cliente = crear_cliente(cfg["llm"])
    filas, ejemplos = [], {}

    for tipo_mapa, cfg_mapa in MAPAS.items():
        episodios = generar_episodios(cfg_mapa, n, SEMILLA_EPISODIOS)
        for clase in CLASES:
            esperado = 1.0 if clase == "META" else 0.0
            for i, ep in enumerate(episodios[clase]):
                for variante, (armar, leer) in VARIANTES.items():
                    prompt = armar(ep)
                    t0 = time.perf_counter()
                    resp = cliente.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}], model=cfg["llm"]["modelo"],
                        temperature=cfg["llm"]["temperatura"], max_tokens=cfg["llm"]["max_tokens"])
                    latencia = time.perf_counter() - t0
                    texto = (resp.choices[0].message.content or "").strip()
                    valor = leer(texto)
                    filas.append({"mapa": tipo_mapa, "clase": clase, "episodio": i, "pasos": len(ep[2]),
                                  "mapa_desc": "|".join(desc_a_filas(ep[0])),
                                  "acciones": "".join(ACCION_NOMBRE[a][0] for a in ep[2]),
                                  "variante": variante, "respuesta": texto, "valor": valor,
                                  "esperado": esperado, "correcto": valor == esperado,
                                  "tokens_entrada": resp.usage.prompt_tokens if resp.usage else None,
                                  "tokens_salida": resp.usage.completion_tokens if resp.usage else None,
                                  "latencia_s": latencia})
                    if tipo_mapa == "fijo" and clase == "META" and variante not in ejemplos:
                        ejemplos[variante] = (prompt, texto)
        pd.DataFrame(filas).to_csv(os.path.join(run_dir, "resultados.csv"), index=False)
        print(f"=== mapa {tipo_mapa} terminado | {len(filas)} llamadas acumuladas ===")

    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(run_dir, "resultados.csv"), index=False)
    total = resumir(df, ["variante"])
    por_clase = resumir(df, ["variante", "clase"])
    por_mapa_clase = resumir(df, ["variante", "mapa", "clase"])
    total.to_csv(os.path.join(run_dir, "resumen_por_variante.csv"), index=False)
    por_clase.to_csv(os.path.join(run_dir, "resumen_por_variante_clase.csv"), index=False)
    por_mapa_clase.to_csv(os.path.join(run_dir, "resumen_por_variante_mapa_clase.csv"), index=False)

    with open(os.path.join(run_dir, "prompts_ejemplo.txt"), "w", encoding="utf-8") as f:
        for variante, (prompt, texto) in ejemplos.items():
            f.write(f"########## {variante} (mapa fijo, episodio que llega a la meta)\n{prompt}\n>>> Respuesta del LLM: {texto!r}\n\n")

    guardar_figuras(por_clase, total, run_dir)
    print("\nAciertos por variante:")
    print(total[["variante", "n", "aciertos_%", "ic95_bajo_%", "ic95_alto_%", "sin_respuesta_valida"]].to_string(index=False))
    print("\nAciertos por variante y clase:")
    print(por_clase.pivot(index="variante", columns="clase", values="aciertos_%")[CLASES].to_string())

# --------------------  6. Figuras --------------------
def guardar_figuras(por_clase, total, run_dir):
    figs = os.path.join(run_dir, "figs")
    os.makedirs(figs, exist_ok=True)
    variantes = list(VARIANTES)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    alto = 0.26
    for j, clase in enumerate(CLASES):
        d = por_clase[por_clase["clase"] == clase].set_index("variante").loc[variantes]
        y = [i + (j - 1) * alto for i in range(len(variantes))]
        err = [d["aciertos_%"] - d["ic95_bajo_%"], d["ic95_alto_%"] - d["aciertos_%"]]
        ax.barh(y, d["aciertos_%"], height=alto, xerr=err, capsize=2, label=clase)
    ax.set_yticks(range(len(variantes))); ax.set_yticklabels(variantes)
    ax.invert_yaxis(); ax.set_xlim(0, 105); ax.grid(axis="x", alpha=0.3)
    ax.set_xlabel("Aciertos (%) con IC 95%")
    ax.set_title("Réplica de la recompensa según el formato del prompt, por resultado del episodio")
    ax.legend(title="Resultado", loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.tight_layout(); plt.savefig(os.path.join(figs, "aciertos_por_variante_y_clase.png"), dpi=150); plt.close()

    d = total.set_index("variante").loc[variantes]
    fig, ax = plt.subplots(figsize=(10, 4))
    err = [d["aciertos_%"] - d["ic95_bajo_%"], d["ic95_alto_%"] - d["aciertos_%"]]
    ax.barh(range(len(variantes)), d["aciertos_%"], xerr=err, capsize=3)
    ax.set_yticks(range(len(variantes))); ax.set_yticklabels(variantes)
    ax.invert_yaxis(); ax.set_xlim(0, 105); ax.grid(axis="x", alpha=0.3)
    ax.set_xlabel("Aciertos (%) con IC 95%")
    ax.set_title("Réplica de la recompensa según el formato del prompt (total)")
    plt.tight_layout(); plt.savefig(os.path.join(figs, "aciertos_por_variante.png"), dpi=150); plt.close()

# --------------------  7. Main --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--salida", type=str, default=RUNS_DIR)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    run_dir = os.path.join(args.salida, f"estudio_prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"llm": cfg["llm"], "n_por_clase_y_mapa": args.n, "mapas": MAPAS, "clases": CLASES,
                   "variantes": list(VARIANTES), "semilla_episodios": SEMILLA_EPISODIOS}, f, indent=2, ensure_ascii=False)
    correr_estudio(cfg, args.n, run_dir)
    print(f"\n=== ESTUDIO COMPLETADO | {run_dir} ===")
