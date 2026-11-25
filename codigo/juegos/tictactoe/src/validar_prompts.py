# ======================  Tic-Tac-Toe: validación de los prompts antes de entrenar  ======================
# Mide qué tan bien replica el LLM la recompensa estándar, con turnos y partidas de respuesta conocida:
#   replica_jugada    (T1): turnos que terminan en GANA, PIERDE, BLOQUEA, EMPATE u OTRA
#   replica_resultado (T2, T4): partidas completas que terminan en GANA, EMPATE o PIERDE
# Los turnos salen de partidas entre dos jugadores aleatorios; el agente juega con la letra del config.
# Uso:
#   python validar_prompts.py --config ../configs/T1.json [--n 30] [--salida DIR]
import os
import json
import math
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import TicTacToe, categoria_jugada, categoria_partida, jugadas_ganadoras, rival_de
from prompts import DIRECTIVAS, categorias, palabra_de, construir_prompt_jugada, construir_prompt_partida
from llm_reward import crear_cliente, nuevas_stats, llm_reward

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
SEMILLA_CASOS = 2026
MAX_PARTIDAS = 200_000

# --------------------  1. Casos de prueba --------------------
def partida_aleatoria(rng, letra, agente_empieza):
    """Partida entre dos jugadores aleatorios. Regresa los turnos del agente y la partida completa."""
    env, rival = TicTacToe(), rival_de(letra)
    historial, turnos = [], []
    if not agente_empieza:
        a = int(rng.choice(env.jugadas_legales())); env.jugar(a, rival); historial.append((rival, a))
    while not env.terminado():
        antes = env.board.copy()
        a = int(rng.choice(env.jugadas_legales())); env.jugar(a, letra); historial.append((letra, a))
        despues, a_rival = env.board.copy(), None
        if not env.terminado():
            a_rival = int(rng.choice(env.jugadas_legales())); env.jugar(a_rival, rival); historial.append((rival, a_rival))
        turnos.append({"antes": antes, "accion": a, "despues": despues, "accion_rival": a_rival,
                       "final": env.board.copy(), "terminada": env.terminado(), "ganador": env.ganador,
                       "rival_amenazaba": bool(jugadas_ganadoras(antes, rival))})
    partida = {"historial": historial, "final": env.board.copy(), "ganador": env.ganador}
    return turnos, partida

def generar_casos(n, letra, agente_empieza, semilla):
    """{'replica_jugada': {palabra: [turnos]}, 'replica_resultado': {palabra: [partidas]}}, n por palabra."""
    rng = np.random.default_rng(semilla)
    casos = {d: {p: [] for p in DIRECTIVAS[d]["categorias"]} for d in DIRECTIVAS}
    vistos = set()
    for _ in range(MAX_PARTIDAS):
        turnos, partida = partida_aleatoria(rng, letra, agente_empieza)
        for t in turnos:
            p = palabra_de("replica_jugada", categoria_jugada(t["antes"], t["accion"], t["final"], letra))
            clave = (tuple(t["antes"]), t["accion"], t["accion_rival"])
            if len(casos["replica_jugada"][p]) < n and clave not in vistos:
                vistos.add(clave)
                casos["replica_jugada"][p].append(t)
        p = palabra_de("replica_resultado", categoria_partida(partida["final"], letra))
        clave = tuple(partida["historial"])
        if len(casos["replica_resultado"][p]) < n and clave not in vistos:
            vistos.add(clave)
            casos["replica_resultado"][p].append(partida)
        if all(len(v) >= n for d in casos.values() for v in d.values()):
            break
    for d, por_palabra in casos.items():
        for p, lista in por_palabra.items():
            if len(lista) < n:
                print(f"[WARN] {d}/{p}: solo {len(lista)} casos")
    return casos

# --------------------  2. Intervalo de confianza y resumen --------------------
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
    for llave, g in df.groupby(por, sort=False):
        llave = llave if isinstance(llave, tuple) else (llave,)
        aciertos, n = int(g["correcto"].sum()), len(g)
        bajo, alto = wilson(aciertos, n)
        filas.append({**dict(zip(por, llave)), "n": n, "aciertos": aciertos,
                      "aciertos_%": round(100 * aciertos / n, 1), "ic95_bajo_%": bajo, "ic95_alto_%": alto,
                      "sin_respuesta_valida": int(g["categoria"].isna().sum()),
                      "tokens_salida_prom": round(g["tokens_salida"].mean(), 2),
                      "latencia_prom_s": round(g["latencia_s"].mean(), 3)})
    return pd.DataFrame(filas)

# --------------------  3. Estudio --------------------
def correr_validacion(cfg, n, run_dir):
    letra, empieza = cfg["juego"]["letra_agente"], cfg["juego"]["agente_empieza"]
    valores = cfg["recompensa"]["valores"]
    cliente, stats = crear_cliente(cfg["llm"]), nuevas_stats()
    casos = generar_casos(n, letra, empieza, SEMILLA_CASOS)
    filas, ejemplos = [], {}

    for directiva, por_palabra in casos.items():
        cats = categorias(directiva, valores)
        for esperado, lista in por_palabra.items():
            for i, c in enumerate(lista):
                if DIRECTIVAS[directiva]["tipo"] == "jugada":
                    prompt = construir_prompt_jugada(c["antes"], c["accion"], c["despues"], c["accion_rival"], c["final"],
                                                     letra, c["terminada"], c["ganador"], directiva, valores)
                    extra = {"rival_amenazaba": c["rival_amenazaba"]}
                else:
                    prompt = construir_prompt_partida(c["historial"], c["final"], letra, c["ganador"], directiva, valores)
                    extra = {"rival_amenazaba": None}
                _, reg = llm_reward(cliente, prompt, cfg["llm"], stats, cats)
                filas.append({"directiva": directiva, "esperado": esperado, "caso": i, **extra,
                              "tablero_final": "".join(x if x != ' ' else '.' for x in c["final"]),
                              "categoria": reg["categoria"], "respuesta": reg["respuesta"],
                              "correcto": reg["categoria"] == esperado,
                              "tokens_entrada": reg["tokens_entrada"], "tokens_salida": reg["tokens_salida"],
                              "latencia_s": reg["latencia_s"]})
                ejemplos.setdefault((directiva, esperado), (prompt, reg["respuesta"]))
        pd.DataFrame(filas).to_csv(os.path.join(run_dir, "resultados.csv"), index=False)
        print(f"=== {directiva} terminada | {len(filas)} llamadas acumuladas | fallbacks={stats['fallbacks_totales']} ===")

    df = pd.DataFrame(filas)
    total = resumir(df, ["directiva"])
    por_clase = resumir(df, ["directiva", "esperado"])
    confusion = pd.crosstab([df["directiva"], df["esperado"]], df["categoria"].fillna("(sin respuesta)"))
    total.to_csv(os.path.join(run_dir, "resumen_por_directiva.csv"), index=False)
    por_clase.to_csv(os.path.join(run_dir, "resumen_por_directiva_clase.csv"), index=False)
    confusion.to_csv(os.path.join(run_dir, "confusion.csv"))
    with open(os.path.join(run_dir, "llm_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(run_dir, "prompts_ejemplo.txt"), "w", encoding="utf-8") as f:
        for (directiva, esperado), (prompt, texto) in ejemplos.items():
            f.write(f"########## {directiva} (respuesta correcta: {esperado})\n{prompt}\n>>> Respuesta del LLM: {texto!r}\n\n")

    guardar_figuras(por_clase, run_dir)
    print("\nAciertos por directiva:")
    print(total[["directiva", "n", "aciertos_%", "ic95_bajo_%", "ic95_alto_%", "sin_respuesta_valida"]].to_string(index=False))
    print("\nAciertos por respuesta correcta:")
    print(por_clase[["directiva", "esperado", "n", "aciertos_%", "ic95_bajo_%", "ic95_alto_%"]].to_string(index=False))
    print("\nQué respondió el LLM (filas: respuesta correcta):")
    print(confusion.to_string())

# --------------------  4. Figuras --------------------
def guardar_figuras(por_clase, run_dir):
    figs = os.path.join(run_dir, "figs")
    os.makedirs(figs, exist_ok=True)
    for directiva in DIRECTIVAS:
        d = por_clase[por_clase["directiva"] == directiva]
        if d.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 1.2 + 0.6 * len(d)))
        err = [d["aciertos_%"] - d["ic95_bajo_%"], d["ic95_alto_%"] - d["aciertos_%"]]
        ax.barh(range(len(d)), d["aciertos_%"], xerr=err, capsize=3)
        ax.set_yticks(range(len(d))); ax.set_yticklabels(d["esperado"])
        ax.invert_yaxis(); ax.set_xlim(0, 105); ax.grid(axis="x", alpha=0.3)
        ax.set_xlabel("Aciertos (%) con IC 95%"); ax.set_ylabel("Respuesta correcta")
        ax.set_title(f"Directiva '{directiva}': aciertos del LLM")
        plt.tight_layout(); plt.savefig(os.path.join(figs, f"aciertos_{directiva}.png"), dpi=150); plt.close()

# --------------------  5. Main --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--salida", type=str, default=RUNS_DIR)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    run_dir = os.path.join(args.salida, f"validacion_prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"llm": cfg["llm"], "juego": cfg["juego"], "valores": cfg["recompensa"]["valores"],
                   "n_por_respuesta": args.n, "semilla_casos": SEMILLA_CASOS,
                   "directivas": {k: {c: v[c] for c in v} for k, v in DIRECTIVAS.items()}},
                  f, indent=2, ensure_ascii=False)
    correr_validacion(cfg, args.n, run_dir)
    print(f"\n=== VALIDACIÓN COMPLETADA | {run_dir} ===")
