# ======================  Tic-Tac-Toe: correr un experimento desde su config  ======================
# Uso:
#   python correr.py --config ../configs/T0.json [--salida DIR] [--humo] [--traza]
#   python correr.py --reanudar ../runs/T0_20261125_153000
import os
import sys
import json
import time
import pickle
import random
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import TicTacToe, set_global_seeds, categoria_jugada, categoria_partida, encode_obs, mascara, rival_de
from prompts import categorias, palabra_de, construir_prompt_jugada, construir_prompt_partida, tablero_texto
from agentes import QLearningAgent, DQNAgent
from llm_reward import crear_cliente, nuevas_stats, llm_reward

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
RESULTADO = {"gana": "GANA", "empate": "EMPATE", "pierde": "PIERDE"}

# --------------------  1. Config --------------------
def cargar_config(ruta, humo=False):
    with open(ruta, encoding="utf-8") as f:
        cfg = json.load(f)
    if humo:
        cfg["entrenamiento"].update(epocas=2, episodios_por_epoca=10, checkpoint_cada=1)
        cfg["evaluacion"]["partidas"] = min(cfg["evaluacion"]["partidas"], 10)
        if cfg["dqn"]:
            cfg["dqn"]["inicio_aprendizaje"] = 64
    return cfg

def crear_agente(cfg):
    if cfg["agente"] == "qlearning":
        q = cfg["qlearning"]
        return QLearningAgent(alpha=q["alpha"], gamma=q["gamma"], epsilon=q["epsilon"])
    return DQNAgent(cfg["dqn"], n_actions=9, device="cpu")

# --------------------  2. Una partida --------------------
def jugar_partida(cfg, agente, rng_rival, entrenar=True, cliente=None, llm_stats=None):
    """Juega una partida contra el rival aleatorio. Si entrenar=True actualiza el agente y, si toca, llama al LLM.
    Un turno = jugada del agente + respuesta del rival; la recompensa del turno se asigna a la jugada del agente."""
    es_dqn = cfg["agente"] == "dqn"
    rec = cfg["recompensa"]
    usa_llm = entrenar and rec["fuente"] == "llm"
    letra = cfg["juego"]["letra_agente"]
    rival = rival_de(letra)
    env = TicTacToe()
    historial, transiciones, llamadas = [], [], []

    if not cfg["juego"]["agente_empieza"]:
        a_rival = int(rng_rival.choice(env.jugadas_legales()))
        env.jugar(a_rival, rival)
        historial.append((rival, a_rival))

    while not env.terminado():
        board_antes = env.board.copy()
        legales = env.jugadas_legales()
        obs = encode_obs(board_antes, letra) if es_dqn else None
        a = agente.choose_action(obs if es_dqn else board_antes, legales, greedy=not entrenar)
        env.jugar(a, letra)
        historial.append((letra, a))
        board_agente = env.board.copy()

        a_rival = None
        if not env.terminado():
            a_rival = int(rng_rival.choice(env.jugadas_legales()))
            env.jugar(a_rival, rival)
            historial.append((rival, a_rival))
        board_final = env.board.copy()
        terminada = env.terminado()
        cat = categoria_jugada(board_antes, a, board_final, letra)

        # --- Recompensa del turno ---
        r = 0.0
        if rec["cuando"] == "por_jugada":
            if usa_llm:
                prompt = construir_prompt_jugada(board_antes, a, board_agente, a_rival, board_final, letra,
                                                 terminada, env.ganador, rec["directiva"], rec["valores"])
                r, reg = llm_reward(cliente, prompt, cfg["llm"], llm_stats, categorias(rec["directiva"], rec["valores"]))
                llamadas.append({"turno": len(transiciones) + 1, "esperado": palabra_de(rec["directiva"], cat),
                                 "prompt": prompt, **reg})
            else:
                r = rec["valores"][cat]
        elif terminada:
            cat_p = categoria_partida(board_final, letra)
            if usa_llm:
                prompt = construir_prompt_partida(historial, board_final, letra, env.ganador, rec["directiva"], rec["valores"])
                r, reg = llm_reward(cliente, prompt, cfg["llm"], llm_stats, categorias(rec["directiva"], rec["valores"]))
                llamadas.append({"turno": len(transiciones) + 1, "esperado": palabra_de(rec["directiva"], cat_p),
                                 "prompt": prompt, **reg})
            else:
                r = rec["valores"][cat_p]

        if entrenar:
            legales2 = env.jugadas_legales()
            eps = agente.epsilon() if es_dqn else agente.epsilon
            if es_dqn:
                agente.step(obs, a, r, encode_obs(board_final, letra), terminada, mascara(legales2))
            else:
                agente.update(board_antes, a, r, board_final, terminada, legales2)
            transiciones.append({"turno": len(transiciones) + 1, "tablero_antes": "".join(c if c != ' ' else '.' for c in board_antes),
                                 "accion": a, "accion_rival": a_rival, "categoria_estandar": cat, "r": r,
                                 "terminada": terminada, "epsilon": eps})

    return {"resultado": RESULTADO[categoria_partida(env.board, letra)], "historial": historial,
            "transiciones": transiciones, "llamadas": llamadas, "board_final": env.board.copy()}

# --------------------  3. Evaluación greedy --------------------
def evaluar(cfg, agente):
    """Mismas partidas del rival en cada evaluación (semilla fija). Regresa % de victorias, empates y derrotas."""
    rng = np.random.default_rng(cfg["evaluacion"]["semilla_prueba"])
    n = cfg["evaluacion"]["partidas"]
    res = [jugar_partida(cfg, agente, rng, entrenar=False)["resultado"] for _ in range(n)]
    return {k: 100.0 * res.count(k) / n for k in ("GANA", "EMPATE", "PIERDE")}

# --------------------  4. Checkpoints --------------------
def guardar_checkpoint(run_dir, epoca, agente, rng_rival, tablas, llm_stats, tiempo_total):
    estado = {"epoca": epoca, "agente": agente.estado(), "tablas": tablas, "llm_stats": llm_stats,
              "tiempo_total": tiempo_total, "rng_rival": rng_rival.bit_generator.state,
              "rng_python": random.getstate(), "rng_numpy": np.random.get_state(), "rng_torch": torch.get_rng_state()}
    with open(os.path.join(run_dir, "checkpoint.pkl"), "wb") as f:
        pickle.dump(estado, f)
    guardar_tablas(run_dir, tablas)

def guardar_tablas(run_dir, tablas):
    for nombre, filas in tablas.items():
        pd.DataFrame(filas).to_csv(os.path.join(run_dir, f"{nombre}.csv"), index=False)

# --------------------  5. Figuras generales --------------------
def guardar_figuras(run_dir, df_epocas, run_id):
    figs_dir = os.path.join(run_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(df_epocas["epoca"], df_epocas["winrate_entrenamiento"], label="Entrenamiento (con ε)")
    plt.plot(df_epocas["epoca"], df_epocas["winrate_greedy"], label="Evaluación greedy")
    plt.xlabel("Época"); plt.ylabel("Victorias (%)"); plt.title(f"{run_id}: victorias por época")
    plt.ylim(0, 100); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "victorias.png"), dpi=150); plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(df_epocas["epoca"], df_epocas["winrate_greedy"], label="Victorias")
    plt.plot(df_epocas["epoca"], df_epocas["empates_greedy"], label="Empates")
    plt.plot(df_epocas["epoca"], df_epocas["derrotas_greedy"], label="Derrotas")
    plt.xlabel("Época"); plt.ylabel("Partidas (%)"); plt.title(f"{run_id}: resultados de la evaluación greedy")
    plt.ylim(0, 100); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "resultados_greedy.png"), dpi=150); plt.close()

    if df_epocas["acuerdo_llm_%"].notna().any():
        plt.figure(figsize=(8, 4))
        plt.plot(df_epocas["epoca"], df_epocas["acuerdo_llm_%"])
        plt.xlabel("Época"); plt.ylabel("Acuerdo (%)"); plt.title(f"{run_id}: acuerdo del LLM con la recompensa estándar")
        plt.ylim(0, 100); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, "acuerdo_llm.png"), dpi=150); plt.close()

# --------------------  6. Entrenamiento --------------------
def correr(cfg, run_dir, reanudar=False):
    ent = cfg["entrenamiento"]
    set_global_seeds(ent["semilla"], ent["determinista"])
    usa_llm = cfg["recompensa"]["fuente"] == "llm"
    cliente = crear_cliente(cfg["llm"]) if usa_llm else None
    agente = crear_agente(cfg)
    rng_rival = np.random.default_rng(ent["semilla"])

    tablas = {"transiciones": [], "partidas": [], "epocas": [], "llm_llamadas": []}
    llm_stats = nuevas_stats()
    tiempo_total, epoca_inicio = 0.0, 0

    if reanudar:
        with open(os.path.join(run_dir, "checkpoint.pkl"), "rb") as f:
            ck = pickle.load(f)
        agente.cargar_estado(ck["agente"])
        tablas, llm_stats, tiempo_total, epoca_inicio = ck["tablas"], ck["llm_stats"], ck["tiempo_total"], ck["epoca"]
        rng_rival.bit_generator.state = ck["rng_rival"]
        random.setstate(ck["rng_python"]); np.random.set_state(ck["rng_numpy"]); torch.set_rng_state(ck["rng_torch"])
        print(f"[REANUDAR] desde la época {epoca_inicio}/{ent['epocas']}")

    for epoca in range(epoca_inicio + 1, ent["epocas"] + 1):
        filas_ep, llamadas_ep = [], []
        for ep in range(1, ent["episodios_por_epoca"] + 1):
            t0 = time.perf_counter()
            res = jugar_partida(cfg, agente, rng_rival, entrenar=True, cliente=cliente, llm_stats=llm_stats)
            dt = time.perf_counter() - t0
            tiempo_total += dt

            for t in res["transiciones"]:
                tablas["transiciones"].append({"epoca": epoca, "partida": ep, **t})
            for ll in res["llamadas"]:
                fila_ll = {"epoca": epoca, "partida": ep, **{k: v for k, v in ll.items() if k != "prompt"}}
                tablas["llm_llamadas"].append(fila_ll)
                llamadas_ep.append(fila_ll)
            fila = {"epoca": epoca, "partida": ep, "resultado": res["resultado"],
                    "jugadas": " ".join(f"{l}{i}" for l, i in res["historial"]),
                    "turnos": len(res["transiciones"]),
                    "bloqueos": sum(t["categoria_estandar"] == "bloquea" for t in res["transiciones"]),
                    "r_total": sum(t["r"] for t in res["transiciones"]),
                    "llamadas_llm": len(res["llamadas"]),
                    "desacuerdos_llm": sum(ll["categoria"] != ll["esperado"] for ll in res["llamadas"]),
                    "tiempo_s": dt}
            filas_ep.append(fila)
            tablas["partidas"].append(fila)

        df_ep = pd.DataFrame(filas_ep)
        greedy = evaluar(cfg, agente)
        acuerdo = (100.0 * np.mean([ll["categoria"] == ll["esperado"] for ll in llamadas_ep])) if llamadas_ep else None
        fila_epoca = {
            "epoca": epoca,
            "winrate_entrenamiento": 100.0 * (df_ep["resultado"] == "GANA").mean(),
            "empates_entrenamiento": 100.0 * (df_ep["resultado"] == "EMPATE").mean(),
            "derrotas_entrenamiento": 100.0 * (df_ep["resultado"] == "PIERDE").mean(),
            "winrate_greedy": greedy["GANA"], "empates_greedy": greedy["EMPATE"], "derrotas_greedy": greedy["PIERDE"],
            "epsilon": agente.epsilon() if cfg["agente"] == "dqn" else agente.epsilon,
            "r_promedio": df_ep["r_total"].mean(),
            "acuerdo_llm_%": acuerdo,
            "fallbacks_acumulados": llm_stats["fallbacks_totales"],
        }
        tablas["epocas"].append(fila_epoca)
        print(f"=== {cfg['id']} | Epoch {epoca}/{ent['epocas']} | winrate={fila_epoca['winrate_entrenamiento']:.1f}% "
              f"| greedy V/E/D={greedy['GANA']:.0f}/{greedy['EMPATE']:.0f}/{greedy['PIERDE']:.0f}%"
              + (f" | acuerdo LLM={acuerdo:.1f}% | fallbacks={llm_stats['fallbacks_totales']}" if usa_llm else "") + " ===")

        if epoca % ent["checkpoint_cada"] == 0:
            guardar_checkpoint(run_dir, epoca, agente, rng_rival, tablas, llm_stats, tiempo_total)

    # --------------------  Guardado final --------------------
    guardar_tablas(run_dir, tablas)
    if cfg["agente"] == "dqn":
        torch.save(agente.policy.state_dict(), os.path.join(run_dir, "policy.pth"))
    else:
        with open(os.path.join(run_dir, "qtable.pkl"), "wb") as f:
            pickle.dump(dict(agente.Q), f)

    df_epocas = pd.DataFrame(tablas["epocas"])
    df_par = pd.DataFrame(tablas["partidas"])
    resumen = {
        "id": cfg["id"],
        "descripcion": cfg["descripcion"],
        "partidas": len(df_par),
        "victorias": int((df_par["resultado"] == "GANA").sum()),
        "empates": int((df_par["resultado"] == "EMPATE").sum()),
        "derrotas": int((df_par["resultado"] == "PIERDE").sum()),
        "winrate_entrenamiento_%": round(100.0 * (df_par["resultado"] == "GANA").mean(), 2),
        "greedy_final_%": {"victorias": float(df_epocas["winrate_greedy"].iloc[-1]),
                           "empates": float(df_epocas["empates_greedy"].iloc[-1]),
                           "derrotas": float(df_epocas["derrotas_greedy"].iloc[-1])},
        "tiempo_total_segundos": round(tiempo_total, 2),
        "llm": llm_stats if usa_llm else None,
        "python": sys.version.split()[0], "torch": torch.__version__,
    }
    if usa_llm and tablas["llm_llamadas"]:
        df_ll = pd.DataFrame(tablas["llm_llamadas"])
        resumen["acuerdo_llm"] = {
            "acuerdo_%": round(100.0 * (df_ll["categoria"] == df_ll["esperado"]).mean(), 2),
            "por_categoria_esperada_%": {k: round(100.0 * (g["categoria"] == g["esperado"]).mean(), 2)
                                         for k, g in df_ll.groupby("esperado")},
            "tiempo_llm_%": round(100.0 * llm_stats["tiempo_respuesta_total_segundos"] / tiempo_total, 1) if tiempo_total else None,
        }
    with open(os.path.join(run_dir, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    guardar_figuras(run_dir, df_epocas, cfg["id"])
    ck = os.path.join(run_dir, "checkpoint.pkl")
    if os.path.exists(ck):
        os.remove(ck)
    print(f"\n=== {cfg['id']} COMPLETADO | {run_dir} ===")

# --------------------  7. Traza de una partida (ejemplo para la tesis) --------------------
def traza(cfg, run_dir):
    if cfg["recompensa"]["fuente"] != "llm":
        print(f"[TRAZA] {cfg['id']} no usa LLM: no hay prompt que mostrar")
        return
    set_global_seeds(cfg["entrenamiento"]["semilla"], cfg["entrenamiento"]["determinista"])
    rec = cfg["recompensa"]
    res = jugar_partida(cfg, crear_agente(cfg), np.random.default_rng(cfg["entrenamiento"]["semilla"]),
                        entrenar=True, cliente=crear_cliente(cfg["llm"]), llm_stats=nuevas_stats())
    texto = (f"EXPERIMENTO: {cfg['id']} | {cfg['descripcion']}\n"
             f"Resultado de la partida: {res['resultado']} | jugadas: {' '.join(f'{l}{i}' for l, i in res['historial'])}\n"
             f"Tablero final:\n{tablero_texto(res['board_final'])}\n\n"
             f"===== REGLA =====\nPrimera palabra de la respuesta que sea una categoría válida, convertida a recompensa: "
             f"{categorias(rec['directiva'], rec['valores'])}; {cfg['llm']['reintentos']} intentos; "
             f"si no hay categoría válida, r = {cfg['llm']['fallback']}\n")
    for ll in res["llamadas"]:
        texto += (f"\n===== TURNO {ll['turno']} | PROMPT =====\n{ll['prompt']}\n"
                  f"===== RESPUESTA DEL LLM =====\n{ll['respuesta']!r}\n"
                  f"categoría = {ll['categoria']} (correcta: {ll['esperado']}) -> r = {ll['r_llm']} (fallback={ll['fallback']})\n"
                  f"tokens entrada={ll['tokens_entrada']} | salida={ll['tokens_salida']} | latencia={ll['latencia_s']:.2f} s\n")
    print(texto)
    with open(os.path.join(run_dir, "traza.txt"), "w", encoding="utf-8") as f:
        f.write(texto)

# --------------------  8. Main --------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", default=None)
    parser.add_argument("--salida", type=str, default=RUNS_DIR)
    parser.add_argument("--humo", action="store_true")
    parser.add_argument("--traza", action="store_true")
    parser.add_argument("--reanudar", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.reanudar:
        with open(os.path.join(args.reanudar, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        correr(cfg, args.reanudar, reanudar=True)
        sys.exit(0)

    for ruta in args.config:
        cfg = cargar_config(ruta, humo=args.humo)
        sufijo = "_traza" if args.traza else ("_humo" if args.humo else "")
        run_id = f"{cfg['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{sufijo}"
        run_dir = os.path.join(args.salida, run_id)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)

        if args.traza:
            traza(cfg, run_dir)
        else:
            correr(cfg, run_dir)
