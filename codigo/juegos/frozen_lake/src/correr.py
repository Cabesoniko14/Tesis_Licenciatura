# ======================  Frozen Lake: correr un experimento desde su config  ======================
# Uso:
#   python correr.py --config ../configs/B3.json [--salida DIR] [--humo] [--traza]
#   python correr.py --reanudar ../runs/B3_20261125_153000
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

from common import (set_global_seeds, semillas_de_mapas, make_env, desc_a_filas,
                    encode_obs_from_desc, metricas_ruta, ACCION_NOMBRE)
from prompts import construir_prompt, categorias
from agentes import QLearningAgent, DQNAgent
from llm_reward import crear_cliente, nuevas_stats, llm_reward

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")

# --------------------  1. Config --------------------
def cargar_config(ruta, humo=False):
    with open(ruta, encoding="utf-8") as f:
        cfg = json.load(f)
    if humo:
        cfg["entrenamiento"].update(epocas=2, episodios_por_epoca=10, checkpoint_cada=1)
        cfg["evaluacion"]["episodios"] = min(cfg["evaluacion"]["episodios"], 10)
        if cfg["dqn"]:
            cfg["dqn"]["inicio_aprendizaje"] = 64
    return cfg

def crear_agente(cfg):
    if cfg["agente"] == "qlearning":
        q = cfg["qlearning"]
        n = cfg["mapa"]["tamano"] ** 2
        return QLearningAgent(n, 4, alpha=q["alpha"], gamma=q["gamma"], epsilon=q["epsilon"])
    return DQNAgent(cfg["dqn"], n_actions=4, device="cpu")

# --------------------  2. Un episodio --------------------
def jugar_episodio(cfg, agente, env, desc, entrenar=True, cliente=None, llm_stats=None):
    """Juega un episodio. Si entrenar=True actualiza el agente y, si toca, llama al LLM al final."""
    es_dqn = cfg["agente"] == "dqn"
    usa_llm = entrenar and cfg["recompensa"]["fuente"] == "llm"
    s, _ = env.reset()
    s = int(s)
    obs = encode_obs_from_desc(desc, s) if es_dqn else None
    estados, acciones, transiciones = [s], [], []
    registro_llm, prompt = None, None
    terminated = truncated = False
    r_env = 0.0

    while not (terminated or truncated):
        a = agente.choose_action(obs if es_dqn else s, greedy=not entrenar)
        s2, r_env, terminated, truncated, _ = env.step(a)
        s2, r_env = int(s2), float(r_env)
        estados.append(s2)
        acciones.append(a)
        done = terminated or truncated

        r = r_env
        if usa_llm:
            r = 0.0
            if done:
                prompt = construir_prompt(desc, estados, acciones, terminated, truncated, cfg["recompensa"]["directiva"])
                r, registro_llm = llm_reward(cliente, prompt, cfg["llm"], llm_stats,
                                             categorias(cfg["recompensa"]["directiva"]))

        obs2 = encode_obs_from_desc(desc, s2) if es_dqn else None
        if entrenar:
            eps = agente.epsilon() if es_dqn else agente.epsilon
            if es_dqn:
                agente.step(obs, a, r, obs2, terminated)
            else:
                agente.update(s, a, r, s2, terminated)
            transiciones.append({"paso": len(acciones), "s": s, "a": a, "s2": s2, "r_env": r_env, "r": r,
                                 "terminated": terminated, "truncated": truncated, "epsilon": eps})
        s, obs = s2, obs2

    if terminated and r_env == 1.0:
        resultado = "META"
    elif terminated:
        resultado = "HOYO"
    else:
        resultado = "TRUNCADO"
    return {"resultado": resultado, "estados": estados, "acciones": acciones, "transiciones": transiciones,
            "r_env": r_env, "registro_llm": registro_llm, "prompt": prompt,
            "terminated": terminated, "truncated": truncated}

# --------------------  3. Evaluación greedy --------------------
def evaluar(cfg, agente, envs_eval):
    victorias = sum(jugar_episodio(cfg, agente, env, desc, entrenar=False)["resultado"] == "META" for env, desc in envs_eval)
    return 100.0 * victorias / len(envs_eval)

def crear_envs_eval(cfg):
    if cfg["mapa"]["modo"] == "fijo":
        return [make_env(cfg["mapa"])]
    semillas = semillas_de_mapas(cfg["evaluacion"]["semilla_mapas_prueba"], cfg["evaluacion"]["episodios"])
    return [make_env(cfg["mapa"], sm) for sm in semillas]

# --------------------  4. Checkpoints --------------------
def guardar_checkpoint(run_dir, epoca, agente, tablas, llm_stats, tiempo_total):
    estado = {"epoca": epoca, "agente": agente.estado(), "tablas": tablas, "llm_stats": llm_stats,
              "tiempo_total": tiempo_total, "rng_python": random.getstate(),
              "rng_numpy": np.random.get_state(), "rng_torch": torch.get_rng_state()}
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
    plt.plot(df_epocas["epoca"], df_epocas["r_env_promedio"], label="Recompensa de Gym")
    if df_epocas["r_llm_promedio"].notna().any():
        plt.plot(df_epocas["epoca"], df_epocas["r_llm_promedio"], label="Recompensa del LLM")
    plt.xlabel("Época"); plt.ylabel("Recompensa promedio por episodio"); plt.title(f"{run_id}: recompensa por época")
    plt.ylim(0, 1.05); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "recompensas.png"), dpi=150); plt.close()

    if df_epocas["derrotas_premiadas"].notna().any():
        plt.figure(figsize=(8, 4))
        plt.plot(df_epocas["epoca"], df_epocas["derrotas_premiadas"], label="Derrotas premiadas (r_llm > 0)")
        plt.plot(df_epocas["epoca"], df_epocas["victorias_castigadas"], label="Victorias castigadas (r_llm = 0)")
        plt.xlabel("Época"); plt.ylabel("Episodios"); plt.title(f"{run_id}: desacuerdo LLM vs Gym")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, "desacuerdo_llm.png"), dpi=150); plt.close()

# --------------------  6. Entrenamiento --------------------
def correr(cfg, run_dir, reanudar=False):
    ent = cfg["entrenamiento"]
    set_global_seeds(ent["semilla"], ent["determinista"])
    usa_llm = cfg["recompensa"]["fuente"] == "llm"
    cliente = crear_cliente(cfg["llm"]) if usa_llm else None
    agente = crear_agente(cfg)
    envs_eval = crear_envs_eval(cfg)
    fijo = cfg["mapa"]["modo"] == "fijo"
    if fijo:
        env_fijo, desc_fijo = make_env(cfg["mapa"])
    else:
        semillas = semillas_de_mapas(cfg["mapa"]["semilla_mapas"], ent["epocas"] * ent["episodios_por_epoca"])

    tablas = {"transiciones": [], "episodios": [], "epocas": [], "llm_llamadas": []}
    llm_stats = nuevas_stats()
    tiempo_total, epoca_inicio = 0.0, 0

    if reanudar:
        with open(os.path.join(run_dir, "checkpoint.pkl"), "rb") as f:
            ck = pickle.load(f)
        agente.cargar_estado(ck["agente"])
        tablas, llm_stats, tiempo_total, epoca_inicio = ck["tablas"], ck["llm_stats"], ck["tiempo_total"], ck["epoca"]
        random.setstate(ck["rng_python"]); np.random.set_state(ck["rng_numpy"]); torch.set_rng_state(ck["rng_torch"])
        print(f"[REANUDAR] desde la época {epoca_inicio}/{ent['epocas']}")

    for epoca in range(epoca_inicio + 1, ent["epocas"] + 1):
        filas_ep = []
        for ep in range(1, ent["episodios_por_epoca"] + 1):
            n_global = (epoca - 1) * ent["episodios_por_epoca"] + (ep - 1)
            if fijo:
                env, desc, seed_mapa = env_fijo, desc_fijo, None
            else:
                seed_mapa = semillas[n_global]
                env, desc = make_env(cfg["mapa"], seed_mapa)

            t0 = time.perf_counter()
            res = jugar_episodio(cfg, agente, env, desc, entrenar=True, cliente=cliente, llm_stats=llm_stats)
            dt = time.perf_counter() - t0
            tiempo_total += dt
            if not fijo:
                env.close()

            for t in res["transiciones"]:
                tablas["transiciones"].append({"epoca": epoca, "episodio": ep, **t})
            reg = res["registro_llm"]
            if reg:
                tablas["llm_llamadas"].append({"epoca": epoca, "episodio": ep, "r_env": res["r_env"], **reg})
            fila = {"epoca": epoca, "episodio": ep, "seed_mapa": seed_mapa, "mapa": "|".join(desc_a_filas(desc)),
                    "resultado": res["resultado"], "r_env": res["r_env"], "r_llm": reg["r_llm"] if reg else None,
                    "categoria_llm": reg["categoria"] if reg else None,
                    "tiempo_s": dt, **metricas_ruta(desc, res["estados"]),
                    "acciones": "".join(ACCION_NOMBRE[a][0] for a in res["acciones"])}
            filas_ep.append(fila)
            tablas["episodios"].append(fila)

        df_ep = pd.DataFrame(filas_ep)
        ganadas = df_ep["resultado"] == "META"
        fila_epoca = {
            "epoca": epoca,
            "winrate_entrenamiento": 100.0 * ganadas.mean(),
            "winrate_greedy": evaluar(cfg, agente, envs_eval),
            "epsilon": agente.epsilon() if cfg["agente"] == "dqn" else agente.epsilon,
            "r_env_promedio": df_ep["r_env"].mean(),
            "r_llm_promedio": df_ep["r_llm"].mean() if usa_llm else None,
            "derrotas_premiadas": int(((df_ep["r_llm"] > 0) & ~ganadas).sum()) if usa_llm else None,
            "victorias_castigadas": int(((df_ep["r_llm"] == 0) & ganadas).sum()) if usa_llm else None,
            "fallbacks_acumulados": llm_stats["fallbacks_totales"],
        }
        tablas["epocas"].append(fila_epoca)
        print(f"=== {cfg['id']} | Epoch {epoca}/{ent['epocas']} | winrate={fila_epoca['winrate_entrenamiento']:.1f}% "
              f"| greedy={fila_epoca['winrate_greedy']:.1f}%"
              + (f" | derrotas premiadas={fila_epoca['derrotas_premiadas']} | victorias castigadas={fila_epoca['victorias_castigadas']}"
                 f" | fallbacks={llm_stats['fallbacks_totales']}" if usa_llm else "") + " ===")

        if epoca % ent["checkpoint_cada"] == 0:
            guardar_checkpoint(run_dir, epoca, agente, tablas, llm_stats, tiempo_total)

    # --------------------  Guardado final --------------------
    guardar_tablas(run_dir, tablas)
    if cfg["agente"] == "dqn":
        torch.save(agente.policy.state_dict(), os.path.join(run_dir, "policy.pth"))
    else:
        np.save(os.path.join(run_dir, "qtable.npy"), agente.Q)

    df_epocas = pd.DataFrame(tablas["epocas"])
    df_eps = pd.DataFrame(tablas["episodios"])
    resumen = {
        "id": cfg["id"],
        "descripcion": cfg["descripcion"],
        "episodios": len(df_eps),
        "victorias": int((df_eps["resultado"] == "META").sum()),
        "hoyos": int((df_eps["resultado"] == "HOYO").sum()),
        "truncados": int((df_eps["resultado"] == "TRUNCADO").sum()),
        "winrate_entrenamiento_%": round(100.0 * (df_eps["resultado"] == "META").mean(), 2),
        "winrate_greedy_final_%": round(float(df_epocas["winrate_greedy"].iloc[-1]), 2),
        "tiempo_total_segundos": round(tiempo_total, 2),
        "llm": llm_stats if usa_llm else None,
        "python": sys.version.split()[0], "torch": torch.__version__,
    }
    if usa_llm:
        ganadas = df_eps["resultado"] == "META"
        resumen["acuerdo_llm"] = {
            "derrotas_premiadas": int(((df_eps["r_llm"] > 0) & ~ganadas).sum()),
            "victorias_castigadas": int(((df_eps["r_llm"] == 0) & ganadas).sum()),
            "tiempo_llm_%": round(100.0 * llm_stats["tiempo_respuesta_total_segundos"] / tiempo_total, 1) if tiempo_total else None,
        }
    with open(os.path.join(run_dir, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    guardar_figuras(run_dir, df_epocas, cfg["id"])
    ck = os.path.join(run_dir, "checkpoint.pkl")
    if os.path.exists(ck):
        os.remove(ck)
    print(f"\n=== {cfg['id']} COMPLETADO | {run_dir} ===")

# --------------------  7. Traza de un episodio (ejemplo para SN-1) --------------------
def traza(cfg, run_dir):
    if cfg["recompensa"]["fuente"] != "llm":
        print(f"[TRAZA] {cfg['id']} no usa LLM: no hay prompt que mostrar")
        return
    set_global_seeds(cfg["entrenamiento"]["semilla"], cfg["entrenamiento"]["determinista"])
    seed_mapa = None if cfg["mapa"]["modo"] == "fijo" else semillas_de_mapas(cfg["mapa"]["semilla_mapas"], 1)[0]
    env, desc = make_env(cfg["mapa"], seed_mapa)
    res = jugar_episodio(cfg, crear_agente(cfg), env, desc, entrenar=True,
                         cliente=crear_cliente(cfg["llm"]), llm_stats=nuevas_stats())
    reg = res["registro_llm"]
    texto = (f"EXPERIMENTO: {cfg['id']} | {cfg['descripcion']}\n"
             f"Resultado del episodio: {res['resultado']} | recompensa de Gym: {res['r_env']}\n\n"
             f"===== PROMPT =====\n{res['prompt']}\n"
             f"===== RESPUESTA DEL LLM =====\n{reg['respuesta']!r}\n\n"
             f"===== REGLA =====\nPrimera palabra de la respuesta que sea una categoría válida, convertida a recompensa: "
             f"{categorias(cfg['recompensa']['directiva'])}; {cfg['llm']['reintentos']} intentos; "
             f"si no hay categoría válida, r = {cfg['llm']['fallback']}\n\n"
             f"===== RECOMPENSA =====\ncategoría = {reg['categoria']} -> r = {reg['r_llm']} (fallback={reg['fallback']}) asignada a la última transición; "
             f"las {len(res['acciones']) - 1} anteriores valen 0\n"
             f"tokens entrada={reg['tokens_entrada']} | salida={reg['tokens_salida']} | latencia={reg['latencia_s']:.2f} s\n")
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
