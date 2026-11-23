# ======================  Frozen Lake: estudio de qué directivas entiende el LLM  ======================
# Para cada variante de directivas.py, el LLM evalúa episodios con respuesta conocida y se mide cuántas acierta.
# Las variantes de una misma familia ven exactamente los mismos episodios, así que solo cambia cómo se pregunta.
# Solo mapas aleatorios (en el mapa fijo 4x4 no existe ninguna ruta que vaya toda por el borde).
# Uso:
#   python estudio_directivas.py --config ../configs/B2.json [--n 30] [--salida DIR]
import os
import json
import random
import argparse
from collections import deque
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import make_env, semillas_de_mapas, desc_a_filas, ACCION_NOMBRE
from llm_reward import crear_cliente, nuevas_stats, llm_reward
from estudio_prompts import generar_episodios, jugar, resumir, MAPAS
from directivas import FAMILIAS, VARIANTES, categorias, construir_prompt, llego, es_borde, casillas_interior

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
SEMILLA_EPISODIOS = 2026
MAX_PASOS = 14     # rutas a la meta de a lo más 14 pasos: que la longitud no delate la respuesta
N_MAPAS = 3000

# --------------------  1. Rutas a la meta con propiedades variadas --------------------
MOV = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

def bfs(desc, inicio, fin, permitido):
    """Acciones de la ruta más corta de inicio a fin pisando solo casillas permitidas. None si no existe."""
    H, W = desc.shape
    previo, cola = {inicio: None}, deque([inicio])
    while cola:
        r, c = cola.popleft()
        if (r, c) == fin:
            acciones = []
            while previo[(r, c)]:
                (r, c), a = previo[(r, c)]
                acciones.append(a)
            return acciones[::-1]
        for a, (dr, dc) in MOV.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in previo and permitido(nr, nc):
                previo[(nr, nc)] = ((r, c), a)
                cola.append((nr, nc))
    return None

def rutas_meta(desc, rng):
    """La más corta, la más corta por el borde, la más corta por el interior y tres con desvío por una casilla al azar."""
    H, W = desc.shape
    S, G = (0, 0), (H - 1, W - 1)
    libre = lambda r, c: desc[r, c] != b'H'
    rutas = [bfs(desc, S, G, libre),
             bfs(desc, S, G, lambda r, c: libre(r, c) and es_borde(r, c, H, W))]
    interiores = [(r, c) for r in range(1, H - 1) for c in range(1, W - 1) if libre(r, c)]
    libres = [(r, c) for r in range(H) for c in range(W) if libre(r, c) and (r, c) not in (S, G)]
    for w in rng.sample(interiores, min(1, len(interiores))) + rng.sample(libres, min(3, len(libres))):
        ida = bfs(desc, S, w, lambda r, c: libre(r, c) and (r, c) != G)
        vuelta = bfs(desc, w, G, libre)
        if ida is not None and vuelta is not None:
            rutas.append(ida + vuelta)
    return [r for r in rutas if r]

def jugar_ruta(env, desc, acciones):
    s, _ = env.reset()
    estados = [int(s)]
    term = trunc = False
    for a in acciones:
        s, _, term, trunc, _ = env.step(a)
        estados.append(int(s))
        if term or trunc:
            break
    return (desc, estados, list(acciones[:len(estados) - 1]), term, trunc, None)

# --------------------  2. Episodios de prueba --------------------
def generar_casos(n, semilla):
    """{familia: {clase: [n episodios]}}. NO_META es la misma lista para todas las familias."""
    rng = random.Random(semilla)
    pool, vistos = [], set()
    for sm in semillas_de_mapas(semilla, N_MAPAS):
        env, desc = make_env(MAPAS["aleatorio"], sm)
        eps = [jugar_ruta(env, desc, ruta) for ruta in rutas_meta(desc, rng)] + [jugar(env, desc, "META", rng)]
        for ep in eps:
            clave = ("|".join(desc_a_filas(desc)), tuple(ep[2]))
            if llego(ep) and len(ep[2]) <= MAX_PASOS and clave not in vistos:
                vistos.add(clave)
                pool.append(ep)
    otros = generar_episodios(MAPAS["aleatorio"], (n + 1) // 2, semilla)
    no_meta = (otros["HOYO"] + otros["TRUNCADO"])[:n]

    casos = {}
    for familia, f in FAMILIAS.items():
        casos[familia] = {}
        for clase in f["clases"]:
            if clase == "NO_META":
                casos[familia][clase] = no_meta
                continue
            candidatos = [ep for ep in pool if f["clase"](ep) == clase]
            rng.shuffle(candidatos)
            casos[familia][clase] = candidatos[:n]
            if len(candidatos) < n:
                print(f"[WARN] {familia}/{clase}: solo {len(candidatos)} episodios")
    return casos

# --------------------  3. Estudio --------------------
def correr_estudio(cfg, n, run_dir):
    cliente, stats = crear_cliente(cfg["llm"]), nuevas_stats()
    casos = generar_casos(n, SEMILLA_EPISODIOS)
    filas, ejemplos = [], {}

    for variante, v in VARIANTES.items():
        cats = categorias(variante)
        for clase, eps in casos[v["familia"]].items():
            for i, ep in enumerate(eps):
                prompt = construir_prompt(ep, variante)
                esperado = v["esperado"](ep)
                _, reg = llm_reward(cliente, prompt, cfg["llm"], stats, cats,
                                    marcador=v.get("marcador"), max_tokens=v.get("max_tokens"))
                respuesta = reg["categoria"]
                filas.append({"variante": variante, "familia": v["familia"], "clase": clase, "episodio": i,
                              "pasos": len(ep[2]), "casillas_interior": casillas_interior(ep),
                              "repite_casilla": len(set(ep[1])) < len(ep[1]),
                              "mapa_desc": "|".join(desc_a_filas(ep[0])),
                              "acciones": "".join(ACCION_NOMBRE[a][0] for a in ep[2]),
                              "esperado": esperado, "categoria": respuesta, "respuesta": reg["respuesta"],
                              "valor": None if reg["fallback"] else reg["r_llm"],
                              "correcto": respuesta == esperado,
                              "correcto_binario": (respuesta in v["exito"]) == (esperado in v["exito"]) if respuesta else False,
                              "tokens_entrada": reg["tokens_entrada"], "tokens_salida": reg["tokens_salida"],
                              "latencia_s": reg["latencia_s"]})
                if variante not in ejemplos and (clase not in ("META_SI", "META_BORDE", "NO_META")):
                    ejemplos[variante] = (prompt, reg["respuesta"], esperado)
        pd.DataFrame(filas).to_csv(os.path.join(run_dir, "resultados.csv"), index=False)
        print(f"=== {variante} terminada | {len(filas)} llamadas acumuladas | fallbacks={stats['fallbacks_totales']} ===")

    df = pd.DataFrame(filas)
    total = resumir(df, ["variante"])
    binario = resumir(df.assign(correcto=df["correcto_binario"]), ["variante"])
    total["aciertos_binario_%"] = binario["aciertos_%"].values
    por_clase = resumir(df, ["variante", "clase"])
    total.to_csv(os.path.join(run_dir, "resumen_por_variante.csv"), index=False)
    por_clase.to_csv(os.path.join(run_dir, "resumen_por_variante_clase.csv"), index=False)
    with open(os.path.join(run_dir, "llm_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(run_dir, "prompts_ejemplo.txt"), "w", encoding="utf-8") as f:
        for variante, (prompt, texto, esperado) in ejemplos.items():
            f.write(f"########## {variante} (respuesta correcta: {esperado})\n{prompt}\n>>> Respuesta del LLM: {texto!r}\n\n")

    guardar_figuras(por_clase, total, run_dir)
    print("\nAciertos por variante:")
    print(total[["variante", "n", "aciertos_%", "ic95_bajo_%", "ic95_alto_%", "aciertos_binario_%",
                 "sin_respuesta_valida", "tokens_salida_prom"]].to_string(index=False))
    for familia, f in FAMILIAS.items():
        d = por_clase[por_clase["variante"].map(lambda v: VARIANTES[v]["familia"]) == familia]
        print(f"\nAciertos por clase | familia {familia}:")
        print(d.pivot(index="variante", columns="clase", values="aciertos_%")[f["clases"]].to_string())

# --------------------  4. Figuras --------------------
def guardar_figuras(por_clase, total, run_dir):
    figs = os.path.join(run_dir, "figs")
    os.makedirs(figs, exist_ok=True)
    variantes = list(VARIANTES)

    d = total.set_index("variante").loc[variantes]
    fig, ax = plt.subplots(figsize=(10, 5))
    err = [d["aciertos_%"] - d["ic95_bajo_%"], d["ic95_alto_%"] - d["aciertos_%"]]
    ax.barh(range(len(variantes)), d["aciertos_%"], xerr=err, capsize=3)
    ax.set_yticks(range(len(variantes))); ax.set_yticklabels(variantes)
    ax.invert_yaxis(); ax.set_xlim(0, 105); ax.grid(axis="x", alpha=0.3)
    ax.set_xlabel("Aciertos (%) con IC 95%")
    ax.set_title("Aciertos del LLM según la directiva (total)")
    plt.tight_layout(); plt.savefig(os.path.join(figs, "aciertos_por_variante.png"), dpi=150); plt.close()

    for familia, f in FAMILIAS.items():
        vs = [v for v in variantes if VARIANTES[v]["familia"] == familia]
        clases = f["clases"]
        fig, ax = plt.subplots(figsize=(10, 1.2 + 1.1 * len(vs)))
        alto = 0.8 / len(clases)
        for j, clase in enumerate(clases):
            dc = por_clase[por_clase["clase"] == clase].set_index("variante").reindex(vs)
            y = [i + (j - (len(clases) - 1) / 2) * alto for i in range(len(vs))]
            err = [dc["aciertos_%"] - dc["ic95_bajo_%"], dc["ic95_alto_%"] - dc["aciertos_%"]]
            ax.barh(y, dc["aciertos_%"], height=alto, xerr=err, capsize=2, label=clase)
        ax.set_yticks(range(len(vs))); ax.set_yticklabels(vs)
        ax.invert_yaxis(); ax.set_xlim(0, 105); ax.grid(axis="x", alpha=0.3)
        ax.set_xlabel("Aciertos (%) con IC 95%")
        ax.set_title(f"Familia '{familia}': aciertos por tipo de episodio")
        ax.legend(title="Episodio", loc="upper left", bbox_to_anchor=(1.01, 1))
        plt.tight_layout(); plt.savefig(os.path.join(figs, f"aciertos_{familia}_por_clase.png"), dpi=150); plt.close()

# --------------------  5. Main --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--salida", type=str, default=RUNS_DIR)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    run_dir = os.path.join(args.salida, f"estudio_directivas_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"llm": cfg["llm"], "n_por_clase": args.n, "mapa": MAPAS["aleatorio"], "max_pasos": MAX_PASOS,
                   "semilla_episodios": SEMILLA_EPISODIOS,
                   "familias": {k: v["clases"] for k, v in FAMILIAS.items()},
                   "variantes": {k: {c: v[c] for c in v if c != "esperado"} for k, v in VARIANTES.items()}},
                  f, indent=2, ensure_ascii=False)
    correr_estudio(cfg, args.n, run_dir)
    print(f"\n=== ESTUDIO COMPLETADO | {run_dir} ===")
