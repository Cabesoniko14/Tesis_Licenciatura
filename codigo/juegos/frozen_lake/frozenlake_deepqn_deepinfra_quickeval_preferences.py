# ======================  FrozenLake: traza 2 episodios, LLM replica al std  ======================
# Corre 2 episodios con acciones aleatorias (no entrena nada -- esto es solo para comparar la
# función de recompensa). Un solo paso del entorno por trayectoria; sobre esa MISMA trayectoria
# se calculan las dos recompensas en paralelo: la del std (paso directo del reward de Gym) y la
# del LLM (que debe replicar exactamente lo mismo, sin preferencias de ruta).
import os
import re
import random
from datetime import datetime
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from openai import OpenAI

NUM_JUEGOS = 2
SEED = 42
MAP_SIZE = 4
IS_SLIPPERY = False
MAX_STEPS = 30

LLM_BASE_URL = "https://api.deepinfra.com/v1/openai"
LLM_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
LLM_TEMPERATURE = 0.0
LLM_MAX_RETRIES = 3

ACCION_NOMBRE = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

def make_random_env(map_size, is_slippery, seed):
    desc = generate_random_map(size=map_size, seed=seed)
    env = gym.make("FrozenLake-v1", desc=desc, is_slippery=is_slippery, render_mode=None)
    return env

def tile_at(desc, idx, w):
    r, c = idx // w, idx % w
    v = desc[r, c]
    return v.decode("ascii") if isinstance(v, (bytes, np.bytes_)) else str(v)

def grid_str(desc):
    H, W = desc.shape
    filas = []
    for r in range(H):
        fila = "".join(tile_at(desc, r * W + c, W) for c in range(W))
        filas.append(fila)
    return "\n".join(filas)

_PROMPT_TEMPLATE = """Eres una función de recompensas para un episodio de FrozenLake.

El episodio terminó: {terminated}
Llegó a la meta ('G'): {llego_meta}

Si llegó a la meta, responde 1.
Si no llegó a la meta, responde 0.

Responde solo con el número.
"""

def _parse_reward_value(s: str):
    matches = re.findall(r"[-+]?\d*\.?\d+", s.strip())
    if not matches:
        return None
    try:
        v = float(matches[-1])
        return max(0.0, min(1.0, v))
    except:
        return None

def llm_reward_final(client, terminated, llego_meta, max_retries=LLM_MAX_RETRIES):
    for _ in range(max_retries):
        try:
            message = _PROMPT_TEMPLATE.format(terminated=terminated, llego_meta=llego_meta)
            resp = client.chat.completions.create(
                messages=[{"role": "user", "content": message}],
                model=LLM_MODEL_NAME,
                temperature=LLM_TEMPERATURE,
                max_tokens=20,
            )
            content = (resp.choices[0].message.content or "").strip()
            print(f"[LLM RAW] {content!r}")
            parsed = _parse_reward_value(content)
            if parsed is not None:
                return parsed, content
        except Exception as e:
            print(f"[LLM Error] {e}")
    return 0.0, "(sin respuesta válida)"

def jugar_episodio(client, seed_juego, log):
    env = make_random_env(MAP_SIZE, IS_SLIPPERY, seed_juego)
    obs, _ = env.reset(seed=seed_juego)
    desc = env.unwrapped.desc
    W = desc.shape[1]

    log.append(f"Mapa (S=inicio, F=seguro, H=hoyo, G=meta):\n{grid_str(desc)}\n")

    s_idx = int(obs)
    acumulado_std = 0.0
    acumulado_llm = 0.0
    terminated = False
    truncated = False
    paso = 1

    while not (terminated or truncated) and paso <= MAX_STEPS:
        a = random.randint(0, 3)
        next_obs, r_env, terminated, truncated, _ = env.step(a)
        ns_idx = int(next_obs)

        done = terminated or truncated
        # std: paso directo del reward de Gym (0 en cada paso, 1 solo si llega a la meta)
        reward_std = float(r_env) if done else 0.0

        if not done:
            reward_llm = 0.0
            respuesta_cruda = "(episodio no terminó, no se consulta al LLM en este paso)"
        else:
            llego_meta = bool(terminated) and float(r_env) == 1.0
            reward_llm, respuesta_cruda = llm_reward_final(client, terminated, llego_meta)

        acumulado_std += reward_std
        acumulado_llm += reward_llm

        tile_ns = tile_at(desc, ns_idx, W)
        log.append(f"[Paso {paso}] estado={s_idx} -> acción={ACCION_NOMBRE[a]} -> estado={ns_idx} (casilla '{tile_ns}')")
        if done:
            log.append(f"  terminated={terminated}  truncated={truncated}  env_reward={r_env}")
            log.append(f"  LLM respondió: '{respuesta_cruda}'")
        log.append(f"  Reward STD: {reward_std:+.2f}  |  acumulado STD: {acumulado_std:+.2f}")
        log.append(f"  Reward LLM: {reward_llm:+.2f}  |  acumulado LLM: {acumulado_llm:+.2f}")
        log.append("")

        s_idx = ns_idx
        paso += 1

    env.close()

    if terminated and float(r_env) == 1.0:
        resultado = "META"
    elif terminated:
        resultado = "HOYO"
    else:
        resultado = "TRUNCADO (se acabaron los pasos)"
    return resultado, acumulado_std, acumulado_llm

def encabezado():
    return (
        f"MODELO LLM: {LLM_MODEL_NAME} (vía DeepInfra)\n"
        f"MAPA: {MAP_SIZE}x{MAP_SIZE}  |  is_slippery={IS_SLIPPERY}\n"
        "STD: reward = el reward nativo de Gym en cada paso (0 siempre, 1 solo si llega a 'G').\n"
        "Este experimento le pide al LLM que replique EXACTAMENTE eso, sin preferencias de ruta.\n\n"
        "PROMPT DEL LLM (se consulta solo en el paso final del episodio)\n"
        f"{_PROMPT_TEMPLATE}\n"
    )

def correr_traza(client, num_juegos=NUM_JUEGOS):
    log = [encabezado()]

    for i in range(num_juegos):
        log.append(f"--- Episodio {i+1} ---")
        resultado, acum_std, acum_llm = jugar_episodio(client, SEED + i, log)
        log.append(f">>> {resultado} | acumulado STD={acum_std:+.2f}  |  acumulado LLM={acum_llm:+.2f}\n")

    texto_completo = "\n".join(log)
    print(texto_completo)

    os.makedirs("datos_output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modelo_tag = re.sub(r"[^A-Za-z0-9.\-]+", "", LLM_MODEL_NAME.split("/")[-1])
    ruta = f"datos_output/traza_frozenlake_2_episodios_{modelo_tag}_{timestamp}.txt"
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(texto_completo)

    print(f"\n[OK] guardado en: {ruta}")
    return ruta


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepinfra-api-key", type=str, default=None)
    parser.add_argument("--num-juegos", type=int, default=NUM_JUEGOS)
    args = parser.parse_args()

    deepinfra_api_key = (
        args.deepinfra_api_key
        or os.environ.get("DEEPINFRA_API_KEY")
        or os.environ.get("DEEPINFRA")
    )
    if not deepinfra_api_key:
        raise RuntimeError(
            "Falta la API key de DeepInfra. Pásala con DEEPINFRA_API_KEY=tu_key antes del comando, "
            "o con --deepinfra-api-key tu_key."
        )

    client = OpenAI(api_key=deepinfra_api_key, base_url=LLM_BASE_URL)
    correr_traza(client, num_juegos=args.num_juegos)