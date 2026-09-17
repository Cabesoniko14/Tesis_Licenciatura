# ======================  DQN: FrozenLake + LLM Rewards (DeepInfra) — replica std  ======================
import os
import sys
import re
import json
import time
import random
import argparse
import psutil
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import openai
from openai import OpenAI

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# --------------------  0. Metadatos --------------------
MODEL_TYPE = "DeepQN"
USA_LLM = True
REWARD_LEVEL = "episodio"    # el LLM evalúa una vez al final de cada episodio, no por paso
LLM_REWARD_STRATEGY = "replica_std_binaria"  # 1 si llegó a la meta, 0 si no -- igual que el reward nativo de Gym
REWARD_TAG = "llmReplicaStd"
SEED = 42
CHECKPOINT_EVERY = 10

LLM_PROVIDER = "DeepInfra"
LLM_BASE_URL = "https://api.deepinfra.com/v1/openai"
LLM_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
LLM_TEMPERATURE = 0.0
LLM_MAX_RETRIES = 3

# --------------------  1. Reproducibilidad --------------------
def set_global_seeds(seed=None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --------------------  2. Observación (mapa + agente) -> tensor --------------------
TILE_TO_CH = {b'F': 0, b'H': 1, b'G': 2, b'S': 0}

def idx_to_rc(idx, size):
    return idx // size, idx % size

def encode_obs_from_desc(desc, agent_idx):
    H, W = desc.shape
    obs = np.zeros((4, H, W), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            obs[TILE_TO_CH.get(desc[r, c], 0), r, c] = 1.0
    ar, ac = idx_to_rc(agent_idx, W)
    obs[3, ar, ac] = 1.0
    return obs

def tile_at(desc, idx, w):
    r, c = idx // w, idx % w
    v = desc[r, c]
    return v.decode("ascii") if isinstance(v, (bytes, np.bytes_)) else str(v)

# --------------------  3. Red y Replay Buffer --------------------
class QNetCNN(nn.Module):
    """CNN compacta para 4 u 8 celdas por lado."""
    def __init__(self, in_channels=4, n_actions=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        feats = self.features(x)
        if H == 4 and W == 4:
            feats = nn.functional.interpolate(feats, size=(8, 8), mode='nearest')
        return self.head(feats)

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.capacity = capacity
        self.buf = []
        self.pos = 0

    def push(self, s, a, r, s2, d):
        if len(self.buf) < self.capacity:
            self.buf.append(None)
        self.buf[self.pos] = (s, a, r, s2, d)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = map(np.array, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)

# --------------------  4. Envs helpers --------------------
def make_fixed_env(map_size=4, is_slippery=False):
    assert map_size in (4, 8), "map_size fijo soportado: 4 u 8"
    env = gym.make("FrozenLake-v1", map_name=f"{map_size}x{map_size}", is_slippery=is_slippery)
    return env, f"{map_size}x{map_size}"

def make_random_env(map_size=4, is_slippery=False, seed=None):
    desc = generate_random_map(size=map_size, seed=seed)
    env = gym.make("FrozenLake-v1", desc=desc, is_slippery=is_slippery)
    return env, desc

# --------------------  5. Recompensa vía LLM (replica el std) --------------------
# Episódica: se consulta UNA vez al terminar el episodio (terminated o truncated), no por paso.
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

def llm_reward_final(client, terminated, llego_meta, stats, max_retries=LLM_MAX_RETRIES):
    stats["invocaciones_totales"] += 1
    for _ in range(max_retries):
        stats["llamadas_api_totales"] += 1
        try:
            message = _PROMPT_TEMPLATE.format(terminated=terminated, llego_meta=llego_meta)
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                messages=[{"role": "user", "content": message}],
                model=LLM_MODEL_NAME,
                temperature=LLM_TEMPERATURE,
                max_tokens=20,
            )
            stats["tiempo_respuesta_total_segundos"] += time.perf_counter() - t0
            content = (resp.choices[0].message.content or "").strip()
            parsed = _parse_reward_value(content)
            if parsed is not None:
                return parsed, False
        except Exception as e:
            stats["errores_api_totales"] += 1
            print(f"[LLM Error] {e}")
    stats["fallbacks_totales"] += 1
    print(f"[LLM WARN] Respuesta inválida tras {max_retries} intentos -> uso 0.0 de fallback")
    return 0.0, True

# --------------------  6. Resumen de modelos (resumen_modelos/) --------------------
def _slug_modelo_llm(nombre_modelo: str) -> str:
    corto = nombre_modelo.split("/")[-1]
    return re.sub(r"[^A-Za-z0-9.\-]+", "", corto)

def construir_resumen_modelo(
    base_filename, timestamp, num_epochs, episodes_per_epoch, total_episodes,
    tiempo_total, resultados_finales, estructura_recompensas,
    hiperparametros=None, red_neuronal=None, llm_info=None, entorno=None,
):
    return {
        "archivo_base": base_filename,
        "timestamp": timestamp,
        "modelo": {
            "tipo": MODEL_TYPE,
            "usa_llm": USA_LLM,
            "estrategia_recompensa_llm": LLM_REWARD_STRATEGY if USA_LLM else None,
        },
        "entrenamiento": {
            "num_epochs": num_epochs,
            "episodes_per_epoch": episodes_per_epoch,
            "total_episodes": total_episodes,
            "tiempo_total_segundos": round(tiempo_total, 4),
            "tiempo_promedio_por_episodio_segundos": (
                round(tiempo_total / total_episodes, 6) if total_episodes else None
            ),
            "hiperparametros": hiperparametros or {},
        },
        "resultados_finales": resultados_finales,
        "estructura_recompensas": estructura_recompensas,
        "red_neuronal": red_neuronal,
        "llm": llm_info,
        "entorno": entorno or {},
    }

def guardar_resumen_modelo(resumen, carpeta="resumen_modelos"):
    os.makedirs(carpeta, exist_ok=True)
    nombre_archivo = f"resumen_{resumen['archivo_base']}"

    ruta_json = os.path.join(carpeta, f"{nombre_archivo}.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    ruta_txt = os.path.join(carpeta, f"{nombre_archivo}.txt")
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write(f"RESUMEN DE MODELO: {resumen['archivo_base']}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Tipo de modelo : {resumen['modelo']['tipo']}\n")
        f.write(f"Usa LLM        : {'Sí' if resumen['modelo']['usa_llm'] else 'No'}\n")
        if resumen["modelo"].get("estrategia_recompensa_llm"):
            f.write(f"Estrategia LLM : {resumen['modelo']['estrategia_recompensa_llm']}\n")
        f.write(f"Timestamp      : {resumen['timestamp']}\n\n")

        f.write("-- Entrenamiento --\n")
        for k, v in resumen["entrenamiento"].items():
            if k == "hiperparametros":
                f.write("  hiperparametros:\n")
                for hk, hv in v.items():
                    f.write(f"    {hk}: {hv}\n")
            else:
                f.write(f"  {k}: {v}\n")

        f.write("\n-- Resultados finales --\n")
        for k, v in resumen["resultados_finales"].items():
            f.write(f"  {k}: {v}\n")

        f.write("\n-- Estructura de recompensas --\n")
        f.write(f"  nivel_aplicacion: {resumen['estructura_recompensas'].get('nivel')}\n")
        f.write("  componentes:\n")
        for comp, val in resumen["estructura_recompensas"].get("componentes", {}).items():
            f.write(f"    {comp}: {val}\n")

        f.write("\n-- Red neuronal --\n")
        if resumen["red_neuronal"]:
            for k, v in resumen["red_neuronal"].items():
                f.write(f"  {k}: {v}\n")
        else:
            f.write("  N/A\n")

        f.write("\n-- LLM --\n")
        if resumen["llm"]:
            for k, v in resumen["llm"].items():
                if k == "prompt_template":
                    f.write(f"  prompt_template:\n{v}\n")
                elif k == "estadisticas_de_llamadas":
                    f.write("  estadisticas_de_llamadas:\n")
                    for sk, sv in v.items():
                        f.write(f"    {sk}: {sv}\n")
                else:
                    f.write(f"  {k}: {v}\n")
        else:
            f.write("  N/A (no se usó LLM en este experimento)\n")

        if resumen.get("entorno"):
            f.write("\n-- Entorno --\n")
            for k, v in resumen["entorno"].items():
                f.write(f"  {k}: {v}\n")

    print(f"[OK] Resumen de modelo guardado en:\n  {ruta_json}\n  {ruta_txt}")
    return ruta_json, ruta_txt

# --------------------  7. Entrenamiento DQN --------------------
def train_dqn_frozenlake(
    client,
    num_epochs=100,
    episodes_per_epoch=100,
    map_size=4,
    is_slippery=False,
    max_steps_per_ep=50,
    map_mode="per_episode_random",
    seed=SEED,
    gamma=0.99,
    lr=1e-3,
    buffer_capacity=100_000,
    batch_size=64,
    start_learning_after=2_000,
    train_every=1,
    target_update_every=1_000,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=50_000,
):
    assert map_mode in ("per_episode_random", "fixed")
    set_global_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_actions = 4

    llm_stats = {
        "invocaciones_totales": 0,
        "llamadas_api_totales": 0,
        "fallbacks_totales": 0,
        "errores_api_totales": 0,
        "tiempo_respuesta_total_segundos": 0.0,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_episodes = num_epochs * episodes_per_epoch
    modelo_llm_tag = f"_{_slug_modelo_llm(LLM_MODEL_NAME)}"
    base = (f"frozenlake_{MODEL_TYPE}_conLLM{modelo_llm_tag}_{map_mode}_size{map_size}_"
            f"{'slip' if is_slippery else 'noslip'}_{REWARD_TAG}_{timestamp}_ep{total_episodes}")
    os.makedirs("datos_output", exist_ok=True)
    os.makedirs("modelos", exist_ok=True)

    acciones_df  = pd.DataFrame(columns=["Epoch","Episodio","Paso","StateIdx","Accion","NextStateIdx","Reward","Done","Epsilon","LLM_Fallback"])
    computo_df   = pd.DataFrame(columns=["Epoch","Episodio","Tiempo(s)","CPU(%)","RAM(MB)","GPU_mem(MB)"])
    victorias_df = pd.DataFrame(columns=["Epoch","Exitos","Fracasos","WinRate(%)"])

    rng = np.random.default_rng(seed)
    fixed_env = fixed_meta = None
    if map_mode == "fixed":
        fixed_env, fixed_meta = make_fixed_env(map_size=map_size, is_slippery=is_slippery)

    policy = QNetCNN(in_channels=4, n_actions=n_actions).to(device)
    target = QNetCNN(in_channels=4, n_actions=n_actions).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_capacity)
    step_count = 0

    def get_epsilon(t):
        if eps_decay_steps <= 0:
            return eps_end
        frac = min(1.0, t / eps_decay_steps)
        return eps_start + (eps_end - eps_start) * frac

    total_time = 0.0
    successes_total, fails_total = 0, 0

    for epoch in range(num_epochs):
        success_epoch, fail_epoch = 0, 0

        for ep in range(episodes_per_epoch):
            t0 = time.perf_counter()

            if map_mode == "per_episode_random":
                env, desc = make_random_env(map_size=map_size, is_slippery=is_slippery, seed=int(rng.integers(0, 1e9)))
            else:
                env, desc = fixed_env, fixed_env.unwrapped.desc

            obs, _ = env.reset()
            s_idx = int(obs)
            desc = env.unwrapped.desc
            obs_tensor = encode_obs_from_desc(desc, s_idx)
            done = False

            for t in range(max_steps_per_ep):
                eps = get_epsilon(step_count)
                if np.random.rand() < eps:
                    a = np.random.randint(n_actions)
                else:
                    with torch.no_grad():
                        x = torch.from_numpy(obs_tensor).unsqueeze(0).to(device)
                        a = int(torch.argmax(policy(x), dim=1).item())

                next_obs, r_env, terminated, truncated, _ = env.step(a)
                ns_idx = int(next_obs)
                done = terminated or truncated
                next_obs_tensor = encode_obs_from_desc(desc, ns_idx)

                fue_fallback = False
                if not done:
                    r = 0.0
                else:
                    llego_meta = bool(terminated) and float(r_env) == 1.0
                    r, fue_fallback = llm_reward_final(client, terminated, llego_meta, llm_stats)

                buffer.push(obs_tensor, a, r, next_obs_tensor, float(done))
                acciones_df.loc[len(acciones_df)] = [epoch+1, ep+1, t+1, s_idx, a, ns_idx, r, done, eps, fue_fallback]

                if len(buffer) >= start_learning_after and step_count % train_every == 0:
                    s_b, a_b, r_b, s2_b, d_b = buffer.sample(batch_size)
                    s_b_t  = torch.from_numpy(s_b).to(device)
                    a_b_t  = torch.from_numpy(a_b).long().to(device)
                    r_b_t  = torch.from_numpy(r_b).float().to(device)
                    s2_b_t = torch.from_numpy(s2_b).to(device)
                    d_b_t  = torch.from_numpy(d_b).float().to(device)

                    qsa = policy(s_b_t).gather(1, a_b_t.view(-1,1)).squeeze(1)
                    with torch.no_grad():
                        next_actions = policy(s2_b_t).argmax(dim=1)
                        q_next = target(s2_b_t).gather(1, next_actions.view(-1,1)).squeeze(1)
                        target_q = r_b_t + gamma * (1.0 - d_b_t) * q_next
                    loss = nn.MSELoss()(qsa, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                    optimizer.step()

                if step_count % target_update_every == 0:
                    target.load_state_dict(policy.state_dict())

                step_count += 1
                obs_tensor = next_obs_tensor
                s_idx = ns_idx

                if done:
                    if float(r_env) > 0:
                        successes_total += 1; success_epoch += 1
                    else:
                        fails_total += 1; fail_epoch += 1
                    break

            elapsed = time.perf_counter() - t0
            total_time += elapsed
            computo_df.loc[len(computo_df)] = [
                epoch+1, ep+1, elapsed, psutil.cpu_percent(),
                psutil.Process(os.getpid()).memory_info().rss/1024/1024,
                torch.cuda.memory_allocated()/1024/1024 if torch.cuda.is_available() else 0
            ]

            if map_mode == "per_episode_random":
                env.close()

        wr = 100.0 * success_epoch / episodes_per_epoch
        victorias_df.loc[len(victorias_df)] = [epoch+1, success_epoch, fail_epoch, wr]
        print(f"=== Epoch {epoch+1}/{num_epochs} terminado | winrate={wr:.2f}% "
              f"| LLM fallbacks acumulados={llm_stats['fallbacks_totales']} ===")

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            acciones_df.to_csv(f"datos_output/acciones_{base}.csv", index=False)
            computo_df.to_csv(f"datos_output/computo_{base}.csv", index=False)
            victorias_df.to_csv(f"datos_output/victorias_{base}.csv", index=False)
            torch.save(policy.state_dict(), f"modelos/policy_{base}.pth")
            print(f"[CHECKPOINT] Progreso guardado hasta epoch {epoch+1}/{num_epochs}")

    if map_mode == "fixed" and fixed_env is not None:
        fixed_env.close()

    win_rate_global = 100.0 * successes_total / total_episodes if total_episodes else 0.0

    acciones_df.to_csv(f"datos_output/acciones_{base}.csv", index=False)
    computo_df.to_csv(f"datos_output/computo_{base}.csv", index=False)
    victorias_df.to_csv(f"datos_output/victorias_{base}.csv", index=False)
    torch.save(policy.state_dict(), f"modelos/policy_{base}.pth")
    torch.save(target.state_dict(), f"modelos/target_{base}.pth")

    resultados_finales = {
        "exitos": successes_total,
        "fracasos": fails_total,
        "win_rate_%": round(win_rate_global, 2),
    }

    estructura_recompensas = {
        "nivel": REWARD_LEVEL,  # "episodio": se consulta al LLM una sola vez, al terminar el episodio
        "componentes": {
            "fuente": "LLM (DeepInfra) — replica el reward nativo de Gym, sin preferencias de ruta",
            "rango_valores": "{0, 1}",
            "regla": "1 si terminated=True y env_reward==1.0 (llegó a 'G'); 0 en cualquier otro caso (hoyo o truncado por límite de pasos)",
        },
    }

    hiperparametros = {
        "gamma": gamma, "lr": lr, "buffer_capacity": buffer_capacity, "batch_size": batch_size,
        "start_learning_after": start_learning_after, "train_every": train_every,
        "target_update_every": target_update_every, "eps_start": eps_start, "eps_end": eps_end,
        "eps_decay_steps": eps_decay_steps, "seed": seed,
        "map_size": map_size, "is_slippery": is_slippery, "map_mode": map_mode,
        "max_steps_per_ep": max_steps_per_ep,
    }

    red_neuronal = {
        "arquitectura": "QNetCNN (Double DQN)",
        "entrada": "tensor (4, H, W): canales [FROZEN, HOLE, GOAL, AGENT] one-hot sobre el mapa",
        "capas_conv": "Conv2d(4->32, k3, pad1) + ReLU -> Conv2d(32->64, k3, pad1) + ReLU",
        "adaptacion_tamano": "si el mapa es 4x4, se interpola a 8x8 (nearest) antes de la head, para reusar la misma head en mapas 4x4 y 8x8",
        "head": "Flatten -> Linear(64*8*8 -> 256) + ReLU -> Linear(256 -> n_actions=4)",
        "funcion_de_perdida": "MSELoss",
        "optimizador": "Adam",
        "variante": "Double DQN (acción elegida por policy, evaluada por target)",
        "red_objetivo": f"se sincroniza con policy cada {target_update_every} steps",
    }

    entorno = {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "torch_version": torch.__version__,
        "openai_version": getattr(openai, "__version__", "desconocida"),
        "gymnasium_version": gym.__version__,
    }

    llm_info = {
        "proveedor": LLM_PROVIDER,
        "modelo_llm": LLM_MODEL_NAME,
        "temperature": LLM_TEMPERATURE,
        "estrategia_recompensa": LLM_REWARD_STRATEGY,
        "api_key_source": "variable de entorno DEEPINFRA_API_KEY (o DEEPINFRA) o --deepinfra-api-key",
        "cuando_se_invoca": "Una vez por episodio, solo cuando termina (terminated o truncated). En pasos intermedios el reward es 0.0 sin consultar al LLM.",
        "max_retries_por_llamada": LLM_MAX_RETRIES,
        "parseo_respuesta": "Se extrae el último número de la respuesta con regex y se recorta a [0, 1]; si no hay número parseable, se reintenta.",
        "valor_fallback_si_falla": 0.0,
        "prompt_template": _PROMPT_TEMPLATE,
        "estadisticas_de_llamadas": {
            "invocaciones_totales": llm_stats["invocaciones_totales"],
            "llamadas_api_totales": llm_stats["llamadas_api_totales"],
            "fallbacks_totales": llm_stats["fallbacks_totales"],
            "fallback_rate_%": (
                round(100 * llm_stats["fallbacks_totales"] / llm_stats["invocaciones_totales"], 2)
                if llm_stats["invocaciones_totales"] else 0.0
            ),
            "errores_api_totales": llm_stats["errores_api_totales"],
            "tiempo_respuesta_promedio_segundos": (
                round(llm_stats["tiempo_respuesta_total_segundos"] / llm_stats["llamadas_api_totales"], 4)
                if llm_stats["llamadas_api_totales"] else 0.0
            ),
            "tiempo_respuesta_total_segundos": round(llm_stats["tiempo_respuesta_total_segundos"], 2),
        },
    }

    resumen_modelo = construir_resumen_modelo(
        base_filename=base, timestamp=timestamp, num_epochs=num_epochs,
        episodes_per_epoch=episodes_per_epoch, total_episodes=total_episodes,
        tiempo_total=total_time, resultados_finales=resultados_finales,
        estructura_recompensas=estructura_recompensas, hiperparametros=hiperparametros,
        red_neuronal=red_neuronal, llm_info=llm_info, entorno=entorno,
    )
    guardar_resumen_modelo(resumen_modelo)

    meta = {"base": base, **hiperparametros}
    with open(f"modelos/meta_{base}.json", "w") as f:
        json.dump(meta, f)

    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    print(f"policy: modelos/policy_{base}.pth")
    return base

# --------------------  8. Evaluación greedy --------------------
@torch.no_grad()
def evaluate_agent_dqn(policy_path, map_size=4, is_slippery=False, max_steps=50, episodes=200,
                        map_mode="per_episode_random", seed=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = QNetCNN(in_channels=4, n_actions=4).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()

    rng = np.random.default_rng(seed)
    successes = 0
    if map_mode == "fixed":
        env, _ = make_fixed_env(map_size=map_size, is_slippery=is_slippery)

    for _ in range(episodes):
        if map_mode == "per_episode_random":
            env, _ = make_random_env(map_size=map_size, is_slippery=is_slippery, seed=int(rng.integers(0, 1e9)))
        obs, _ = env.reset()
        s_idx = int(obs)
        for _ in range(max_steps):
            desc = env.unwrapped.desc
            x = torch.from_numpy(encode_obs_from_desc(desc, s_idx)).unsqueeze(0).to(device)
            q = policy(x).cpu().numpy().squeeze(0)
            best = np.flatnonzero(q == q.max())
            a = int(np.random.choice(best))
            next_obs, r, terminated, truncated, _ = env.step(a)
            s_idx = int(next_obs)
            if terminated or truncated:
                if float(r) > 0:
                    successes += 1
                break
        if map_mode == "per_episode_random":
            env.close()

    if map_mode == "fixed":
        env.close()
    return 100.0 * successes / episodes

# --------------------  9. API key + Main --------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepinfra-api-key", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--episodes-per-epoch", type=int, default=100)
    parser.add_argument("--map-size", type=int, default=4)
    parser.add_argument("--is-slippery", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

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

    base = train_dqn_frozenlake(
        client=client,
        num_epochs=args.num_epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        map_size=args.map_size,
        is_slippery=args.is_slippery,
        max_steps_per_ep=50,
        map_mode="per_episode_random",
        seed=SEED,
        gamma=0.99,
        lr=1e-3,
        buffer_capacity=100_000,
        batch_size=64,
        start_learning_after=2_000,
        train_every=1,
        target_update_every=1_000,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=50_000,
    )

    wr = evaluate_agent_dqn(
        policy_path=f"modelos/policy_{base}.pth",
        map_size=args.map_size,
        is_slippery=args.is_slippery,
        max_steps=50,
        episodes=200,
        map_mode="per_episode_random",
        seed=123,
    )
    print(f"\nWinRate de evaluación (env reward): {wr:.2f}%")