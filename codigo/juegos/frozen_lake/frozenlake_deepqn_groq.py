# ======================  DQN: FrozenLake (Gymnasium) — mapa aleatorio POR EPISODIO o FIJO  ======================
import os
import time
import json
import psutil
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import random

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

import re  # para parsear el número devuelto por el LLM
from groq import Groq  # o deja OpenAI si prefieres; aquí uso Groq porque ya lo tienes

# Cliente LLM (toma la API key de variable de entorno)
llm_client = Groq(api_key)

def _parse_first_float(text: str, lo=-1.0, hi=1.0):
    """
    Extrae el primer número de un string y lo recorta a [lo, hi].
    Devuelve None si no hay número válido.
    """
    if not text:
        return None
    m = re.search(r"[-+]?\d*\.?\d+", text.strip())
    if not m:
        return None
    try:
        v = float(m.group(0))
        if not np.isfinite(v):
            return None
        return float(np.clip(v, lo, hi))
    except Exception:
        return None


# -------------------------------------------------------
# 0) Utils reproducibilidad (opcional)
# -------------------------------------------------------
def set_global_seeds(seed=None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------
# 1) Observación enriquecida (mapa + agente) -> tensor
# -------------------------------------------------------
# Capas: [FROZEN, HOLE, GOAL, AGENT]
TILE_TO_CH = {
    b'F': 0,
    b'H': 1,
    b'G': 2,
    b'S': 0,  # Start tratado como Frozen
}

def idx_to_rc(idx, size):
    return idx // size, idx % size

def encode_obs_from_desc(desc, agent_idx):
    """
    desc: ndarray (H,W) con bytes [b'S', b'F', b'H', b'G']
    agent_idx: índice entero del estado (0..H*W-1)
    -> tensor numpy de forma (C=4, H, W) en float32
    """
    H, W = desc.shape
    C = 4
    obs = np.zeros((C, H, W), dtype=np.float32)

    for r in range(H):
        for c in range(W):
            tile = desc[r, c]
            ch = TILE_TO_CH.get(tile, 0)
            obs[ch, r, c] = 1.0

    ar, ac = idx_to_rc(agent_idx, W)
    obs[3, ar, ac] = 1.0  # canal del agente
    return obs


# -------------------------------------------------------
# 2) Redes y Replay Buffer
# -------------------------------------------------------
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
            nn.Linear(64 * 8 * 8, 256),  # se ajusta a 8x8; si es 4x4 reacomodamos en forward
            nn.ReLU(inplace=True),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        # x: (B, C=4, H, W) con H,W en {4,8}
        B, C, H, W = x.shape
        feats = self.features(x)
        # Si H,W=4, adaptamos con AdaptiveAvgPool a 8x8 para reutilizar la misma head
        if H == 4 and W == 4:
            feats = nn.functional.interpolate(feats, size=(8, 8), mode='nearest')
        out = self.head(feats)
        return out


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


# -------------------------------------------------------
# 3) Envs helpers
# -------------------------------------------------------
def make_fixed_env(map_size=4, is_slippery=False, render_mode=None):
    """
    FrozenLake estándar de Gymnasium:
    map_size=4 -> '4x4', map_size=8 -> '8x8'
    """
    assert map_size in (4, 8), "map_size fijo soportado: 4 u 8"
    map_name = f"{map_size}x{map_size}"
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery, render_mode=render_mode)
    return env, map_name

def make_random_env(map_size=4, is_slippery=False, seed=None, render_mode=None):
    desc = generate_random_map(size=map_size, seed=seed)
    env = gym.make("FrozenLake-v1", desc=desc, is_slippery=is_slippery, render_mode=render_mode)
    return env, desc  # desc es lista de strings


# -------------------------------------------------------
# 3.5) Recompensa PROXY (idéntica a Gym, sin usar Gym)
# -------------------------------------------------------
def reward_proxy(terminated, truncated, env_reward_last):
    """
    MISMO CONTRATO QUE ANTES:
    - En pasos NO terminales -> 0.0
    - En paso TERMINAL -> pedir al LLM que devuelva ESTRICTAMENTE 0 o 1.
      Si el LLM falla, usamos la regla del entorno:
        expected = 1.0 si (terminated True y env_reward_last == 1.0), si no 0.0
    Además: imprime WARNING si el valor del LLM no coincide con expected.
    """
    done = bool(terminated) or bool(truncated)
    if not done:
        return 0.0  # idéntico al estándar

    # Regla "ground truth" del entorno
    expected = 1.0 if (bool(terminated) and float(env_reward_last) == 1.0) else 0.0

    # Prompt ultra explícito (sin contexto, solo la regla)
    prompt = (
        "Responde SOLO con un número: 0 o 1. SIN texto extra.\n"
        "Regla EXACTA:\n"
        "- Devuelve 1 si terminated es true Y env_reward_last es 1.\n"
        "- En todos los demás casos (incluye truncated true o env_reward_last 0), devuelve 0.\n\n"
        f"terminated={bool(terminated)}\n"
        f"truncated={bool(truncated)}\n"
        f"env_reward_last={float(env_reward_last)}\n"
    )

    try:
        resp = llm_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content.strip()
        val = _parse_first_float(raw, lo=0.0, hi=1.0)

        if val is None:
            print(f"[LLM WARN] No se pudo parsear número del LLM. raw={raw!r} -> fallback expected={expected}")
            return expected

        # Si parseó, comparamos y avisamos si no coincide
        if float(val) != expected:
            print(f"[LLM WARN] Mismatch recompensa final. expected={expected} "
                  f"(term={bool(terminated)} trunc={bool(truncated)} env_r={float(env_reward_last)}) "
                  f"vs llm={float(val)} | raw={raw!r}")

        # Usamos la recompensa del LLM (tal como pediste)
        return float(val)

    except Exception as e:
        print(f"[LLM WARN] Excepción consultando LLM: {e} -> fallback expected={expected}")
        return expected




# -------------------------------------------------------
# 4) Entrenamiento DQN (estructura de guardado estilo TTT)
# -------------------------------------------------------
def train_dqn_frozenlake(
    num_epochs=100,
    episodes_per_epoch=100,
    map_size=4,
    is_slippery=False,
    max_steps_per_ep=100,
    map_mode="per_episode_random",   # "per_episode_random" o "fixed"
    seed=None,
    # Hiperparámetros DQN (compactos)
    gamma=0.99,
    lr=1e-3,
    buffer_capacity=100_000,
    batch_size=64,
    start_learning_after=1_000,      # warmup
    train_every=1,
    target_update_every=1_000,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=50_000
):
    """
    Devuelve el 'base' para cargar resultados/modelos luego.
    """
    assert map_mode in ("per_episode_random", "fixed")
    set_global_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_actions = 4
    n_states = map_size * map_size  # solo meta (para reportes)

    # Directorios y base
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_episodes = num_epochs * episodes_per_epoch
    base = f"frozenlake_dqn_groq_replicate_gym_reward_{map_mode}_size{map_size}_{'slip' if is_slippery else 'noslip'}_{timestamp}_episodes_{total_episodes}"
    os.makedirs("datos_output", exist_ok=True)
    os.makedirs("modelos", exist_ok=True)

    # Tablas
    acciones_df  = pd.DataFrame(columns=["Epoch","Episodio","Paso","StateIdx","Accion","NextStateIdx","Reward","Done","Epsilon"])
    computo_df   = pd.DataFrame(columns=["Epoch","Episodio","Tiempo(s)","CPU(%)","RAM(MB)","GPU_mem(MB)"])
    victorias_df = pd.DataFrame(columns=["Epoch","Exitos","Fracasos","WinRate(%)"])
    resumen_df   = pd.DataFrame(columns=["Métrica","Valor"])

    # Envs
    rng = np.random.default_rng(seed)
    fixed_env = None
    fixed_meta = None
    if map_mode == "fixed":
        fixed_env, fixed_meta = make_fixed_env(map_size=map_size, is_slippery=is_slippery, render_mode=None)

    # Redes + buffer + optim
    policy = QNetCNN(in_channels=4, n_actions=n_actions).to(device)
    target = QNetCNN(in_channels=4, n_actions=n_actions).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_capacity)
    step_count = 0

    # Epsilon schedule lineal
    def get_epsilon(t):
        if eps_decay_steps <= 0:
            return eps_end
        frac = min(1.0, t / eps_decay_steps)
        return eps_start + (eps_end - eps_start) * frac

    total_time = 0.0
    successes_total, fails_total = 0, 0

    # bucle de entrenamiento
    for epoch in range(num_epochs):
        success_epoch, fail_epoch = 0, 0

        for ep in range(episodes_per_epoch):
            t0 = time.perf_counter()

            # entorno por episodio
            if map_mode == "per_episode_random":
                env, env_meta = make_random_env(
                    map_size=map_size, is_slippery=is_slippery,
                    seed=int(rng.integers(0, 1e9)), render_mode=None
                )
            else:
                env, env_meta = fixed_env, fixed_meta

            obs, _ = env.reset()
            s_idx = int(obs)
            desc = env.unwrapped.desc  # bytes (H,W)
            obs_tensor = encode_obs_from_desc(desc, s_idx)  # (C,H,W)
            done = False
            reward_acc = 0.0

            for t in range(max_steps_per_ep):
                # epsilon-greedy
                eps = get_epsilon(step_count)
                if np.random.rand() < eps:
                    a = np.random.randint(n_actions)
                else:
                    with torch.no_grad():
                        x = torch.from_numpy(obs_tensor).unsqueeze(0).to(device)  # (1,C,H,W)
                        q = policy(x)  # (1,4)
                        a = int(torch.argmax(q, dim=1).item())

                next_obs, r_env, terminated, truncated, _ = env.step(a)
                ns_idx = int(next_obs)
                done = terminated or truncated
                next_obs_tensor = encode_obs_from_desc(desc if map_mode=="fixed" else env.unwrapped.desc, ns_idx)

                # ===== ÚNICO CAMBIO: usar la recompensa proxy en lugar de r_env =====
                r = reward_proxy(terminated, truncated, r_env)
                # =====================================================================

                # push to buffer
                buffer.push(obs_tensor, a, r, next_obs_tensor, float(done))

                # tracking
                reward_acc += r
                acciones_df.loc[len(acciones_df)] = [epoch+1, ep+1, t+1, s_idx, a, ns_idx, r, done, eps]

                # aprender
                if len(buffer) >= start_learning_after and step_count % train_every == 0:
                    s_batch, a_batch, r_batch, s2_batch, d_batch = buffer.sample(batch_size)

                    s_batch_t  = torch.from_numpy(s_batch).to(device)              # (B,C,H,W)
                    a_batch_t  = torch.from_numpy(a_batch).long().to(device)       # (B,)
                    r_batch_t  = torch.from_numpy(r_batch).float().to(device)      # (B,)
                    s2_batch_t = torch.from_numpy(s2_batch).to(device)             # (B,C,H,W)
                    d_batch_t  = torch.from_numpy(d_batch).float().to(device)      # (B,)

                    # Q(s,a)
                    qsa = policy(s_batch_t).gather(1, a_batch_t.view(-1,1)).squeeze(1)  # (B,)

                    # Double DQN: argmax(policy(s2)) -> eval en target
                    with torch.no_grad():
                        next_actions = policy(s2_batch_t).argmax(dim=1)  # (B,)
                        q_next = target(s2_batch_t).gather(1, next_actions.view(-1,1)).squeeze(1)
                        target_q = r_batch_t + gamma * (1.0 - d_batch_t) * q_next

                    loss = nn.MSELoss()(qsa, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                    optimizer.step()

                # target update
                if step_count % target_update_every == 0:
                    target.load_state_dict(policy.state_dict())

                step_count += 1
                obs_tensor = next_obs_tensor
                s_idx = ns_idx

                if done:
                    if r_env > 0:
                        successes_total += 1; success_epoch += 1
                    else:
                        fails_total += 1; fail_epoch += 1
                    break

            # cómputo episodio
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            computo_df.loc[len(computo_df)] = [
                epoch+1, ep+1, elapsed,
                psutil.cpu_percent(),
                psutil.Process(os.getpid()).memory_info().rss/1024/1024,
                torch.cuda.memory_allocated()/1024/1024 if torch.cuda.is_available() else 0
            ]

            if map_mode == "per_episode_random":
                env.close()

        # fin de epoch
        wr = 100.0 * success_epoch / episodes_per_epoch
        victorias_df.loc[len(victorias_df)] = [epoch+1, success_epoch, fail_epoch, wr]
        print(f"=== Epoch {epoch+1}/{num_epochs} terminado | winrate={wr:.2f}% ===")

    # cierre si fijo
    if map_mode == "fixed" and fixed_env is not None:
        fixed_env.close()

    # Resumen global y guardado
    resumen_df.loc[len(resumen_df)] = ["Éxitos", successes_total]
    resumen_df.loc[len(resumen_df)] = ["Fracasos", fails_total]
    resumen_df.loc[len(resumen_df)] = ["Tiempo total (s)", total_time]
    resumen_df.loc[len(resumen_df)] = ["GPU usada", torch.cuda.is_available()]
    resumen_df.loc[len(resumen_df)] = ["Modo", f"{map_mode} | is_slippery={is_slippery} | size={map_size}"]
    resumen_df.loc[len(resumen_df)] = ["Episodios totales", total_episodes]

    # Guardados
    acciones_df.to_csv(f"datos_output/acciones_{base}.csv", index=False)
    computo_df.to_csv(f"datos_output/computo_{base}.csv", index=False)
    victorias_df.to_csv(f"datos_output/victorias_{base}.csv", index=False)
    resumen_df.to_csv(f"datos_output/resumen_{base}.csv", index=False)

    torch.save(policy.state_dict(), f"modelos/policy_{base}.pth")
    torch.save(target.state_dict(), f"modelos/target_{base}.pth")

    meta = {
        "map_mode": map_mode,
        "map_size": map_size,
        "is_slippery": is_slippery,
        "n_states": int(n_states),
        "n_actions": int(n_actions),
        "max_steps_per_ep": max_steps_per_ep,
        "timestamp": timestamp,
        "base": base,
        "gamma": gamma,
        "lr": lr,
        "buffer_capacity": buffer_capacity,
        "batch_size": batch_size,
        "start_learning_after": start_learning_after,
        "train_every": train_every,
        "target_update_every": target_update_every,
        "eps_start": eps_start,
        "eps_end": eps_end,
        "eps_decay_steps": eps_decay_steps,
    }
    with open(f"modelos/meta_{base}.json", "w") as f:
        json.dump(meta, f)

    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    print(f"policy: modelos/policy_{base}.pth")
    print(f"target: modelos/target_{base}.pth")
    print(f"meta:   modelos/meta_{base}.json")
    return base


# -------------------------------------------------------
# 5) Evaluación (greedy con desempate aleatorio)
# -------------------------------------------------------
@torch.no_grad()
def evaluate_agent_dqn(
    policy_path,
    map_size=4,
    is_slippery=False,
    max_steps=100,
    episodes=200,
    map_mode="per_episode_random",
    seed=None,
    device=None
):
    assert map_mode in ("per_episode_random", "fixed")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Carga red
    policy = QNetCNN(in_channels=4, n_actions=4).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()

    rng = np.random.default_rng(seed)
    successes = 0

    if map_mode == "fixed":
        env, _ = make_fixed_env(map_size=map_size, is_slippery=is_slippery, render_mode=None)

    for _ in range(episodes):
        if map_mode == "per_episode_random":
            env, _ = make_random_env(map_size=map_size, is_slippery=is_slippery,
                                     seed=int(rng.integers(0, 1e9)), render_mode=None)

        obs, _ = env.reset()
        s_idx = int(obs)
        for _ in range(max_steps):
            desc = env.unwrapped.desc
            x = torch.from_numpy(encode_obs_from_desc(desc, s_idx)).unsqueeze(0).to(device)
            q = policy(x).cpu().numpy().squeeze(0)
            # greedy tie-break aleatorio
            m = q.max()
            best = np.flatnonzero(q == m)
            a = int(np.random.choice(best))
            next_obs, r, terminated, truncated, _ = env.step(a)
            s_idx = int(next_obs)
            if terminated or truncated:
                if r > 0: successes += 1
                break

        if map_mode == "per_episode_random":
            env.close()

    if map_mode == "fixed":
        env.close()
    return 100.0 * successes / episodes


# -------------------------------------------------------
# 6) Main de ejemplo
# -------------------------------------------------------
if __name__ == "__main__":
    # Ejemplo A: entrenar con mapas ALEATORIOS por episodio (no resbaloso, 4x4)
    base = train_dqn_frozenlake(
        num_epochs=50,
        episodes_per_epoch=100,
        map_size=4,
        is_slippery=False,
        max_steps_per_ep=50,
        map_mode="per_episode_random",
        seed=42,
        # Hiperparámetros (compactos y razonables)
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

    # Evaluación rápida (mismas condiciones de entrenamiento)
    wr = evaluate_agent_dqn(
        policy_path=f"modelos/policy_{base}.pth",
        map_size=4,
        is_slippery=False,
        max_steps=50,
        episodes=200,
        map_mode="per_episode_random",
        seed=123
    )
    print(f"\nWinRate de evaluación (aleatorio por episodio): {wr:.2f}%")
