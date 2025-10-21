# ======================  DQN: FrozenLake (Gymnasium) — mapa aleatorio POR EPISODIO o FIJO (LLM Rewards, sin debug)  ======================
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
import re
from groq import Groq

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

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
TILE_TO_CH = {
    b'F': 0,
    b'H': 1,
    b'G': 2,
    b'S': 0,
}

def idx_to_rc(idx, size):
    return idx // size, idx % size

def encode_obs_from_desc(desc, agent_idx):
    H, W = desc.shape
    obs = np.zeros((4, H, W), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            tile = desc[r, c]
            ch = TILE_TO_CH.get(tile, 0)
            obs[ch, r, c] = 1.0
    ar, ac = idx_to_rc(agent_idx, W)
    obs[3, ar, ac] = 1.0
    return obs

# -------------------------------------------------------
# 2) Redes y Replay Buffer
# -------------------------------------------------------
class QNetCNN(nn.Module):
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
    assert map_size in (4, 8)
    map_name = f"{map_size}x{map_size}"
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery, render_mode=render_mode)
    return env, map_name

def make_random_env(map_size=4, is_slippery=False, seed=None, render_mode=None):
    desc = generate_random_map(size=map_size, seed=seed)
    env = gym.make("FrozenLake-v1", desc=desc, is_slippery=is_slippery, render_mode=render_mode)
    return env, desc

# -------------------------------------------------------
# 3.5) Recompensa Groq (API key inline, parse; sin prints)
# -------------------------------------------------------
client = Groq(api_key="")  # <-- pon tu API key aquí

def _parse_llm_reward(text: str, lo=-1.0, hi=1.0):
    m = re.search(r"[-+]?\d*\.?\d+", (text or "").strip())
    if not m:
        return None
    try:
        v = float(m.group(0))
        if not np.isfinite(v):
            return None
        return float(np.clip(v, lo, hi))
    except Exception:
        return None

def llm_reward_groq(state_idx, action, next_state_idx, terminated, truncated, env_reward, retries=1):
    """
    Devuelve: (reward_float, raw_text, elapsed_seconds)
    Si no hay respuesta válida, regresa env_reward como reward.
    """
    prompt = f"""Eres una función de recompensa para un agente DQN en FrozenLake.
Devuelve un valor en el rango [-1, 1] evaluando la acción dada.

Tu método de evaluación es exactamente el siguiente: NUNCA nada fuera de esto. 1 si llega a la meta, 0 para todo lo demás. NUNCA cambies esta regla.
Hechos:
- s (estado previo): {int(state_idx)}
- a (acción): {int(action)}  (0=Left,1=Down,2=Right,3=Up)
- s' (nuevo estado): {int(next_state_idx)}
- terminated: {bool(terminated)}
- truncated: {bool(truncated)}
- env_reward (gym): {float(env_reward)}

Tu respuesta (solo un número en [-1,1]):"""
    raw = None
    t0 = time.perf_counter()
    for _ in range(retries):
        try:
            resp = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0
            )
            raw = resp.choices[0].message.content.strip()
            val = _parse_llm_reward(raw)
            if val is not None:
                elapsed = time.perf_counter() - t0
                return val, raw, elapsed
        except Exception:
            break
    elapsed = time.perf_counter() - t0
    return float(env_reward), raw, elapsed

# -------------------------------------------------------
# 4) Entrenamiento DQN
# -------------------------------------------------------
def train_dqn_frozenlake(
    num_epochs=100,
    episodes_per_epoch=100,
    map_size=4,
    is_slippery=False,
    max_steps_per_ep=50,
    map_mode="per_episode_random",
    seed=None,
    gamma=0.99,
    lr=1e-3,
    buffer_capacity=100_000,
    batch_size=64,
    start_learning_after=1_000,
    train_every=1,
    target_update_every=1_000,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=50_000
):
    assert map_mode in ("per_episode_random", "fixed")
    set_global_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_actions = 4
    n_states = map_size * map_size

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_episodes = num_epochs * episodes_per_epoch
    base = f"frozenlake_dqn_groq_replicate_gym_rewards_{map_mode}_size{map_size}_{'slip' if is_slippery else 'noslip'}_{timestamp}_episodes_{total_episodes}"
    os.makedirs("datos_output", exist_ok=True)
    os.makedirs("modelos", exist_ok=True)

    acciones_df = pd.DataFrame(columns=["Epoch","Episodio","Paso","StateIdx","Accion","NextStateIdx","Reward","Done","Epsilon"])
    computo_df = pd.DataFrame(columns=["Epoch","Episodio","Tiempo(s)","CPU(%)","RAM(MB)","GPU_mem(MB)"])
    victorias_df = pd.DataFrame(columns=["Epoch","Exitos","Fracasos","WinRate(%)"])
    resumen_df = pd.DataFrame(columns=["Métrica","Valor"])

    rng = np.random.default_rng(seed)
    fixed_env = None
    fixed_meta = None
    if map_mode == "fixed":
        fixed_env, fixed_meta = make_fixed_env(map_size, is_slippery)

    policy = QNetCNN(4, n_actions).to(device)
    target = QNetCNN(4, n_actions).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)
    step_count = 0

    def get_epsilon(t):
        if eps_decay_steps <= 0: return eps_end
        frac = min(1.0, t / eps_decay_steps)
        return eps_start + (eps_end - eps_start) * frac

    total_time = 0.0
    successes_total, fails_total = 0, 0

    for epoch in range(num_epochs):
        success_epoch, fail_epoch = 0, 0
        for ep in range(episodes_per_epoch):
            t0 = time.perf_counter()

            if map_mode == "per_episode_random":
                env, env_meta = make_random_env(map_size, is_slippery, seed=int(rng.integers(0, 1e9)))
            else:
                env, env_meta = fixed_env, fixed_meta

            obs, _ = env.reset()
            s_idx = int(obs)
            desc = env.unwrapped.desc
            obs_tensor = encode_obs_from_desc(desc, s_idx)
            done = False
            reward_acc = 0.0

            for t in range(max_steps_per_ep):
                eps = get_epsilon(step_count)
                if np.random.rand() < eps:
                    a = np.random.randint(n_actions)
                else:
                    with torch.no_grad():
                        x = torch.from_numpy(obs_tensor).unsqueeze(0).to(device)
                        q = policy(x)
                        a = int(torch.argmax(q, dim=1).item())

                next_obs, r_env, terminated, truncated, _ = env.step(a)
                ns_idx = int(next_obs)

                # Recompensa Groq (sin prints)
                r, _, _ = llm_reward_groq(s_idx, a, ns_idx, terminated, truncated, r_env, retries=1)

                done = terminated or truncated
                next_obs_tensor = encode_obs_from_desc(desc if map_mode=="fixed" else env.unwrapped.desc, ns_idx)

                buffer.push(obs_tensor, a, r, next_obs_tensor, float(done))
                reward_acc += r
                acciones_df.loc[len(acciones_df)] = [epoch+1, ep+1, t+1, s_idx, a, ns_idx, r, done, eps]

                if len(buffer) >= start_learning_after and step_count % train_every == 0:
                    s_batch, a_batch, r_batch, s2_batch, d_batch = buffer.sample(batch_size)
                    s_batch_t = torch.from_numpy(s_batch).to(device)
                    a_batch_t = torch.from_numpy(a_batch).long().to(device)
                    r_batch_t = torch.from_numpy(r_batch).float().to(device)
                    s2_batch_t = torch.from_numpy(s2_batch).to(device)
                    d_batch_t = torch.from_numpy(d_batch).float().to(device)

                    qsa = policy(s_batch_t).gather(1, a_batch_t.view(-1,1)).squeeze(1)
                    with torch.no_grad():
                        next_actions = policy(s2_batch_t).argmax(dim=1)
                        q_next = target(s2_batch_t).gather(1, next_actions.view(-1,1)).squeeze(1)
                        target_q = r_batch_t + gamma * (1.0 - d_batch_t) * q_next

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
                    if r_env > 0:
                        successes_total += 1; success_epoch += 1
                    else:
                        fails_total += 1; fail_epoch += 1
                    break

            elapsed = time.perf_counter() - t0
            total_time += elapsed
            computo_df.loc[len(computo_df)] = [epoch+1, ep+1, elapsed,
                psutil.cpu_percent(),
                psutil.Process(os.getpid()).memory_info().rss/1024/1024,
                torch.cuda.memory_allocated()/1024/1024 if torch.cuda.is_available() else 0]

            if map_mode == "per_episode_random":
                env.close()

        wr = 100.0 * success_epoch / episodes_per_epoch
        victorias_df.loc[len(victorias_df)] = [epoch+1, success_epoch, fail_epoch, wr]
        print(f"=== Fin epoch {epoch+1} | winrate={wr:.2f}% ===")

    if map_mode == "fixed" and fixed_env is not None:
        fixed_env.close()

    resumen_df.loc[len(resumen_df)] = ["Éxitos", successes_total]
    resumen_df.loc[len(resumen_df)] = ["Fracasos", fails_total]
    resumen_df.loc[len(resumen_df)] = ["Tiempo total (s)", total_time]
    resumen_df.loc[len(resumen_df)] = ["GPU usada", torch.cuda.is_available()]
    resumen_df.loc[len(resumen_df)] = ["Modo", f"{map_mode} | is_slippery={is_slippery} | size={map_size}"]
    resumen_df.loc[len(resumen_df)] = ["Episodios totales", total_episodes]

    acciones_df.to_csv(f"datos_output/acciones_{base}.csv", index=False)
    computo_df.to_csv(f"datos_output/computo_{base}.csv", index=False)
    victorias_df.to_csv(f"datos_output/victorias_{base}.csv", index=False)
    resumen_df.to_csv(f"datos_output/resumen_{base}.csv", index=False)

    torch.save(policy.state_dict(), f"modelos/policy_{base}.pth")
    torch.save(target.state_dict(), f"modelos/target_{base}.pth")

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
    base = train_dqn_frozenlake(
        num_epochs=50,
        episodes_per_epoch=100,
        map_size=4,
        is_slippery=False,
        max_steps_per_ep=50,
        map_mode="per_episode_random",
        seed=42,
        gamma=0.99,
        lr=1e-3,
        buffer_capacity=10_000,
        batch_size=32,
        start_learning_after=64,
        train_every=1,
        target_update_every=200,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=5_000,
    )

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
