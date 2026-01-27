# ======================  Q-Learning: FrozenLake (Gymnasium, Tabular) ======================
import os
import time
import json
import psutil
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# --------------------  1) Agente Q-Learning --------------------------
class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = n_states
        self.n_actions = n_actions
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

    def choose_action_train(self, s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        q = self.Q[s]
        m = q.max()
        idx = np.flatnonzero(q == m)
        return np.random.choice(idx)

    def update(self, s, a, r, s_next, done):
        max_q_next = 0.0 if done else np.max(self.Q[s_next])
        td_target = r + self.gamma * max_q_next
        self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])

# --------------------  2) Helpers de entorno --------------------------
def make_fixed_env(map_size=4, is_slippery=False, render_mode=None):
    """
    Crea el FrozenLake FIJO (mapa estándar de Gymnasium) según el tamaño.
    map_size: 4 u 8 típicamente.
    """
    assert map_size in (4, 8), "map_size fijo soportado: 4 u 8 (mapas estándar)."
    map_name = f"{map_size}x{map_size}"
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery, render_mode=render_mode)
    return env, map_name  # (env, "4x4" o "8x8")

def make_random_env(map_size=4, is_slippery=False, seed=None, render_mode=None):
    """
    Crea un FrozenLake con un mapa ALEATORIO NUEVO.
    """
    desc = generate_random_map(size=map_size, seed=seed)
    env = gym.make("FrozenLake-v1", desc=desc, is_slippery=is_slippery, render_mode=render_mode)
    return env, desc  # (env, ["SFFF", "FHFH", ...])

# --------------------  3) Entrenamiento --------------------------
def train_qlearning_frozenlake(
    num_epochs=100,
    episodes_per_epoch=100,
    map_size=4,
    is_slippery=False,
    max_steps_per_ep=200,
    map_mode="per_episode_random",   # "per_episode_random" o "fixed"
    seed=None
):
    """
    map_mode:
      - "per_episode_random": genera un mapa nuevo por episodio (mismo tamaño).
      - "fixed": usa el mapa estándar fijo ("4x4" o "8x8") en TODOS los episodios.
    """
    assert map_mode in ("per_episode_random", "fixed")
    n_states = map_size * map_size
    n_actions = 4  # 0=Left,1=Down,2=Right,3=Up
    agent = QLearningAgent(n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1)

    # Tablas
    acciones_df  = pd.DataFrame(columns=["Epoch","Episodio","Estado","Accion","EstadoSig","Reward","RewardAcum","Done"])
    computo_df   = pd.DataFrame(columns=["Epoch","Episodio","Tiempo(s)","CPU(%)","RAM(MB)","GPU_mem(MB)"])
    victorias_df = pd.DataFrame(columns=["Epoch","Exitos","Fracasos","WinRate(%)"])
    resumen_df   = pd.DataFrame(columns=["Métrica","Valor"])

    total_time = 0.0
    successes_total, fails_total = 0, 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_episodes = num_epochs * episodes_per_epoch
    base = f"frozenlake_qlearning_std_{map_mode}_size{map_size}_{'slip' if is_slippery else 'noslip'}_{timestamp}_episodes_{total_episodes}"

    os.makedirs("datos_output", exist_ok=True)
    os.makedirs("modelos", exist_ok=True)

    rng = np.random.default_rng(seed)

    # Si el modo es fijo, instanciamos UNA sola vez el env fijo
    fixed_env = None
    fixed_map_id = None
    if map_mode == "fixed":
        fixed_env, fixed_map_id = make_fixed_env(map_size=map_size, is_slippery=is_slippery, render_mode=None)

    for epoch in range(num_epochs):
        success_epoch, fail_epoch = 0, 0

        for ep in range(episodes_per_epoch):
            t0 = time.perf_counter()

            # Crear/usar entorno según modo
            if map_mode == "per_episode_random":
                env, map_meta = make_random_env(
                    map_size=map_size,
                    is_slippery=is_slippery,
                    seed=int(rng.integers(0, 1e9)),
                    render_mode=None
                )
            else:
                env, map_meta = fixed_env, fixed_map_id  # reusar el fijo

            obs, _ = env.reset()
            s = int(obs)
            done = False
            reward_acc = 0.0

            for _ in range(max_steps_per_ep):
                a = agent.choose_action_train(s)
                obs2, r, terminated, truncated, _ = env.step(a)
                s2 = int(obs2)
                done = terminated or truncated

                agent.update(s, a, r, s2, done)

                reward_acc += r
                acciones_df.loc[len(acciones_df)] = [epoch+1, ep+1, s, a, s2, r, reward_acc, done]

                s = s2
                if done:
                    if r > 0:
                        successes_total += 1; success_epoch += 1
                    else:
                        fails_total += 1; fail_epoch += 1
                    break

            elapsed = time.perf_counter() - t0
            total_time += elapsed
            computo_df.loc[len(computo_df)] = [
                epoch+1, ep+1, elapsed,
                psutil.cpu_percent(),
                psutil.Process(os.getpid()).memory_info().rss/1024/1024,
                torch.cuda.memory_allocated()/1024/1024 if torch.cuda.is_available() else 0
            ]

            if map_mode == "per_episode_random":
                env.close()  # en fijo, no cerramos para reusar

        wr = 100.0 * success_epoch / episodes_per_epoch
        victorias_df.loc[len(victorias_df)] = [epoch+1, success_epoch, fail_epoch, wr]
        print(f"=== Epoch {epoch+1}/{num_epochs} terminado | winrate={wr:.2f}% ===")

    # Cierra si fue fijo
    if map_mode == "fixed" and fixed_env is not None:
        fixed_env.close()

    # Resumen
    resumen_df.loc[len(resumen_df)] = ["Éxitos", successes_total]
    resumen_df.loc[len(resumen_df)] = ["Fracasos", fails_total]
    resumen_df.loc[len(resumen_df)] = ["Tiempo total (s)", total_time]
    resumen_df.loc[len(resumen_df)] = ["GPU usada", torch.cuda.is_available()]
    resumen_df.loc[len(resumen_df)] = ["Modo", f"{map_mode} | is_slippery={is_slippery} | size={map_size}"]
    resumen_df.loc[len(resumen_df)] = ["Episodios totales", total_episodes]

    # Guardar resultados y modelo
    acciones_df.to_csv(f"datos_output/acciones_{base}.csv", index=False)
    computo_df.to_csv(f"datos_output/computo_{base}.csv", index=False)
    victorias_df.to_csv(f"datos_output/victorias_{base}.csv", index=False)
    resumen_df.to_csv(f"datos_output/resumen_{base}.csv", index=False)
    np.save(f"modelos/qtable_{base}.npy", agent.Q)

    # Metadatos para reproducir
    meta = {
        "map_mode": map_mode,
        "map_size": map_size,
        "is_slippery": is_slippery,
        "n_states": int(n_states),
        "n_actions": int(n_actions),
        "max_steps_per_ep": max_steps_per_ep,
        "timestamp": timestamp,
        "base": base
    }
    if map_mode == "fixed":
        meta["fixed_map"] = str(fixed_map_id)  # "4x4" o "8x8"
    with open(f"modelos/meta_{base}.json", "w") as f:
        json.dump(meta, f)

    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    print(f"Q guardada: modelos/qtable_{base}.npy")
    print(f"Meta:       modelos/meta_{base}.json")
    return base

# --------------------  4) Evaluación --------------------------
def evaluate_agent(Q, episodes=100, map_size=4, is_slippery=False, max_steps=200,
                   map_mode="per_episode_random", seed=None):
    """
    - map_mode='fixed': evalúa SIEMPRE en el mapa estándar fijo de Gymnasium (4x4 u 8x8).
    - map_mode='per_episode_random': evalúa con mapa NUEVO en cada episodio.
    """
    assert map_mode in ("per_episode_random", "fixed")
    rng = np.random.default_rng(seed)
    successes = 0

    if map_mode == "fixed":
        env, _ = make_fixed_env(map_size=map_size, is_slippery=is_slippery, render_mode=None)

    for _ in range(episodes):
        if map_mode == "per_episode_random":
            env, _ = make_random_env(map_size=map_size, is_slippery=is_slippery,
                                     seed=int(rng.integers(0, 1e9)), render_mode=None)

        obs, _ = env.reset()
        s = int(obs)

        for _ in range(max_steps):
            q = Q[s]
            m = q.max()
            idx = np.flatnonzero(q == m)
            a = np.random.choice(idx)   # Greedy con desempate aleatorio
            obs2, r, terminated, truncated, _ = env.step(a)
            s = int(obs2)
            if terminated or truncated:
                if r > 0:
                    successes += 1
                break

        if map_mode == "per_episode_random":
            env.close()

    if map_mode == "fixed":
        env.close()

    return 100.0 * successes / episodes

# --------------------  Main (ejemplos de uso) --------------------------
if __name__ == "__main__":
    # Ejemplo A: ENTRENAR con mapa ALEATORIO POR EPISODIO (size=4, no resbaloso)
    base = train_qlearning_frozenlake(
        num_epochs=50,
        episodes_per_epoch=100,
        map_size=4,
        is_slippery=False,
        max_steps_per_ep=50,
        map_mode="fixed",  # <-- opción 1
        seed=None
    )

    # Cargar y evaluar en mapas aleatorios por episodio
    Q = np.load(f"modelos/qtable_{base}.npy")
    with open(f"modelos/meta_{base}.json", "r") as f:
        meta = json.load(f)

    wr_random = evaluate_agent(
        Q,
        episodes=200,
        map_size=meta["map_size"],
        is_slippery=meta["is_slippery"],
        max_steps=meta["max_steps_per_ep"],
        map_mode="per_episode_random",
        seed=None
    )
    print(f"\nWinRate (eval aleatorio por episodio): {wr_random:.2f}%")

