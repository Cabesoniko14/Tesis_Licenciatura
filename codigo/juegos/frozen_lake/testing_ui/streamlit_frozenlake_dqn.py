# streamlit_eval_frozenlake.py
# UI para: (1) evaluar EXACTAMENTE evaluate_agent_dqn, (2) animar 1 episodio
# usando el render nativo de Gymnasium (rgb_array), no matplotlib.

import os
import time
import tempfile
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import streamlit as st

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
# 2) Red
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
            nn.Linear(256, 4)
        )

    def forward(self, x):
        # x: (B, C=4, H, W) con H,W en {4,8}
        B, C, H, W = x.shape
        feats = self.features(x)
        if H == 4 and W == 4:
            feats = nn.functional.interpolate(feats, size=(8, 8), mode='nearest')
        out = self.head(feats)
        return out


# -------------------------------------------------------
# 3) Envs helpers
# -------------------------------------------------------
def make_fixed_env(map_size=4, is_slippery=False, render_mode=None):
    """FrozenLake estándar de Gymnasium."""
    assert map_size in (4, 8), "map_size fijo soportado: 4 u 8"
    map_name = f"{map_size}x{map_size}"
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery, render_mode=render_mode)
    return env, map_name

def make_random_env(map_size=4, is_slippery=False, seed=None, render_mode=None):
    desc = generate_random_map(size=map_size, seed=seed)
    env = gym.make("FrozenLake-v1", desc=desc, is_slippery=is_slippery, render_mode=render_mode)
    return env, desc  # desc es lista de strings


# -------------------------------------------------------
# 4) Evaluación (greedy con desempate aleatorio)
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
            # greedy con desempate aleatorio
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
# 5) Simulación de 1 episodio — render Gym (rgb_array)
# -------------------------------------------------------
@torch.no_grad()
def run_one_episode_gym_render(
    policy_path,
    map_size=4,
    is_slippery=False,
    max_steps=100,
    map_mode="per_episode_random",
    seed=None,
    device=None,
    frame_delay=0.25,
    img_width=520,
    placeholder=None,
):
    """
    Ejecuta 1 episodio y muestra frames con el render nativo de Gymnasium (rgb_array).
    Devuelve {'win': bool, 'steps': int}.
    """
    assert map_mode in ("per_episode_random", "fixed")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar política
    policy = QNetCNN(in_channels=4, n_actions=4).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()

    rng = np.random.default_rng(seed)

    # Para animación: crear env con render_mode="rgb_array"
    if map_mode == "per_episode_random":
        env, _ = make_random_env(
            map_size=map_size, is_slippery=is_slippery,
            seed=int(rng.integers(0, 1e9)), render_mode="rgb_array"
        )
    else:
        env, _ = make_fixed_env(map_size=map_size, is_slippery=is_slippery, render_mode="rgb_array")

    obs, _ = env.reset()
    s_idx = int(obs)

    # Frame inicial
    if placeholder is not None:
        frame0 = env.render()
        placeholder.image(frame0, caption="Paso 0", width=img_width)

    # Bucle
    for step in range(1, max_steps + 1):
        desc = env.unwrapped.desc
        x = torch.from_numpy(encode_obs_from_desc(desc, s_idx)).unsqueeze(0).to(device)
        q = policy(x).cpu().numpy().squeeze(0)
        m = q.max()
        best = np.flatnonzero(q == m)
        a = int(np.random.choice(best))

        next_obs, r, terminated, truncated, _ = env.step(a)
        s_idx = int(next_obs)

        if placeholder is not None:
            frame = env.render()
            placeholder.image(frame, caption=f"Paso {step}", width=img_width)
            time.sleep(max(0.0, float(frame_delay)))

        if terminated or truncated:
            win = bool(r > 0)
            env.close()
            return {"win": win, "steps": step}

    env.close()
    return {"win": False, "steps": max_steps}


# -------------------------------------------------------
# 6) STREAMLIT UI
# -------------------------------------------------------
st.set_page_config(page_title="FrozenLake — Evaluar DQN", page_icon="🧊", layout="centered")
st.title("🧊 FrozenLake — Evaluar DQN (render original Gym)")

# --------- Panel de configuración ---------
st.subheader("Configuración")

col1, col2 = st.columns(2)
with col1:
    map_size = st.radio("Tamaño de mapa", options=[4, 8], index=0, horizontal=True)
    is_slippery = st.checkbox("is_slippery (hielo resbaloso)", value=False)
    map_mode = st.radio("Modo de mapa", options=["per_episode_random", "fixed"], index=0)

with col2:
    max_steps = st.number_input("Máx. pasos por episodio", min_value=1, max_value=1000, value=50, step=1)
    episodes = st.number_input("N episodios (evaluación)", min_value=1, max_value=10000, value=200, step=10)
    seed = st.number_input("Seed (opcional)", value=123, step=1)

st.divider()

# --------- Selección de modelo (.pth) ---------
st.subheader("Modelo (.pth)")

pth_path = None
uploaded = st.file_uploader("Sube tu policy_*.pth (opcional)", type=["pth", "pt", "bin"])

# Lista rápida de modelos locales (carpeta 'modelos')
local_paths = []
if os.path.isdir("modelos"):
    local_paths = [os.path.join("modelos", f) for f in os.listdir("modelos") if f.endswith(".pth")]
    local_paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)

option = st.selectbox(
    "O elige un .pth local (carpeta ./modelos)",
    options=["<ninguno>"] + local_paths,
    index=1 if local_paths else 0,
)

if uploaded is not None:
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
    tmpfile.write(uploaded.getbuffer())
    tmpfile.flush()
    pth_path = tmpfile.name
    st.info(f"Usando modelo subido: {os.path.basename(uploaded.name)}")
elif option != "<ninguno>":
    pth_path = option
    st.info(f"Usando modelo local: {os.path.basename(pth_path)}")
else:
    st.warning("Selecciona un .pth local o sube uno para poder evaluar/animar.")

st.divider()

# --------- Botón de evaluación (N episodios) ---------
run_eval = st.button("📊 Ejecutar evaluación (llama evaluate_agent_dqn)")

if run_eval:
    if not pth_path:
        st.error("Falta el archivo del modelo (.pth).")
    else:
        with st.spinner("Evaluando..."):
            t0 = time.perf_counter()
            try:
                wr = evaluate_agent_dqn(
                    policy_path=pth_path,
                    map_size=int(map_size),
                    is_slippery=bool(is_slippery),
                    max_steps=int(max_steps),
                    episodes=int(episodes),
                    map_mode=str(map_mode),
                    seed=int(seed),
                )
                elapsed = time.perf_counter() - t0
                st.success(f"✅ WinRate: **{wr:.2f}%**  |  episodios: {int(episodes)}  |  tiempo: {elapsed:.2f}s")
            except Exception as e:
                st.error(f"Error durante evaluate_agent_dqn: {e}")

# --------- Botón para animar 1 episodio (render Gym) ---------
st.divider()
st.subheader("🎞️ Simulación de 1 episodio (render original de Gym)")

colA, colB = st.columns([2,1])
with colA:
    anim_delay = st.slider("Retardo entre frames (seg)", 0.05, 1.0, 0.25, 0.05)
with colB:
    render_width = st.number_input("Ancho del render (px)", min_value=200, max_value=1200, value=520, step=20)

one_ep_steps = st.number_input("Máx. pasos (1 ep.)", min_value=1, max_value=500, value=int(max_steps), step=1)
run_anim = st.button("▶️ Simular 1 episodio (rgb_array)")

if run_anim:
    if not pth_path:
        st.error("Falta el archivo del modelo (.pth).")
    else:
        st.info("Se usa la MISMA política y el MISMO criterio greedy con desempate aleatorio que la evaluación.")
        ph = st.empty()  # aquí se pintan los frames (st.image)
        with st.spinner("Simulando..."):
            res = run_one_episode_gym_render(
                policy_path=pth_path,
                map_size=int(map_size),
                is_slippery=bool(is_slippery),
                max_steps=int(one_ep_steps),
                map_mode=str(map_mode),
                seed=int(seed),
                frame_delay=float(anim_delay),
                img_width=int(render_width),
                placeholder=ph,
            )
        if res["win"]:
            st.success(f"✅ ¡Objetivo alcanzado! Pasos: {res['steps']}")
        else:
            st.warning(f"⛔ No se alcanzó el objetivo. Pasos: {res['steps']}")
