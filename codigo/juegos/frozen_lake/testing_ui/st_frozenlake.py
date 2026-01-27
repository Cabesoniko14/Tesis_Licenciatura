# streamlit_frozenlake_unified.py
# ============================================================
# FrozenLake — UI unificada (Q-table .npy + DQN policy .pth)
# - Q-table (.npy)
# - DQN policy (.pth) con 2 arquitecturas:
#     (A) CNN + interpolate a 8x8 (legacy)
#     (B) CNN + Global Average Pooling (GAP)  ✅ NUEVO
# - Mapa fijo (4x4, 8x8) o Random NxN (N=4..8)
# - Animación 1 episodio (render Gymnasium rgb_array)
# - Testing multi-size: barplot WinRate por tamaño (4..8),
#   usando 1 mapa aleatorio por tamaño (misma desc para todos los episodios)
# ============================================================

import os
import time
import glob
import random
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import streamlit as st
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

import torch
import torch.nn as nn

import matplotlib.pyplot as plt


# --------------------------- Constantes ---------------------------

ACTION_NAMES = {0: "⬅ Left", 1: "⬇ Down", 2: "➡ Right", 3: "⬆ Up"}

TILE_TO_CH = {
    b"F": 0,
    b"H": 1,
    b"G": 2,
    b"S": 0,  # Start tratado como Frozen
}

DQN_ARCH_OPTIONS = [
    "CNN Interp(8x8) (legacy)",
    "CNN GAP (GlobalAvgPool) ✅",
]


# --------------------------- Utils de IO ---------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def list_local_models(models_dir: str = "modelos") -> Dict[str, List[str]]:
    """Devuelve paths locales para .npy y .pth."""
    ensure_dir(models_dir)
    npys = sorted(glob.glob(os.path.join(models_dir, "*.npy")), key=os.path.getmtime, reverse=True)
    pths = sorted(glob.glob(os.path.join(models_dir, "*.pth")), key=os.path.getmtime, reverse=True)
    return {"npy": npys, "pth": pths}

def save_upload_temp(uploaded, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.flush()
    return tmp.name


# --------------------------- Encode obs (CNN) ---------------------------

def idx_to_rc(idx: int, size: int) -> Tuple[int, int]:
    return idx // size, idx % size

def encode_obs_from_desc(desc: np.ndarray, agent_idx: int) -> np.ndarray:
    """
    desc: ndarray (H,W) con bytes [b'S', b'F', b'H', b'G']
    agent_idx: 0..H*W-1
    -> (C=4, H, W) float32 one-hot: [FROZEN, HOLE, GOAL, AGENT]
    """
    H, W = desc.shape
    obs = np.zeros((4, H, W), dtype=np.float32)

    for r in range(H):
        for c in range(W):
            tile = desc[r, c]
            ch = TILE_TO_CH.get(tile, 0)
            obs[ch, r, c] = 1.0

    ar, ac = idx_to_rc(int(agent_idx), W)
    obs[3, ar, ac] = 1.0
    return obs


# --------------------------- DQN Nets (2 arquitecturas) ---------------------------

class QNetCNN(nn.Module):
    """
    CNN compacta. Para simplificar la cabeza, interpolamos SIEMPRE a 8x8
    (válido para tamaños 4..8).
    """
    def __init__(self, in_channels: int = 4, n_actions: int = 4):
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
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)  # (B,64,H,W)
        feats = nn.functional.interpolate(feats, size=(8, 8), mode="nearest")
        return self.head(feats)   # (B,4)


class QNetCNN_GAP(nn.Module):
    """
    CNN size-agnostic con Global Average Pooling (GAP).
    Soporta naturalmente tamaños 4..8 (y más) sin interpolación.
    """
    def __init__(self, in_channels: int = 4, n_actions: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # (B,64,H,W) -> (B,64,1,1)
        self.head = nn.Sequential(
            nn.Flatten(),          # -> (B,64)
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        feats = self.gap(feats)
        return self.head(feats)


# --------------------------- Envs ---------------------------

def make_fixed_env(map_size: int, is_slippery: bool, render_mode: Optional[str]):
    assert map_size in (4, 8), "map_size fijo soportado: 4 u 8"
    map_name = f"{map_size}x{map_size}"
    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery, render_mode=render_mode)
    return env, map_name

def make_random_env(map_size: int, is_slippery: bool, seed: Optional[int], render_mode: Optional[str]):
    desc_list = generate_random_map(size=map_size, seed=seed)  # lista de strings
    env = gym.make("FrozenLake-v1", desc=desc_list, is_slippery=is_slippery, render_mode=render_mode)
    return env, desc_list


# --------------------------- Políticas ---------------------------

def qtable_action(Q: np.ndarray, state: int) -> int:
    if Q is None:
        return 0
    if state < 0 or state >= Q.shape[0]:
        return 0
    return int(np.argmax(Q[int(state)]))

@torch.no_grad()
def dqn_action_from_state(policy: nn.Module, desc: np.ndarray, state_idx: int, device: str) -> int:
    x = torch.from_numpy(encode_obs_from_desc(desc, int(state_idx))).unsqueeze(0).to(device)
    q = policy(x).detach().cpu().numpy().squeeze(0)
    m = q.max()
    best = np.flatnonzero(q == m)
    return int(np.random.choice(best))


# --------------------------- Render helpers ---------------------------

def render_frame(env, width: int, caption: str, slot):
    frame = env.render()
    slot.image(frame, caption=caption, width=width)


# --------------------------- Evaluación genérica ---------------------------

@dataclass
class EvalConfig:
    model_kind: str                 # "qtable" o "dqn"
    qtable_path: Optional[str]
    dqn_path: Optional[str]
    dqn_arch: str                   # "interp" | "gap"
    device: str
    is_slippery: bool
    max_steps: int
    episodes: int
    map_mode: str                   # "fixed" o "per_episode_random"
    fixed_size: int                 # si fixed
    random_size: int                # si random
    seed: int

def load_qtable(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=True)

def load_dqn(path: str, device: str, dqn_arch: str) -> nn.Module:
    if dqn_arch == "gap":
        net = QNetCNN_GAP(in_channels=4, n_actions=4).to(device)
    else:
        net = QNetCNN(in_channels=4, n_actions=4).to(device)

    state = torch.load(path, map_location=device)
    net.load_state_dict(state)
    net.eval()
    return net

def evaluate_model(cfg: EvalConfig) -> float:
    """
    Evalúa WinRate% en el modo seleccionado.
    Si map_mode == per_episode_random: mapa aleatorio nuevo por episodio.
    Si map_mode == fixed: mismo mapa.
    """
    rng = np.random.default_rng(int(cfg.seed))

    Q = None
    policy = None
    if cfg.model_kind == "qtable":
        Q = load_qtable(cfg.qtable_path)
    else:
        policy = load_dqn(cfg.dqn_path, cfg.device, cfg.dqn_arch)

    successes = 0

    if cfg.map_mode == "fixed":
        env, _ = make_fixed_env(map_size=int(cfg.fixed_size), is_slippery=bool(cfg.is_slippery), render_mode=None)

    for _ in range(int(cfg.episodes)):
        if cfg.map_mode == "per_episode_random":
            env_seed = int(rng.integers(0, 1_000_000_000))
            env, _ = make_random_env(map_size=int(cfg.random_size), is_slippery=bool(cfg.is_slippery), seed=env_seed, render_mode=None)

        obs, _ = env.reset()
        s_idx = int(obs)

        for _t in range(int(cfg.max_steps)):
            if cfg.model_kind == "qtable":
                a = qtable_action(Q, s_idx)
            else:
                desc = env.unwrapped.desc  # bytes (H,W)
                a = dqn_action_from_state(policy, desc, s_idx, cfg.device)

            next_obs, r, terminated, truncated, _info = env.step(int(a))
            s_idx = int(next_obs)

            if terminated or truncated:
                if float(r) > 0:
                    successes += 1
                break

        if cfg.map_mode == "per_episode_random":
            env.close()

    if cfg.map_mode == "fixed":
        env.close()

    return 100.0 * successes / max(1, int(cfg.episodes))


# --------------------------- Animación 1 episodio ---------------------------

@torch.no_grad()
def animate_one_episode(
    model_kind: str,
    qtable_path: Optional[str],
    dqn_path: Optional[str],
    dqn_arch: str,
    map_mode: str,
    map_size: int,
    is_slippery: bool,
    max_steps: int,
    seed: int,
    device: str,
    frame_delay: float,
    img_width: int,
    slot,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))

    Q = None
    policy = None
    if model_kind == "qtable":
        Q = load_qtable(qtable_path)
    else:
        policy = load_dqn(dqn_path, device, dqn_arch)

    if map_mode == "fixed":
        env, meta = make_fixed_env(map_size=int(map_size), is_slippery=bool(is_slippery), render_mode="rgb_array")
        used_seed = int(seed)
    else:
        used_seed = int(rng.integers(0, 1_000_000_000))
        env, meta = make_random_env(map_size=int(map_size), is_slippery=bool(is_slippery), seed=used_seed, render_mode="rgb_array")

    obs, _ = env.reset(seed=used_seed)
    s_idx = int(obs)

    render_frame(
        env,
        width=img_width,
        caption=f"Paso 0 | seed={used_seed} | mapa={meta if isinstance(meta,str) else f'random {map_size}x{map_size}'}",
        slot=slot,
    )
    time.sleep(max(0.0, float(frame_delay)))

    reward_acc = 0.0

    for step in range(1, int(max_steps) + 1):
        if model_kind == "qtable":
            a = qtable_action(Q, s_idx)
        else:
            desc = env.unwrapped.desc
            a = dqn_action_from_state(policy, desc, s_idx, device)

        next_obs, r, terminated, truncated, _info = env.step(int(a))
        s_idx = int(next_obs)
        reward_acc += float(r)
        done = bool(terminated or truncated)

        render_frame(
            env,
            width=img_width,
            caption=f"Paso {step} | acción={ACTION_NAMES.get(int(a), str(a))} | reward_acc={reward_acc:.2f}",
            slot=slot,
        )
        time.sleep(max(0.0, float(frame_delay)))

        if done:
            win = bool(float(r) > 0)
            env.close()
            return {"win": win, "steps": step, "seed": used_seed}

    env.close()
    return {"win": False, "steps": int(max_steps), "seed": used_seed}


# --------------------------- Multi-size test (1 mapa por tamaño) ---------------------------

def evaluate_multisize_one_map_per_size(
    model_kind: str,
    qtable_path: Optional[str],
    dqn_path: Optional[str],
    dqn_arch: str,
    sizes: List[int],
    is_slippery: bool,
    episodes_per_size: int,
    max_steps: int,
    base_seed: int,
    device: str,
) -> Dict[int, float]:
    rng = np.random.default_rng(int(base_seed))

    Q = None
    policy = None
    if model_kind == "qtable":
        Q = load_qtable(qtable_path)
    else:
        policy = load_dqn(dqn_path, device, dqn_arch)

    results: Dict[int, float] = {}

    for N in sizes:
        seed_map = int(rng.integers(0, 1_000_000_000))
        env, _desc_list = make_random_env(map_size=int(N), is_slippery=bool(is_slippery), seed=seed_map, render_mode=None)

        wins = 0
        for ep in range(int(episodes_per_size)):
            obs, _ = env.reset(seed=seed_map + ep)
            s_idx = int(obs)

            for _t in range(int(max_steps)):
                if model_kind == "qtable":
                    a = qtable_action(Q, s_idx)
                else:
                    desc = env.unwrapped.desc
                    a = dqn_action_from_state(policy, desc, s_idx, device)

                next_obs, r, terminated, truncated, _info = env.step(int(a))
                s_idx = int(next_obs)

                if terminated or truncated:
                    if float(r) > 0:
                        wins += 1
                    break

        env.close()
        results[int(N)] = 100.0 * wins / max(1, int(episodes_per_size))

    return results


def plot_bar_results(results: Dict[int, float], title: str):
    xs = sorted(results.keys())
    ys = [results[k] for k in xs]
    fig, ax = plt.subplots()
    ax.bar([str(x) for x in xs], ys)
    ax.set_xlabel("Tamaño del mapa (NxN)")
    ax.set_ylabel("WinRate (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.25)
    return fig


# ===========================
# STREAMLIT APP
# ===========================

st.set_page_config(page_title="FrozenLake — Q-table & DQN Tester", page_icon="🧊", layout="wide")
st.title("🧊 FrozenLake — Tester unificado (Q-table .npy + DQN .pth)")

ss = st.session_state
for k, v in {
    "_qtable_path": None,
    "_dqn_path": None,
    "_qtable_loaded": False,
    "_dqn_loaded": False,
    "_dqn_arch": "interp",  # default
}.items():
    if k not in ss:
        ss[k] = v

device = "cuda" if torch.cuda.is_available() else "cpu"
st.caption(f"Device detectado: **{device}**")

models = list_local_models("modelos")

with st.sidebar:
    st.header("⚙️ Modelo")

    model_kind_ui = st.radio("Tipo de modelo", options=["Q-table (.npy)", "DQN policy (.pth)"], index=0)
    kind = "qtable" if model_kind_ui.startswith("Q-table") else "dqn"

    # ✅ NUEVO: selector de arquitectura DQN
    if kind == "dqn":
        dqn_arch_ui = st.selectbox("Arquitectura DQN", options=DQN_ARCH_OPTIONS, index=0)
        ss._dqn_arch = "gap" if "GAP" in dqn_arch_ui else "interp"

        st.caption(
            "Tip: si tu archivo se llama algo como `..._gap_...pth`, elige **CNN GAP**. "
            "Si es de tu versión vieja, elige **Interp(8x8)**."
        )

    st.divider()
    st.subheader("📦 Cargar desde ./modelos")

    if kind == "qtable":
        local_npys = models["npy"]
        opt = st.selectbox("Q-table local", options=["<ninguno>"] + local_npys, index=0)
        up = st.file_uploader("O sube un .npy", type=["npy"])
        if st.button("Cargar Q-table"):
            if up is not None:
                path = save_upload_temp(up, ".npy")
                ss._qtable_path = path
                ss._qtable_loaded = True
                st.success(f"Cargado upload: {up.name}")
            elif opt != "<ninguno>":
                ss._qtable_path = opt
                ss._qtable_loaded = True
                st.success(f"Cargado local: {os.path.basename(opt)}")
            else:
                st.warning("Selecciona un archivo .npy (local o upload).")
    else:
        local_pths = models["pth"]
        opt = st.selectbox("Policy local", options=["<ninguno>"] + local_pths, index=0)
        up = st.file_uploader("O sube un .pth", type=["pth", "pt", "bin"])
        if st.button("Cargar policy DQN"):
            if up is not None:
                path = save_upload_temp(up, ".pth")
                ss._dqn_path = path
                ss._dqn_loaded = True
                st.success(f"Cargado upload: {up.name}")
            elif opt != "<ninguno>":
                ss._dqn_path = opt
                ss._dqn_loaded = True
                st.success(f"Cargado local: {os.path.basename(opt)}")
            else:
                st.warning("Selecciona un archivo .pth (local o upload).")

    st.divider()
    st.header("🗺️ Entorno")

    is_slippery = st.checkbox("is_slippery (hielo resbaloso)", value=False)

    map_mode_ui = st.radio("Modo de mapa", options=["Random por episodio", "Fijo (4x4 o 8x8)"], index=0)
    map_mode = "per_episode_random" if map_mode_ui.startswith("Random") else "fixed"

    if map_mode == "fixed":
        map_size = st.radio("Tamaño fijo", options=[4, 8], index=0, horizontal=True)
        random_size = 4
    else:
        random_size = st.slider("Tamaño Random NxN", min_value=4, max_value=8, value=8, step=1)
        map_size = int(random_size)

    st.divider()
    st.header("⏱️ Parámetros")
    max_steps = st.number_input("Máx. pasos por episodio", min_value=1, max_value=2000, value=100, step=1)
    episodes = st.number_input("Episodios (evaluación)", min_value=1, max_value=20000, value=200, step=10)
    seed = st.number_input("Seed base", min_value=0, max_value=2_000_000_000, value=123, step=1)


colL, colR = st.columns([7, 5])

with colL:
    st.subheader("🎞️ Animar 1 episodio")

    anim_delay = st.slider("Retardo entre frames (seg)", 0.0, 1.0, 0.25, 0.05)
    render_width = st.slider("Ancho render (px)", 240, 1000, 520, 20)
    one_ep_steps = st.number_input("Máx. pasos (1 ep.)", min_value=1, max_value=2000, value=int(max_steps), step=1)

    can_run = (ss._qtable_loaded if kind == "qtable" else ss._dqn_loaded)
    if not can_run:
        st.warning("Carga un modelo en la barra lateral para poder animar y evaluar.")
    else:
        ph = st.empty()
        if st.button("▶️ Simular 1 episodio (render Gym)"):
            res = animate_one_episode(
                model_kind=kind,
                qtable_path=ss._qtable_path,
                dqn_path=ss._dqn_path,
                dqn_arch=ss._dqn_arch if kind == "dqn" else "interp",
                map_mode=map_mode,
                map_size=int(map_size),
                is_slippery=bool(is_slippery),
                max_steps=int(one_ep_steps),
                seed=int(seed),
                device=device,
                frame_delay=float(anim_delay),
                img_width=int(render_width),
                slot=ph,
            )
            if res["win"]:
                st.success(f"✅ ¡Objetivo alcanzado! pasos={res['steps']} | seed={res['seed']}")
            else:
                st.warning(f"⛔ No se alcanzó objetivo. pasos={res['steps']} | seed={res['seed']}")

with colR:
    st.subheader("📊 Evaluación")

    can_eval = (ss._qtable_loaded if kind == "qtable" else ss._dqn_loaded)
    if not can_eval:
        st.info("Carga un modelo para habilitar evaluación.")
    else:
        if st.button("📈 Ejecutar evaluación (WinRate)"):
            cfg = EvalConfig(
                model_kind=kind,
                qtable_path=ss._qtable_path,
                dqn_path=ss._dqn_path,
                dqn_arch=ss._dqn_arch if kind == "dqn" else "interp",
                device=device,
                is_slippery=bool(is_slippery),
                max_steps=int(max_steps),
                episodes=int(episodes),
                map_mode=map_mode,
                fixed_size=int(map_size),
                random_size=int(random_size),
                seed=int(seed),
            )
            with st.spinner("Evaluando..."):
                t0 = time.perf_counter()
                wr = evaluate_model(cfg)
                elapsed = time.perf_counter() - t0
            st.success(f"✅ WinRate: **{wr:.2f}%** | episodios={int(episodes)} | tiempo={elapsed:.2f}s")

    st.divider()
    st.subheader("🧪 Testing multi-tamaño (4..8)")

    st.caption("Genera **1 mapa aleatorio por tamaño** y evalúa varios episodios en ese mapa. (Barplot WinRate por tamaño)")
    eps_per_size = st.number_input("Episodios por tamaño", min_value=1, max_value=5000, value=200, step=10)
    max_steps_ms = st.number_input("Máx. pasos (multi-size)", min_value=1, max_value=2000, value=200, step=10)
    if st.button("📊 Correr multi-size (4..8) y graficar"):
        if not can_eval:
            st.error("Falta cargar modelo.")
        else:
            sizes = [4, 5, 6, 7, 8]
            with st.spinner("Corriendo multi-size..."):
                t0 = time.perf_counter()
                res = evaluate_multisize_one_map_per_size(
                    model_kind=kind,
                    qtable_path=ss._qtable_path,
                    dqn_path=ss._dqn_path,
                    dqn_arch=ss._dqn_arch if kind == "dqn" else "interp",
                    sizes=sizes,
                    is_slippery=bool(is_slippery),
                    episodes_per_size=int(eps_per_size),
                    max_steps=int(max_steps_ms),
                    base_seed=int(seed),
                    device=device,
                )
                elapsed = time.perf_counter() - t0

            st.write("Resultados (WinRate%):", res)
            fig = plot_bar_results(res, title=f"WinRate por tamaño (1 mapa random por tamaño) | modelo={kind} | arch={ss._dqn_arch} | slippery={is_slippery}")
            st.pyplot(fig)
            st.success(f"Listo. Tiempo: {elapsed:.2f}s")

st.info(
    "Tip: si tu Q-table fue entrenada en 4x4, lo normal es que no generalice a tamaños mayores "
    "(porque el espacio de estados cambia). Para DQN, aquí la entrada es el grid one-hot + canal del agente. "
    "Ahora puedes evaluar dos arquitecturas: Interp(8x8) y GAP."
)
