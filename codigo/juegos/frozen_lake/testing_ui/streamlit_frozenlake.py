# testing_ui/streamlit_frozenlake.py
# ============================================================
# FrozenLake — Visualizador de Q-tables (Gymnasium + Streamlit)
# - Carga Q-table (.npy) desde ./modelos
# - Mapa 4x4 / 8x8 o Random NxN (con opción de nuevo mapa en cada reset)
# - Solo ver al MODELO jugar (sin control manual)
# - Render animado con delay configurables entre acciones
# ============================================================

import os
import time
import glob
import random
import numpy as np
import streamlit as st
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# --------------------------- Utilidades ---------------------------

ACTION_NAMES = {0: "⬅ Left", 1: "⬇ Down", 2: "➡ Right", 3: "⬆ Up"}

def list_qtables(models_dir: str = "modelos"):
    os.makedirs(models_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(models_dir, "*.npy")))
    labels = [os.path.basename(f) for f in files]
    return files, labels

def make_env(map_kind: str, is_slippery: bool, n_random: int, seed: int | None):
    """Crea el entorno FrozenLake con render RGB."""
    if map_kind == "random":
        rand_map = generate_random_map(size=n_random, seed=seed)
        env = gym.make(
            "FrozenLake-v1",
            desc=rand_map,
            is_slippery=is_slippery,
            render_mode="rgb_array",
        )
    else:
        env = gym.make(
            "FrozenLake-v1",
            map_name=map_kind,
            is_slippery=is_slippery,
            render_mode="rgb_array",
        )
    return env

def policy_action(Q: np.ndarray, state: int) -> int:
    if state < Q.shape[0]:
        return int(np.argmax(Q[state]))
    return 0

def render_frame(env, width: int = 480, caption: str | None = None, slot=None):
    frame = env.render()
    (slot or st).image(frame, caption=caption, width=width)

# --------------------------- UI ---------------------------

st.set_page_config(page_title="FrozenLake — Q-table Player", layout="wide")
st.title("🧊 FrozenLake — Probar Q-tables (modo espectador)")

# Estado de sesión
ss = st.session_state
defaults = {
    "_env": None, "_Q": None, "_state": 0, "_done": False,
    "_reward_acc": 0.0, "_info": {}, "_playing": False,
    "_seed_used": None
}
for k, v in defaults.items():
    if k not in ss: ss[k] = v

with st.sidebar:
    st.header("⚙️ Configuración")

    # Modelos
    files, labels = list_qtables("modelos")
    if labels:
        sel_idx = st.selectbox("Modelo (Q-table)", range(len(labels)), format_func=lambda i: labels[i])
        model_file = files[sel_idx]
    else:
        model_file = None
        st.info("No se encontraron Q-tables en ./modelos/*.npy")

    st.divider()

    # Mapa
    map_choice = st.selectbox("Mapa",
        ["4x4 (determinístico)","8x8 (determinístico)","Random NxN"])
    if "4x4" in map_choice:
        map_kind = "4x4"; allow_random = False
    elif "8x8" in map_choice:
        map_kind = "8x8"; allow_random = False
    else:
        map_kind = "random"; allow_random = True

    is_slippery = st.checkbox("is_slippery (hielo resbaloso)", value=False)

    n_random = st.number_input("Tamaño del mapa (si Random)", value=8, min_value=4, max_value=32, step=1, disabled=not allow_random)
    randomize_each_reset = st.checkbox("Nuevo mapa en cada reset", value=True, disabled=not allow_random)
    seed_val = st.number_input("Seed base (si Random)", value=42, min_value=0, max_value=10_000, step=1, disabled=not allow_random)

    st.caption("💡 Si activas *Nuevo mapa en cada reset*, el seed usado se mostrará abajo del tablero.")

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button("📦 Cargar modelo"):
        if model_file and os.path.exists(model_file):
            try:
                ss._Q = np.load(model_file)
                st.success(f"Modelo cargado: {os.path.basename(model_file)} | Q shape={ss._Q.shape}")
            except Exception as e:
                st.error(f"Error al cargar Q-table: {e}")
        else:
            st.warning("Selecciona un archivo válido en ./modelos")

    if c2.button("🔁 Reset env"):
        ss._env = None  # forzará recreación abajo

# Crear / recrear entorno si no existe
reseed = None
if ss._env is None:
    if map_kind == "random":
        if randomize_each_reset:
            reseed = random.randint(0, 10_000_000)
        else:
            reseed = int(seed_val)
    ss._env = make_env(map_kind, is_slippery, int(n_random), reseed)
    ss._state, _ = ss._env.reset(seed=reseed)
    ss._seed_used = reseed
    ss._done = False
    ss._reward_acc = 0.0
    ss._info = {}

# Info del entorno
desc = f"Mapa: {'Random '+str(n_random) if map_kind=='random' else map_kind} | is_slippery: {is_slippery} | Estado inicial: {ss._state}"
if map_kind == "random":
    desc += f" | seed usado: {ss._seed_used}"
st.caption(desc)

left, right = st.columns([7,5])

with left:
    st.subheader("🎥 Entorno (modo espectador)")
    render_width = st.slider("Ancho de render (px)", 260, 900, 520, step=20)
    render_slot = st.empty()
    render_frame(ss._env, width=render_width, caption="Vista del lago", slot=render_slot)

    st.markdown("#### ▶️ Reproducir episodio con la política")
    col = st.columns([1,1,2,2])
    max_steps = col[0].number_input("Máx. pasos", value=200, min_value=1, max_value=1000)
    delay_ms  = col[1].slider("Delay por acción (ms)", 0, 1000, 200, step=10)
    play_btn  = col[2].button("Correr política (animado)")
    stop_btn  = col[3].button("Detener")

    if play_btn:
        if ss._Q is None:
            st.warning("Primero carga un modelo (Q-table).")
        else:
            ss._playing = True
            # Nuevo mapa si corresponde
            if map_kind == "random" and randomize_each_reset:
                reseed = random.randint(0, 10_000_000)
                ss._env = make_env(map_kind, is_slippery, int(n_random), reseed)
                ss._seed_used = reseed
                ss._state, _ = ss._env.reset(seed=reseed)
            else:
                ss._state, _ = ss._env.reset()
            ss._done = False
            ss._reward_acc = 0.0

            # Bucle animado
            for t in range(int(max_steps)):
                if stop_btn:
                    ss._playing = False
                    break
                act = policy_action(ss._Q, ss._state)
                obs, rew, terminated, truncated, info = ss._env.step(act)
                ss._state = obs
                ss._reward_acc += float(rew)
                ss._done = bool(terminated or truncated)
                ss._info = info

                # Render y delay explícito
                render_frame(ss._env, width=render_width, caption=f"Paso {t+1} | Reward acumulado: {ss._reward_acc:.2f}", slot=render_slot)
                if ss._done:
                    break
                time.sleep(max(0.0, delay_ms/1000.0))

with right:
    st.subheader("📊 Estado")
    st.markdown(f"**Estado actual:** {ss._state}")
    st.markdown(f"**Reward acumulado:** {ss._reward_acc:.2f}")
    st.markdown(f"**Done:** {ss._done}")

    st.markdown("#### Q-values del estado actual")
    if ss._Q is not None and ss._state < ss._Q.shape[0]:
        q_row = ss._Q[ss._state]
    else:
        q_row = np.zeros(4, dtype=float)

    st.dataframe(
        {"Acción": [ACTION_NAMES[i] for i in range(4)],
         "Q": [float(q_row[i]) for i in range(4)]},
        use_container_width=True, hide_index=True
    )

st.info("Tip: si quieres un nivel distinto **Random NxN + Nuevo mapa en cada reset** y pulsa **Reset env**.")
