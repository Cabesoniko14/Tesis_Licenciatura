# streamlit_connect4_play.py
# Juega Conecta-4 contra tu modelo DQN (greedy con desempate aleatorio)

import os
import time
import tempfile
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt


# ---------- Utilidad RNG ----------
def set_global_seeds(seed=None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------- Entorno Conecta-4 ----------
class Connect4:
    """
    Tablero 6x7. Jugador actual: 1 (rojo) o -1 (amarillo).
    Acciones: 0..6 (columna). Si columna llena -> jugada inválida.
    Episodio termina si hay 4 en línea o tablero lleno.
    """
    H, W, N_TO_WIN = 6, 7, 4

    def __init__(self, starter_player: int = 1):
        self.board = np.zeros((self.H, self.W), dtype=np.int8)
        self.current_player = starter_player  # 1 (humano por defecto) ó -1
        self.winner = 0  # 1 ó -1 si hay ganador, 0 si no
        self.done = False

    def reset(self, starter_player: int = 1):
        self.board[:] = 0
        self.current_player = starter_player
        self.winner = 0
        self.done = False

    def valid_moves(self) -> List[int]:
        # movimiento válido si la fila superior (0) está libre
        return [c for c in range(self.W) if self.board[0, c] == 0]

    def step(self, action: int) -> Tuple[np.ndarray, bool, int]:
        """
        Aplica la acción del jugador actual.
        Devuelve: (board, done, winner)
        - Si acción inválida (columna llena), no cambia estado; sigue el mismo jugador.
        - Si acción válida:
            * coloca ficha en la fila disponible más baja
            * verifica victoria o tablas
            * si no termina, alterna jugador
        """
        if self.done:
            return self.board.copy(), True, self.winner

        if action not in self.valid_moves():
            # jugada inválida: no termina, no cambia jugador
            return self.board.copy(), False, 0

        # caer hasta la fila más baja libre
        col = action
        for r in range(self.H - 1, -1, -1):
            if self.board[r, col] == 0:
                self.board[r, col] = self.current_player
                break

        # ¿ganó?
        if self._check_winner_from(r, col, self.current_player):
            self.done = True
            self.winner = self.current_player
            return self.board.copy(), True, self.winner

        # ¿empate?
        if len(self.valid_moves()) == 0:
            self.done = True
            self.winner = 0
            return self.board.copy(), True, self.winner

        # alternar jugador
        self.current_player *= -1
        return self.board.copy(), False, 0

    def _check_winner_from(self, r, c, player) -> bool:
        # direcciones: (dr, dc)
        dirs = [(0,1), (1,0), (1,1), (1,-1)]
        for dr, dc in dirs:
            cnt = 1
            # hacia +dir
            rr, cc = r + dr, c + dc
            while 0 <= rr < self.H and 0 <= cc < self.W and self.board[rr, cc] == player:
                cnt += 1
                rr += dr; cc += dc
            # hacia -dir
            rr, cc = r - dr, c - dc
            while 0 <= rr < self.H and 0 <= cc < self.W and self.board[rr, cc] == player:
                cnt += 1
                rr -= dr; cc -= dc
            if cnt >= self.N_TO_WIN:
                return True
        return False


# ---------- Codificación de estado -> tensor ----------
def encode_state(board: np.ndarray, current_player: int) -> np.ndarray:
    """
    Produce (C=4, H=6, W=7) float32, como en el entrenamiento estándar:
      ch0: posiciones del jugador +1
      ch1: posiciones del jugador -1
      ch2: todo '1' si current_player == +1, else todo '0'
      ch3: todo '1' si current_player == -1, else todo '0'
    """
    H, W = board.shape
    obs = np.zeros((4, H, W), dtype=np.float32)
    obs[0] = (board == 1).astype(np.float32)
    obs[1] = (board == -1).astype(np.float32)
    obs[2] = 1.0 if current_player == 1 else 0.0
    obs[3] = 1.0 if current_player == -1 else 0.0
    return obs


# ---------- Red Q (match con entrenamiento: conv 4->32->64, MLP 256) ----------
class QNetC4_Native(nn.Module):
    """Sin upsample: flatten 64*6*7 = 2688."""
    def __init__(self, in_channels=4, n_actions=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),           nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 7, 256), nn.ReLU(inplace=True),
            nn.Linear(256, n_actions),
        )
    def forward(self, x):
        feats = self.features(x)        # (B,64,6,7)
        return self.head(feats)


class QNetC4_Upsample8(nn.Module):
    """Con upsample a 8x8: flatten 64*8*8 = 4096 (como tu checkpoint)."""
    def __init__(self, in_channels=4, n_actions=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),           nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(inplace=True),
            nn.Linear(256, n_actions),
        )
    def forward(self, x):
        feats = self.features(x)                   # (B,64,6,7)
        feats = nn.functional.interpolate(feats, size=(8, 8), mode="nearest")
        return self.head(feats)


def build_qnet_from_checkpoint(state_dict, device, in_channels=4, n_actions=7):
    """
    Inspecciona el checkpoint y crea la arquitectura compatible.
    """
    # pesos de la primera Linear del head
    # (en nuestras Sequential es 'head.1.weight')
    k = "head.1.weight"
    if k not in state_dict:
        # por compat: algunos guardan como 'head.0.weight' si cambió el orden
        k = "head.0.weight" if "head.0.weight" in state_dict else None
    in_feats = None
    if k is not None:
        in_feats = state_dict[k].shape[1]  # 4096 o 2688

    if in_feats == 4096:
        net = QNetC4_Upsample8(in_channels=in_channels, n_actions=n_actions).to(device)
    elif in_feats == 2688:
        net = QNetC4_Native(in_channels=in_channels, n_actions=n_actions).to(device)
    else:
        # si no podemos inferir, intentamos primero Upsample8 y si falla, Native
        try:
            net = QNetC4_Upsample8(in_channels=in_channels, n_actions=n_actions).to(device)
            net.load_state_dict(state_dict, strict=True)
            return net
        except Exception:
            net = QNetC4_Native(in_channels=in_channels, n_actions=n_actions).to(device)
            # dejar que falle arriba si tampoco coincide
    net.load_state_dict(state_dict, strict=True)
    return net



# ---------- Política greedy con desempate aleatorio ----------
@torch.no_grad()
def greedy_action(policy: nn.Module, board: np.ndarray, current_player: int, device: str) -> int:
    x = torch.from_numpy(encode_state(board, current_player)).unsqueeze(0).to(device)  # (1,4,6,7)
    q = policy(x).squeeze(0).cpu().numpy()  # (7,)
    # invalidar columnas llenas
    valid = [c for c in range(7) if board[0, c] == 0]
    if not valid:
        return 0
    mask = np.full_like(q, -1e9, dtype=np.float32)
    mask[valid] = 0.0
    q = q + mask
    m = q.max()
    best = np.flatnonzero(q == m)
    return int(np.random.choice(best))


# ---------- Render del tablero ----------
def draw_board(board: np.ndarray, width=520):
    H, W = board.shape
    fig, ax = plt.subplots(figsize=(width/130, (width*6/7)/130), dpi=130)
    ax.set_xlim(-0.5, W-0.5)
    ax.set_ylim(-0.5, H-0.5)
    ax.set_xticks(range(W)); ax.set_yticks(range(H))
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.grid(True, color="#334", alpha=0.3)
    # dibujar fichas como círculos
    for r in range(H):
        for c in range(W):
            v = board[r, c]
            color = "#e5defa"  # vacío
            if v == 1:   color = "#ff4d4d"  # rojo (humano)
            if v == -1:  color = "#ffd54d"  # amarillo (modelo)
            circ = plt.Circle((c, r), 0.42, color=color, ec="#223", lw=1.2)
            ax.add_patch(circ)
    ax.invert_yaxis()
    fig.tight_layout(pad=0.2)
    st.pyplot(fig, clear_figure=True)


# ---------- App ----------
st.set_page_config(page_title="Conecta-4 — Juega contra tu DQN", page_icon="⭕", layout="centered")
st.title("🟡🔴 Conecta-4 — Juega contra tu modelo DQN")

# ------- Modelo (.pth): subir o elegir de ./modelos -------
st.subheader("Modelo")
pth_path = None
uploaded = st.file_uploader("Sube tu policy_*.pth (opcional)", type=["pth", "pt", "bin"])

local_paths = []
if os.path.isdir("modelos"):
    local_paths = [os.path.join("modelos", f) for f in os.listdir("modelos") if f.endswith(".pth")]
    local_paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)

option = st.selectbox(
    "O elige un .pth local (./modelos)",
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
    st.warning("Selecciona o sube un .pth para jugar.")

st.divider()

# ------- Quién empieza -------
st.subheader("Quién empieza")
starter = st.radio("Turno inicial", ["Tú (Humano - 🔴)", "Modelo (DQN - 🟡)"], index=0, horizontal=True)

# botón reiniciar
if st.button("🔄 Reiniciar partida"):
    st.session_state.env = Connect4(starter_player=(1 if starter == "Tú (Humano - 🔴)" else -1))
    st.session_state.status = "En juego"
    st.session_state.msg = ""
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# ------- Cargar/crear estado -------
if "env" not in st.session_state:
    st.session_state.env = Connect4(starter_player=(1 if starter == "Tú (Humano - 🔴)" else -1))
if "status" not in st.session_state:
    st.session_state.status = "En juego"
if "msg" not in st.session_state:
    st.session_state.msg = ""

env: Connect4 = st.session_state.env

# ------- Cargar modelo -------
device = "cuda" if torch.cuda.is_available() else "cpu"
policy = None
load_error = None
if pth_path:
    try:
        state = torch.load(pth_path, map_location=device)
        # si el checkpoint fue guardado como {'model_state_dict': ...}
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        policy = build_qnet_from_checkpoint(state, device, in_channels=4, n_actions=7)
        policy.eval()
        st.success("Modelo cargado correctamente ✅")
    except Exception as e:
        load_error = str(e)
        st.error(f"No pude cargar el modelo: {load_error}")

# ------- Si empieza el modelo y aún no movió, que mueva una vez -------
if policy is not None and env.current_player == -1 and not env.done:
    # turno del modelo
    a = greedy_action(policy, env.board, env.current_player, device)
    env.step(a)

# ------- Tablero -------
st.subheader("Tablero")
draw_board(env.board, width=560)

# Estado
turno = "Humano (🔴)" if env.current_player == 1 else "Modelo (🟡)"
estado = "✅ Ganaste" if env.winner == 1 else ("❌ Perdiste" if env.winner == -1 else ("🤝 Empate" if env.done else "⏳ En juego"))
st.info(f"Turno: **{turno}**  |  Estado: **{estado}**")

st.divider()

# ------- Controles de jugada (humano) -------
st.subheader("Tu jugada (elige columna)")
cols = st.columns(7)
clicked = None
for c in range(7):
    with cols[c]:
        if st.button(f"↓ {c}"):
            clicked = c

if clicked is not None:
    if env.done:
        st.warning("La partida ya terminó. Reinicia para jugar de nuevo.")
    elif env.current_player != 1:
        st.warning("No es tu turno.")
    else:
        # Humano juega
        if clicked not in env.valid_moves():
            st.warning("Columna llena. Elige otra.")
        else:
            env.step(clicked)  # humano
            # ¿terminó?
            if env.done:
                pass
            else:
                # turno del modelo
                if policy is None:
                    st.error("No hay modelo cargado. Carga un .pth para que el modelo pueda jugar.")
                else:
                    a = greedy_action(policy, env.board, env.current_player, device)
                    env.step(a)

            # refrescar UI
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()

# Nota al pie
st.caption("Greedy con desempate aleatorio. Arquitectura esperada del modelo: Conv(4→32→64), MLP 256 → 7 acciones.")
