# testing_ui/streamlit_tictactoe.py
import os
import random
import numpy as np
import streamlit as st
from collections import defaultdict

# ---------- Utilidades de TicTacToe (idénticas a tu entrenamiento) ----------
def empty_board():
    return [' ' for _ in range(9)]

def available_moves(board):
    return [i for i, s in enumerate(board) if s == ' ']

def winner(board, square, letter):
    row_ind = square // 3
    row = board[row_ind*3:(row_ind+1)*3]
    if all([s == letter for s in row]): return True
    col_ind = square % 3
    col = [board[col_ind + i*3] for i in range(3)]
    if all([s == letter for s in col]): return True
    if square % 2 == 0:
        if all([board[i] == letter for i in [0,4,8]]): return True
        if all([board[i] == letter for i in [2,4,6]]): return True
    return False

def is_draw(board):
    return ' ' not in board

def state_key(board):
    return tuple(board)

def q_value(qdict, board, action):
    # Q se guardó como dict con claves (tuple(state), action)
    return qdict.get((state_key(board), action), 0.0)

def agent_greedy_action(board, qdict):
    moves = available_moves(board)
    if not moves:
        return None
    qs = [q_value(qdict, board, a) for a in moves]
    max_q = max(qs)
    # desempate al azar entre los mejores
    best = [a for a, q in zip(moves, qs) if q == max_q]
    return random.choice(best)

# ---------- Carga de modelos ----------
def list_qtables(models_dir="modelos"):
    if not os.path.isdir(models_dir):
        return []
    return [os.path.join(models_dir, f) for f in os.listdir(models_dir)
            if f.endswith(".npy") and f.startswith("qtable_")]

def load_qtable(path):
    # guardaste con np.save(dict(...)); se carga como objeto
    data = np.load(path, allow_pickle=True).item()
    # Asegurar dict normal (clave: (tuple(state), action) -> float)
    if isinstance(data, (dict, defaultdict)):
        return dict(data)
    raise ValueError("El archivo no contiene un dict de Q-table válido.")

# ---------- UI ----------
st.set_page_config(page_title="TicTacToe Q-Table UI", page_icon="🎮", layout="centered")
st.title("🎮 Tic Tac Toe — Q-Learning Viewer/Tester")

# Panel lateral: selección de modelo y quién empieza
models = list_qtables("modelos")
if not models:
    st.error("No encontré modelos en la carpeta `modelos/`. Entrena y guarda alguno primero.")
    st.stop()

model_label_map = {os.path.basename(p): p for p in models}
chosen_label = st.sidebar.selectbox("Modelo (.npy):", sorted(model_label_map.keys()))
chosen_model_path = model_label_map[chosen_label]
st.sidebar.caption(f"Ruta: `{chosen_model_path}`")

starter = st.sidebar.radio("¿Quién empieza?", ["Humano", "Agente"], index=0)
human_symbol_first = st.sidebar.radio("Símbolo que usa quien empieza", ["X", "O"], index=0)

# Botón para cargar/recargar modelo
if "loaded_model_path" not in st.session_state or st.session_state.loaded_model_path != chosen_model_path:
    try:
        st.session_state.qtable = load_qtable(chosen_model_path)
        st.session_state.loaded_model_path = chosen_model_path
        st.sidebar.success("Modelo cargado ✅")
    except Exception as e:
        st.sidebar.error(f"Error cargando modelo: {e}")
        st.stop()

# Estado del juego en sesión
if "board" not in st.session_state:
    st.session_state.board = empty_board()
if "game_over" not in st.session_state:
    st.session_state.game_over = False
if "human_letter" not in st.session_state or "agent_letter" not in st.session_state:
    # Config inicial según quién empieza
    if starter == "Humano":
        st.session_state.human_letter = human_symbol_first
        st.session_state.agent_letter = "O" if human_symbol_first == "X" else "X"
        st.session_state.turn = "human"
    else:
        st.session_state.agent_letter = human_symbol_first
        st.session_state.human_letter = "O" if human_symbol_first == "X" else "X"
        st.session_state.turn = "agent"
if "status_msg" not in st.session_state:
    st.session_state.status_msg = ""

# Función para reiniciar respetando selecciones actuales
def reset_game():
    st.session_state.board = empty_board()
    st.session_state.game_over = False
    st.session_state.status_msg = ""
    if starter == "Humano":
        st.session_state.human_letter = human_symbol_first
        st.session_state.agent_letter = "O" if human_symbol_first == "X" else "X"
        st.session_state.turn = "human"
    else:
        st.session_state.agent_letter = human_symbol_first
        st.session_state.human_letter = "O" if human_symbol_first == "X" else "X"
        st.session_state.turn = "agent"

# Si el usuario cambia opciones en el sidebar, reiniciamos
if st.sidebar.button("🔄 Reiniciar partida"):
    reset_game()

# Mostrar info actual
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Jugador humano", st.session_state.human_letter)
with c2:
    st.metric("Agente", st.session_state.agent_letter)
with c3:
    st.metric("Turno", "Humano" if st.session_state.turn == "human" else "Agente")

st.write("---")

# Hacer que el agente juegue automáticamente cuando le toque
def agent_move_if_needed():
    if st.session_state.game_over or st.session_state.turn != "agent":
        return
    move = agent_greedy_action(st.session_state.board, st.session_state.qtable)
    if move is None:
        st.session_state.game_over = True
        st.session_state.status_msg = "Empate (no hay movimientos)."
        return
    st.session_state.board[move] = st.session_state.agent_letter
    if winner(st.session_state.board, move, st.session_state.agent_letter):
        st.session_state.game_over = True
        st.session_state.status_msg = "🤖 El agente gana."
    elif is_draw(st.session_state.board):
        st.session_state.game_over = True
        st.session_state.status_msg = "Empate."
    else:
        st.session_state.turn = "human"

# Si el agente empieza, que haga su primer movimiento
agent_move_if_needed()

# Render tablero 3×3
def on_human_click(idx):
    if st.session_state.game_over or st.session_state.turn != "human":
        return
    if st.session_state.board[idx] != ' ':
        return
    st.session_state.board[idx] = st.session_state.human_letter
    if winner(st.session_state.board, idx, st.session_state.human_letter):
        st.session_state.game_over = True
        st.session_state.status_msg = "🎉 ¡Ganas!"
    elif is_draw(st.session_state.board):
        st.session_state.game_over = True
        st.session_state.status_msg = "Empate."
    else:
        st.session_state.turn = "agent"
        # turno del agente
        agent_move_if_needed()

# UI de los 9 botones
for r in range(3):
    cols = st.columns(3)
    for c in range(3):
        i = r*3 + c
        label = st.session_state.board[i] if st.session_state.board[i] != ' ' else " "
        disabled = (st.session_state.board[i] != ' ') or st.session_state.game_over or (st.session_state.turn != "human")
        with cols[c]:
            st.button(label, key=f"cell_{i}", on_click=on_human_click, args=(i,),
                      use_container_width=True, disabled=disabled)

st.write("---")
if st.session_state.status_msg:
    st.info(st.session_state.status_msg)

# Mostrar algunos detalles del modelo
with st.expander("Detalles del modelo cargado"):
    st.write(f"**Archivo:** `{chosen_model_path}`")
    st.write(f"**Entradas Q almacenadas:** {len(st.session_state.qtable)}")
