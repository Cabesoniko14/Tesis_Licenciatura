import streamlit as st
import pickle
import random
import numpy as np

# ---------- Cargar Q-Table entrenada ----------
QTABLE_PATH = "../modelos/qtable_tictactoe.pkl"  # <-- cambia ruta si es necesario
with open(QTABLE_PATH, "rb") as f:
    q_table = pickle.load(f)

# ---------- Funciones del juego ----------
def get_available_moves(board):
    return [i for i, s in enumerate(board) if s == " "]

def get_winner(board):
    combos = [(0,1,2),(3,4,5),(6,7,8),
              (0,3,6),(1,4,7),(2,5,8),
              (0,4,8),(2,4,6)]
    for a,b,c in combos:
        if board[a] != " " and board[a] == board[b] == board[c]:
            return board[a]
    return None

def agent_move(board, agent_letter):
    state = tuple(board)
    available = get_available_moves(board)
    q_values = q_table.get(state, {})
    if q_values:
        best = max(available, key=lambda a: q_values.get(a, 0))
    else:
        best = random.choice(available)
    board[best] = agent_letter
    return board

# ---------- Interfaz Streamlit ----------
st.set_page_config(page_title="Tic Tac Toe vs Q-Learning", page_icon="🎮")
st.title("🎮 Tic Tac Toe vs Q-Learning Agent")

# Inicializar variables en session_state
if "board" not in st.session_state:
    st.session_state.board = [" "] * 9
if "turn" not in st.session_state:
    st.session_state.turn = "X"   # X siempre empieza por default
if "human" not in st.session_state:
    st.session_state.human = "X"
    st.session_state.agent = "O"
if "game_over" not in st.session_state:
    st.session_state.game_over = False

# Configuración inicial
with st.sidebar:
    st.header("⚙️ Configuración")
    choice = st.radio("¿Quieres jugar como?", ["X (empiezas)", "O (segundo)"])
    if choice.startswith("X"):
        st.session_state.human, st.session_state.agent = "X", "O"
        st.session_state.turn = "X"
    else:
        st.session_state.human, st.session_state.agent = "O", "X"
        st.session_state.turn = "X"  # aún empieza X, pero será la IA
        if st.session_state.board == [" "] * 9:
            st.session_state.board = agent_move(st.session_state.board, st.session_state.agent)
            st.session_state.turn = st.session_state.human

    if st.button("🔄 Reiniciar Juego"):
        st.session_state.board = [" "] * 9
        st.session_state.turn = "X"
        st.session_state.game_over = False
        if st.session_state.human == "O":  # IA juega primero
            st.session_state.board = agent_move(st.session_state.board, st.session_state.agent)
            st.session_state.turn = st.session_state.human

# Mostrar tablero (3x3 botones)
cols = st.columns(3)
for i in range(9):
    if st.session_state.board[i] == " " and not st.session_state.game_over and st.session_state.turn == st.session_state.human:
        if cols[i % 3].button(" ", key=f"cell_{i}"):
            st.session_state.board[i] = st.session_state.human
            winner = get_winner(st.session_state.board)
            if winner or " " not in st.session_state.board:
                st.session_state.game_over = True
            else:
                st.session_state.turn = st.session_state.agent
                st.session_state.board = agent_move(st.session_state.board, st.session_state.agent)
                winner = get_winner(st.session_state.board)
                if winner or " " not in st.session_state.board:
                    st.session_state.game_over = True
    else:
        cols[i % 3].button(st.session_state.board[i], key=f"cell_{i}", disabled=True)

# Mostrar estado del juego
winner = get_winner(st.session_state.board)
if winner:
    st.success(f"🏆 Ganador: {winner}")
elif st.session_state.game_over:
    st.info("🤝 Empate!")
else:
    st.write(f"Turno de: {st.session_state.turn}")
