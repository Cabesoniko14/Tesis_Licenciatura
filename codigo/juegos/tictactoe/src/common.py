# ======================  Tic-Tac-Toe: juego, recompensa estándar, red y buffer  ======================
import random

import numpy as np
import torch
import torch.nn as nn

LINEAS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

# --------------------  1. Reproducibilidad --------------------
def set_global_seeds(seed, determinista=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if determinista:
        torch.use_deterministic_algorithms(True)

# --------------------  2. Juego --------------------
class TicTacToe:
    """Tablero = lista de 9 casillas (' ', 'X' u 'O'), numeradas de 0 a 8 por filas."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [' '] * 9
        self.ganador = None
        return self.board.copy()

    def jugadas_legales(self):
        return [i for i, c in enumerate(self.board) if c == ' ']

    def jugar(self, casilla, letra):
        self.board[casilla] = letra
        if gana(self.board, letra):
            self.ganador = letra

    def lleno(self):
        return ' ' not in self.board

    def terminado(self):
        return self.ganador is not None or self.lleno()

def gana(board, letra):
    return any(all(board[i] == letra for i in linea) for linea in LINEAS)

def jugadas_ganadoras(board, letra):
    """Casillas vacías donde 'letra' completaría tres en línea."""
    ganadoras = set()
    for i in range(9):
        if board[i] == ' ':
            b = board.copy()
            b[i] = letra
            if gana(b, letra):
                ganadoras.add(i)
    return ganadoras

def rival_de(letra):
    return 'O' if letra == 'X' else 'X'

def casilla_rc(i):
    return i // 3, i % 3

# --------------------  3. Recompensa estándar (también es la respuesta correcta para el LLM) --------------------
def categoria_jugada(board_antes, accion, board_final, letra_agente):
    """Clasifica un turno completo del agente: su jugada y la respuesta del rival.
    Prioridad: gana > pierde > bloquea > empate > otra."""
    rival = rival_de(letra_agente)
    if gana(board_final, letra_agente):
        return "gana"
    if gana(board_final, rival):
        return "pierde"
    if accion in jugadas_ganadoras(board_antes, rival):
        return "bloquea"
    if ' ' not in board_final:
        return "empate"
    return "otra"

def categoria_partida(board_final, letra_agente):
    if gana(board_final, letra_agente):
        return "gana"
    if gana(board_final, rival_de(letra_agente)):
        return "pierde"
    return "empate"

# --------------------  4. Observación -> tensor --------------------
def encode_obs(board, letra_agente):
    """9 valores: 1 = agente, -1 = rival, 0 = vacía."""
    return np.array([1.0 if c == letra_agente else (-1.0 if c != ' ' else 0.0) for c in board], dtype=np.float32)

def mascara(legales):
    m = np.zeros(9, dtype=np.float32)
    m[legales] = 1.0
    return m

# --------------------  5. Red y Replay Buffer --------------------
class QNetMLP(nn.Module):
    def __init__(self, n_entradas=9, n_actions=9):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_entradas, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.layers(x)

class ReplayBuffer:
    """Guarda también la máscara de jugadas legales del siguiente estado."""
    def __init__(self, capacity=100_000):
        self.capacity = capacity
        self.buf = []
        self.pos = 0

    def push(self, s, a, r, s2, d, m2):
        if len(self.buf) < self.capacity:
            self.buf.append(None)
        self.buf[self.pos] = (s, a, r, s2, d, m2)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d, m2 = map(np.array, zip(*batch))
        return s, a, r, s2, d, m2

    def __len__(self):
        return len(self.buf)
