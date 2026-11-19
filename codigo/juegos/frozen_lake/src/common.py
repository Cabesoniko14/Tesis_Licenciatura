# ======================  Frozen Lake: entorno, red, buffer y métricas  ======================
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

ACCION_NOMBRE = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

# --------------------  1. Reproducibilidad --------------------
def set_global_seeds(seed, determinista=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if determinista:
        torch.use_deterministic_algorithms(True)

def semillas_de_mapas(seed, n):
    """Lista de semillas para generar la misma secuencia de mapas en todas las corridas."""
    rng = np.random.default_rng(seed)
    return [int(s) for s in rng.integers(0, 1e9, size=n)]

# --------------------  2. Entorno --------------------
def make_env(cfg_mapa, seed_mapa=None):
    """Mapa fijo de Gym o aleatorio con semilla. El truncamiento a max_pasos lo hace Gym."""
    if cfg_mapa["modo"] == "fijo":
        env = gym.make("FrozenLake-v1", map_name=cfg_mapa["nombre_gym"],
                       is_slippery=cfg_mapa["resbaloso"], max_episode_steps=cfg_mapa["max_pasos"])
    else:
        desc = generate_random_map(size=cfg_mapa["tamano"], seed=seed_mapa)
        env = gym.make("FrozenLake-v1", desc=desc,
                       is_slippery=cfg_mapa["resbaloso"], max_episode_steps=cfg_mapa["max_pasos"])
    return env, env.unwrapped.desc

def desc_a_filas(desc):
    """desc (bytes H x W) -> ['SFFF', 'FHFH', ...]"""
    return ["".join(c.decode("ascii") for c in fila) for fila in desc]

# --------------------  3. Observación (mapa + agente) -> tensor --------------------
TILE_TO_CH = {b'F': 0, b'H': 1, b'G': 2, b'S': 0}

def idx_to_rc(idx, size):
    return idx // size, idx % size

def tile_at(desc, idx):
    r, c = idx_to_rc(idx, desc.shape[1])
    return desc[r, c].decode("ascii")

def encode_obs_from_desc(desc, agent_idx):
    H, W = desc.shape
    obs = np.zeros((4, H, W), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            obs[TILE_TO_CH.get(desc[r, c], 0), r, c] = 1.0
    ar, ac = idx_to_rc(agent_idx, W)
    obs[3, ar, ac] = 1.0
    return obs

# --------------------  4. Red y Replay Buffer --------------------
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
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        feats = self.features(x)
        if H == 4 and W == 4:
            feats = nn.functional.interpolate(feats, size=(8, 8), mode='nearest')
        return self.head(feats)

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

# --------------------  5. Métricas de ruta --------------------
def ruta_optima(desc):
    """Pasos de la ruta más corta de S a G evitando hoyos (BFS)."""
    H, W = desc.shape
    meta = H * W - 1
    dist = {0: 0}
    cola = deque([0])
    while cola:
        s = cola.popleft()
        r, c = idx_to_rc(s, W)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            ns = nr * W + nc
            if 0 <= nr < H and 0 <= nc < W and desc[nr, nc] != b'H' and ns not in dist:
                dist[ns] = dist[s] + 1
                cola.append(ns)
    return dist.get(meta)

def metricas_ruta(desc, estados):
    """estados: [s0, s1, ..., sT] visitados en el episodio."""
    H, W = desc.shape
    hoyos = [(r, c) for r in range(H) for c in range(W) if desc[r, c] == b'H']
    celdas = [idx_to_rc(s, W) for s in estados]
    dist_hoyo = [min(abs(r - hr) + abs(c - hc) for hr, hc in hoyos) for r, c in celdas] if hoyos else [None] * len(celdas)
    intermedias = dist_hoyo[1:-1] if len(dist_hoyo) > 2 else []
    return {
        "pasos": len(estados) - 1,
        "ruta_optima": ruta_optima(desc),
        "frac_borde": float(np.mean([r in (0, H - 1) or c in (0, W - 1) for r, c in celdas])),
        "celdas_junto_hoyo": sum(d == 1 for d in intermedias),
        "dist_min_hoyo": min(intermedias) if intermedias and hoyos else None,
    }
