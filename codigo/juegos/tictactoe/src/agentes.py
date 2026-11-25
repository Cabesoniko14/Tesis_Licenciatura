# ======================  Tic-Tac-Toe: agentes Q-Learning tabular y DQN  ======================
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common import QNetMLP, ReplayBuffer

# --------------------  1. Q-Learning tabular --------------------
class QLearningAgent:
    """Estado = tablero completo (tupla de 9 casillas). Solo elige entre jugadas legales."""
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(float)

    def choose_action(self, board, legales, greedy=False):
        if not greedy and np.random.rand() < self.epsilon:
            return int(np.random.choice(legales))
        q = np.array([self.Q[(tuple(board), a)] for a in legales])
        return int(np.random.choice(np.array(legales)[q == q.max()]))

    def update(self, board, a, r, board_next, terminal, legales_next):
        max_q_next = 0.0 if terminal else max(self.Q[(tuple(board_next), b)] for b in legales_next)
        td_target = r + self.gamma * max_q_next
        clave = (tuple(board), a)
        self.Q[clave] += self.alpha * (td_target - self.Q[clave])

    def estado(self):
        return {"Q": dict(self.Q)}

    def cargar_estado(self, estado):
        self.Q = defaultdict(float, estado["Q"])

# --------------------  2. Double DQN --------------------
class DQNAgent:
    """Las jugadas ilegales se enmascaran al elegir y al calcular el objetivo."""
    def __init__(self, cfg_dqn, n_actions=9, device="cpu"):
        self.cfg = cfg_dqn
        self.n_actions = n_actions
        self.device = device
        self.policy = QNetMLP(n_actions=n_actions).to(device)
        self.target = QNetMLP(n_actions=n_actions).to(device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg_dqn["lr"])
        self.buffer = ReplayBuffer(capacity=cfg_dqn["buffer"])
        self.step_count = 0

    def epsilon(self):
        frac = min(1.0, self.step_count / self.cfg["eps_pasos"])
        return self.cfg["eps_inicio"] + (self.cfg["eps_fin"] - self.cfg["eps_inicio"]) * frac

    def choose_action(self, obs, legales, greedy=False):
        if not greedy and np.random.rand() < self.epsilon():
            return int(np.random.choice(legales))
        with torch.no_grad():
            q = self.policy(torch.from_numpy(obs).unsqueeze(0).to(self.device)).squeeze(0).cpu().numpy()
        return int(legales[int(np.argmax(q[legales]))])

    def step(self, s, a, r, s2, terminal, mascara2):
        """Guarda la transición, entrena si toca y sincroniza la red objetivo."""
        self.buffer.push(s, a, r, s2, float(terminal), mascara2)
        loss = None
        if len(self.buffer) >= self.cfg["inicio_aprendizaje"] and self.step_count % self.cfg["entrenar_cada"] == 0:
            loss = self._aprender()
        if self.step_count % self.cfg["copia_red_objetivo"] == 0:
            self.target.load_state_dict(self.policy.state_dict())
        self.step_count += 1
        return loss

    def _aprender(self):
        s_b, a_b, r_b, s2_b, d_b, m2_b = self.buffer.sample(self.cfg["batch"])
        s_b_t  = torch.from_numpy(s_b).to(self.device)
        a_b_t  = torch.from_numpy(a_b).long().to(self.device)
        r_b_t  = torch.from_numpy(r_b).float().to(self.device)
        s2_b_t = torch.from_numpy(s2_b).to(self.device)
        d_b_t  = torch.from_numpy(d_b).float().to(self.device)
        m2_b_t = torch.from_numpy(m2_b).to(self.device)

        qsa = self.policy(s_b_t).gather(1, a_b_t.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            q_policy_next = self.policy(s2_b_t).masked_fill(m2_b_t == 0, -1e9)
            next_actions = q_policy_next.argmax(dim=1)
            q_next = self.target(s2_b_t).gather(1, next_actions.view(-1, 1)).squeeze(1)
            target_q = r_b_t + self.cfg["gamma"] * (1.0 - d_b_t) * q_next
        loss = nn.MSELoss()(qsa, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg["clip_gradiente"])
        self.optimizer.step()
        return float(loss.item())

    def estado(self):
        return {"policy": self.policy.state_dict(), "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(), "step_count": self.step_count,
                "buffer": (self.buffer.buf, self.buffer.pos)}

    def cargar_estado(self, estado):
        self.policy.load_state_dict(estado["policy"])
        self.target.load_state_dict(estado["target"])
        self.optimizer.load_state_dict(estado["optimizer"])
        self.step_count = estado["step_count"]
        self.buffer.buf, self.buffer.pos = estado["buffer"]
