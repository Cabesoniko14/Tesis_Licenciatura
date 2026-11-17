# ====================== DQN: Connect-4 (Self-Play con target net) ======================
# - Misma estructura de logging/guardado que tu estándar
# - Recompensa solo al FINAL: win=1, draw=0, loss=0
# - Opponente = target network (greedy con pequeño epsilon) -> self-play estable
# - Convención fija: upsample (6x7 -> 8x8) antes del head para 4096 features

import os
import time
import json
import psutil
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import random

# -------------------------------------------------------
# 0) Utils reproducibilidad
# -------------------------------------------------------
def set_global_seeds(seed=None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------------------------------------------------
# 1) Conecta-4 env (6x7) — sin Gym
# -------------------------------------------------------
class Connect4Env:
    """
    Tablero 6x7; acciones 0..6 (columna para soltar ficha).
    Representación interna:
      0 = vacío, 1 = jugador 1 (P1), 2 = jugador 2 (P2).
    current_player in {1,2}.
    Terminal si hay 4 en raya o no hay movimientos.
    """
    ROWS = 6
    COLS = 7

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self.current_player = 1
        self.winner = 0  # 0 none, 1 P1, 2 P2
        self.done = False

    def reset(self, start_player=None):
        self.board.fill(0)
        self.winner = 0
        self.done = False
        if start_player is None:
            self.current_player = 1 if self.rng.integers(0, 2) == 0 else 2
        else:
            self.current_player = 1 if start_player == 1 else 2
        return self._obs()

    def available_actions(self):
        # columnas donde el tope está vacío
        return [c for c in range(self.COLS) if self.board[0, c] == 0]

    def step(self, action):
        """
        Ejecuta acción del jugador actual.
        Devuelve: next_obs, reward (desde PERSPECTIVA del jugador que acaba de mover),
        terminated, truncated(False), info{}
        NOTA: reward final: 1 si ese jugador ganó; 0 si perdió o empate.
        En pasos no terminales, reward=0.
        """
        assert not self.done, "episode finished"
        if action not in self.available_actions():
            # Acción ilegal: terminamos y dejamos reward=0 (final-only)
            self.done = True
            self.winner = 3 - self.current_player  # el otro gana por ilegal
            r = 0.0
            return self._obs(next_player=True), r, True, False, {"illegal": True}

        # soltar ficha en la fila más baja disponible
        row = self._drop_row(action)
        self.board[row, action] = self.current_player

        # comprobar terminal
        if self._check_win(row, action, self.current_player):
            self.done = True
            self.winner = self.current_player
            r = 1.0  # final-only
            # obs "siguiente" ya sería del otro jugador, pero no importa al terminar
            return self._obs(next_player=True), r, True, False, {}

        if len(self.available_actions()) == 0:
            # empate
            self.done = True
            self.winner = 0
            r = 0.0
            return self._obs(next_player=True), r, True, False, {}

        # no terminal: cambio de jugador
        r = 0.0
        self.current_player = 3 - self.current_player
        return self._obs(), r, False, False, {}

    def _drop_row(self, col):
        for r in range(self.ROWS - 1, -1, -1):
            if self.board[r, col] == 0:
                return r
        raise RuntimeError("columna llena (debería estar filtrada)")

    def _check_win(self, r, c, p):
        # direcciones: (dr, dc)
        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in dirs:
            cnt = 1
            cnt += self._count_dir(r, c, dr, dc, p)
            cnt += self._count_dir(r, c, -dr, -dc, p)
            if cnt >= 4:
                return True
        return False

    def _count_dir(self, r, c, dr, dc, p):
        R, C = self.ROWS, self.COLS
        k = 0
        rr, cc = r + dr, c + dc
        while 0 <= rr < R and 0 <= cc < C and self.board[rr, cc] == p:
            k += 1
            rr += dr
            cc += dc
        return k

    def _obs(self, next_player=False):
        """
        Devuelve (board_copy, current_player) para construir la observación.
        next_player=True fuerza el indicador de "turno" al contrario (para estados terminales).
        """
        cur = self.current_player
        if next_player:
            cur = 3 - cur
        return (self.board.copy(), cur)

# -------------------------------------------------------
# 2) Observación -> tensor (C=4, H=6, W=7)
# -------------------------------------------------------
# Canales:
# 0: posiciones del jugador 1 (P1)
# 1: posiciones del jugador 2 (P2)
# 2: vacíos
# 3: plano constante = 1 si el "jugador al que le toca" es P1, 0 si es P2
def encode_obs_connect4(board_cur):
    board, cur = board_cur
    H, W = board.shape
    obs = np.zeros((4, H, W), dtype=np.float32)
    obs[0] = (board == 1).astype(np.float32)
    obs[1] = (board == 2).astype(np.float32)
    obs[2] = (board == 0).astype(np.float32)
    obs[3] = 1.0 if cur == 1 else 0.0
    return obs

# -------------------------------------------------------
# 3) Red con upsample a 8x8 (convención fija)
# -------------------------------------------------------
class QNetC4(nn.Module):
    def __init__(self, in_channels=4, n_actions=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),  # upsample (6x7 -> 8x8) => 4096
            nn.ReLU(inplace=True),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        feats = self.features(x)  # (B,64,6,7)
        feats = nn.functional.interpolate(feats, size=(8, 8), mode="nearest")
        return self.head(feats)  # (B,7)

# -------------------------------------------------------
# 4) Replay Buffer
# -------------------------------------------------------
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

# -------------------------------------------------------
# 5) Epsilon schedule lineal
# -------------------------------------------------------
def epsilon_by_step(t, eps_start=1.0, eps_end=0.05, eps_decay_steps=50_000):
    if eps_decay_steps <= 0:
        return eps_end
    frac = min(1.0, t / eps_decay_steps)
    return eps_start + (eps_end - eps_start) * frac

# -------------------------------------------------------
# 6) Helpers de acción (máscara de acciones ilegales)
# -------------------------------------------------------
def select_action(q_values, legal_actions):
    # desempate aleatorio solo entre legales con Q máximo
    mask = np.full_like(q_values, -1e9, dtype=np.float32)
    mask[legal_actions] = 0.0
    masked_q = q_values + mask
    m = masked_q.max()
    best = np.flatnonzero(masked_q == m)
    return int(np.random.choice(best))

def board_hash(board):
    # entero reproducible para logging (mismo tipo de columna que tu estándar)
    return int(np.int64(np.abs(hash(bytes(board.tobytes()))) % (10**9 + 7)))

# -------------------------------------------------------
# 7) Entrenamiento DQN (self-play con oponente=target)
# -------------------------------------------------------
def train_dqn_connect4_selfplay(
    num_epochs=50,
    episodes_per_epoch=100,
    max_steps_per_ep=84,   # máx. jugadas en 6x7 (llenar tablero)
    seed=42,
    # DQN
    gamma=0.99,
    lr=1e-3,
    buffer_capacity=100_000,
    batch_size=64,
    start_learning_after=2_000,
    train_every=1,
    target_update_every=1_000,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=50_000,
    # oponente
    opp_epsilon=0.10
):
    """
    Baseline:
    - recompensa solo terminal (win=1, draw=0, loss=0)
    - self-play vs target net (greedy con epsilon chico)
    - logging/CSV idéntico al estándar
    """
    set_global_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_actions = 7
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_episodes = num_epochs * episodes_per_epoch
    base = f"connect4_dqn_selfplay_target_{timestamp}_episodes_{total_episodes}"
    os.makedirs("datos_output", exist_ok=True)
    os.makedirs("modelos", exist_ok=True)

    # Tablas
    acciones_df = pd.DataFrame(columns=["Epoch", "Episodio", "Paso", "StateIdx", "Accion", "NextStateIdx", "Reward", "Done", "Epsilon"])
    computo_df = pd.DataFrame(columns=["Epoch", "Episodio", "Tiempo(s)", "CPU(%)", "RAM(MB)", "GPU_mem(MB)"])
    victorias_df = pd.DataFrame(columns=["Epoch", "Exitos", "Fracasos", "WinRate(%)"])
    resumen_df = pd.DataFrame(columns=["Métrica", "Valor"])

    # Redes + buffer + optim
    policy = QNetC4(in_channels=4, n_actions=n_actions).to(device)
    target = QNetC4(in_channels=4, n_actions=n_actions).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_capacity)
    step_count = 0

    total_time = 0.0
    successes_total, fails_total = 0, 0

    env = Connect4Env(seed=seed)

    # alternar quién inicia por episodio
    def starter_for(ep_idx):
        return 1 if (ep_idx % 2 == 0) else 2

    for epoch in range(num_epochs):
        success_epoch, fail_epoch = 0, 0

        for ep in range(episodes_per_epoch):
            t0 = time.perf_counter()
            env.reset(start_player=starter_for(epoch * episodes_per_epoch + ep))
            done = False

            # estado inicial
            (board, cur) = env._obs()
            s_enc = encode_obs_connect4((board, cur))
            s_idx = board_hash(board)

            for t in range(1, max_steps_per_ep + 1):
                # --- TURNO DEL JUGADOR ACTUAL (usa 'policy')
                eps = epsilon_by_step(step_count, eps_start, eps_end, eps_decay_steps)
                legal = env.available_actions()

                if np.random.rand() < eps:
                    a = int(np.random.choice(legal))
                else:
                    with torch.no_grad():
                        x = torch.from_numpy(s_enc).unsqueeze(0).to(device)  # (1,4,6,7)
                        q = policy(x).squeeze(0).cpu().numpy()               # (7,)
                        a = select_action(q, legal)

                # aplicar acción (del jugador actual)
                next_obs, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                (board2, cur2) = next_obs
                s2_enc = encode_obs_connect4((board2, cur2))
                ns_idx = board_hash(board2)

                # push transición (del jugador actual)
                buffer.push(s_enc, a, r, s2_enc, float(done))

                # logging
                acciones_df.loc[len(acciones_df)] = [epoch + 1, ep + 1, t, s_idx, a, ns_idx, r, done, eps]

                # aprender
                if len(buffer) >= start_learning_after and step_count % train_every == 0:
                    s_b, a_b, r_b, s2_b, d_b = buffer.sample(batch_size)
                    s_b_t = torch.from_numpy(s_b).to(device)
                    a_b_t = torch.from_numpy(a_b).long().to(device)
                    r_b_t = torch.from_numpy(r_b).float().to(device)
                    s2_b_t = torch.from_numpy(s2_b).to(device)
                    d_b_t = torch.from_numpy(d_b).float().to(device)

                    qsa = policy(s_b_t).gather(1, a_b_t.view(-1, 1)).squeeze(1)
                    with torch.no_grad():
                        na = policy(s2_b_t).argmax(dim=1)
                        q_next = target(s2_b_t).gather(1, na.view(-1, 1)).squeeze(1)
                        target_q = r_b_t + gamma * (1.0 - d_b_t) * q_next

                    loss = nn.MSELoss()(qsa, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                    optimizer.step()

                # target update
                if step_count % target_update_every == 0:
                    target.load_state_dict(policy.state_dict())

                step_count += 1
                s_enc = s2_enc
                s_idx = ns_idx

                if done:
                    # éxito/fallo contado desde la perspectiva del que inició el episodio:
                    # si winner==start_player => éxito; si no => fallo (empate=fallo por reward=0)
                    start_p = starter_for(epoch * episodes_per_epoch + ep)
                    if env.winner == start_p:
                        successes_total += 1
                        success_epoch += 1
                    else:
                        fails_total += 1
                        fail_epoch += 1
                    break

                # --- TURNO DEL OPONENTE (self-play usando TARGET + epsilon pequeño)
                legal = env.available_actions()
                if len(legal) == 0:
                    # empate por llenado: se marcará al comenzar el siguiente step
                    continue

                if np.random.rand() < opp_epsilon:
                    a_opp = int(np.random.choice(legal))
                else:
                    with torch.no_grad():
                        x2 = torch.from_numpy(s_enc).unsqueeze(0).to(device)
                        q2 = target(x2).squeeze(0).cpu().numpy()
                        a_opp = select_action(q2, legal)

                next_obs2, r2, term2, trunc2, _ = env.step(a_opp)
                done = term2 or trunc2
                (board3, cur3) = next_obs2
                s3_enc = encode_obs_connect4((board3, cur3))
                ns2_idx = board_hash(board3)

                # entrenamos también con transiciones del oponente
                buffer.push(s_enc, a_opp, r2, s3_enc, float(done))

                # logging "como paso siguiente"
                acciones_df.loc[len(acciones_df)] = [epoch + 1, ep + 1, t, ns_idx, a_opp, ns2_idx, r2, done, opp_epsilon]

                # aprender
                if len(buffer) >= start_learning_after and step_count % train_every == 0:
                    s_b, a_b, r_b, s2_b, d_b = buffer.sample(batch_size)
                    s_b_t = torch.from_numpy(s_b).to(device)
                    a_b_t = torch.from_numpy(a_b).long().to(device)
                    r_b_t = torch.from_numpy(r_b).float().to(device)
                    s2_b_t = torch.from_numpy(s2_b).to(device)
                    d_b_t = torch.from_numpy(d_b).float().to(device)

                    qsa = policy(s_b_t).gather(1, a_b_t.view(-1, 1)).squeeze(1)
                    with torch.no_grad():
                        na = policy(s2_b_t).argmax(dim=1)
                        q_next = target(s2_b_t).gather(1, na.view(-1, 1)).squeeze(1)
                        target_q = r_b_t + gamma * (1.0 - d_b_t) * q_next

                    loss = nn.MSELoss()(qsa, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                    optimizer.step()

                if step_count % target_update_every == 0:
                    target.load_state_dict(policy.state_dict())

                step_count += 1
                s_enc = s3_enc
                s_idx = ns2_idx

                if done:
                    start_p = starter_for(epoch * episodes_per_epoch + ep)
                    if env.winner == start_p:
                        successes_total += 1
                        success_epoch += 1
                    else:
                        fails_total += 1
                        fail_epoch += 1
                    break

            # cómputo episodio
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            computo_df.loc[len(computo_df)] = [
                epoch + 1, ep + 1, elapsed,
                psutil.cpu_percent(),
                psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
                torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            ]

        wr = 100.0 * success_epoch / max(1, episodes_per_epoch)
        victorias_df.loc[len(victorias_df)] = [epoch + 1, success_epoch, fail_epoch, wr]
        print(f"=== Epoch {epoch+1}/{num_epochs} terminado | winrate={wr:.2f}% ===")

    # Resumen global y guardado
    resumen_df.loc[len(resumen_df)] = ["Éxitos", successes_total]
    resumen_df.loc[len(resumen_df)] = ["Fracasos", fails_total]
    resumen_df.loc[len(resumen_df)] = ["Tiempo total (s)", total_time]
    resumen_df.loc[len(resumen_df)] = ["GPU usada", torch.cuda.is_available()]
    resumen_df.loc[len(resumen_df)] = ["Episodios totales", total_episodes]

    acciones_df.to_csv(f"datos_output/acciones_{base}.csv", index=False)
    computo_df.to_csv(f"datos_output/computo_{base}.csv", index=False)
    victorias_df.to_csv(f"datos_output/victorias_{base}.csv", index=False)
    resumen_df.to_csv(f"datos_output/resumen_{base}.csv", index=False)

    torch.save(policy.state_dict(), f"modelos/policy_{base}.pth")
    torch.save(target.state_dict(), f"modelos/target_{base}.pth")

    meta = {
        "game": "connect4",
        "n_actions": int(n_actions),
        "board": [int(Connect4Env.ROWS), int(Connect4Env.COLS)],
        "obs_shape": [4, 6, 7],
        "upsample_to": [8, 8],
        "max_steps_per_ep": max_steps_per_ep,
        "timestamp": timestamp,
        "base": base,
        "gamma": gamma,
        "lr": lr,
        "buffer_capacity": buffer_capacity,
        "batch_size": batch_size,
        "start_learning_after": start_learning_after,
        "train_every": train_every,
        "target_update_every": target_update_every,
        "eps_start": eps_start,
        "eps_end": eps_end,
        "eps_decay_steps": eps_decay_steps,
        "opp_epsilon": opp_epsilon,
    }
    with open(f"modelos/meta_{base}.json", "w") as f:
        json.dump(meta, f)

    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    print(f"policy: modelos/policy_{base}.pth")
    print(f"target: modelos/target_{base}.pth")
    print(f"meta:   modelos/meta_{base}.json")
    return base

# -------------------------------------------------------
# 8) Evaluación (greedy con desempate aleatorio) vs oponente target
# -------------------------------------------------------
@torch.no_grad()
def evaluate_agent_dqn_connect4(
    policy_path,
    episodes=200,
    max_steps=84,
    seed=123,
    opp_epsilon=0.0  # 0 -> oponente puramente greedy target
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = QNetC4(in_channels=4, n_actions=7).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()

    # oponente = copia congelada de policy (target)
    target = QNetC4(in_channels=4, n_actions=7).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    env = Connect4Env(seed=seed)
    wins = 0

    def select_model_action(net, obs, legal):
        x = torch.from_numpy(obs).unsqueeze(0).to(device)
        q = net(x).squeeze(0).cpu().numpy()
        return select_action(q, legal)

    for ep in range(episodes):
        start_player = 1 if (ep % 2 == 0) else 2
        env.reset(start_player=start_player)

        (b, cur) = env._obs()
        s_enc = encode_obs_connect4((b, cur))

        for _ in range(1, max_steps + 1):
            legal = env.available_actions()
            if cur == start_player:
                a = select_model_action(policy, s_enc, legal)
            else:
                if np.random.rand() < opp_epsilon:
                    a = int(np.random.choice(legal))
                else:
                    a = select_model_action(target, s_enc, legal)

            next_obs, _, term, trunc, _ = env.step(a)
            (b2, cur2) = next_obs
            s_enc = encode_obs_connect4((b2, cur2))
            cur = cur2

            if term or trunc:
                if env.winner == start_player:
                    wins += 1
                break

    return 100.0 * wins / max(1, episodes)

# -------------------------------------------------------
# 9) Main
# -------------------------------------------------------
if __name__ == "__main__":
    base = train_dqn_connect4_selfplay(
        num_epochs=50,
        episodes_per_epoch=100,
        max_steps_per_ep=84,
        seed=42,
        gamma=0.99,
        lr=1e-3,
        buffer_capacity=100_000,
        batch_size=64,
        start_learning_after=2_000,
        train_every=1,
        target_update_every=1_000,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=50_000,
        opp_epsilon=0.10,   # oponente (target) con pequeña exploración
    )

    wr = evaluate_agent_dqn_connect4(
        policy_path=f"modelos/policy_{base}.pth",
        episodes=200,
        max_steps=84,
        seed=123,
        opp_epsilon=0.0,    # evaluación contra target greedy
    )
    print(f"\nWinRate evaluación (vs target greedy): {wr:.2f}%")
