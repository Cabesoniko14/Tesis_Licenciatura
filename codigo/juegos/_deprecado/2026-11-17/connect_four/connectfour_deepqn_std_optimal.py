# ====================== DQN: Connect-4 (vs Agente Heurístico rápido, sin árbol) ======================
# - Misma estructura de logging/guardado que tu baseline
# - Recompensa solo al FINAL: win=1, draw=0, loss=0
# - Oponente = reglas heurísticas (bloquea 3-en-línea del rival, busca centro, extiende rachas)
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
        return [c for c in range(self.COLS) if self.board[0, c] == 0]

    def step(self, action):
        assert not self.done, "episode finished"
        if action not in self.available_actions():
            # Acción ilegal: terminamos y dejamos reward=0 (final-only)
            self.done = True
            self.winner = 3 - self.current_player  # el otro gana por ilegal
            r = 0.0
            return self._obs(next_player=True), r, True, False, {"illegal": True}

        row = self._drop_row(action)
        self.board[row, action] = self.current_player

        if self._check_win(row, action, self.current_player):
            self.done = True
            self.winner = self.current_player
            r = 1.0
            return self._obs(next_player=True), r, True, False, {}

        if len(self.available_actions()) == 0:
            self.done = True
            self.winner = 0
            r = 0.0
            return self._obs(next_player=True), r, True, False, {}

        r = 0.0
        self.current_player = 3 - self.current_player
        return self._obs(), r, False, False, {}

    def _drop_row(self, col):
        for r in range(self.ROWS - 1, -1, -1):
            if self.board[r, col] == 0:
                return r
        raise RuntimeError("columna llena (debería estar filtrada)")

    def _check_win(self, r, c, p):
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
        cur = self.current_player
        if next_player:
            cur = 3 - cur
        return (self.board.copy(), cur)

# -------------------------------------------------------
# 2) Observación -> tensor (C=4, H=6, W=7)
# -------------------------------------------------------
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
            nn.Linear(64 * 8 * 8, 256),  # upsample (6x7 -> 8x8)
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
# 5) Epsilon schedule lineal + helpers
# -------------------------------------------------------
def epsilon_by_step(t, eps_start=1.0, eps_end=0.05, eps_decay_steps=50_000):
    if eps_decay_steps <= 0:
        return eps_end
    frac = min(1.0, t / eps_decay_steps)
    return eps_start + (eps_end - eps_start) * frac

def select_action(q_values, legal_actions):
    mask = np.full_like(q_values, -1e9, dtype=np.float32)
    mask[legal_actions] = 0.0
    masked_q = q_values + mask
    m = masked_q.max()
    best = np.flatnonzero(masked_q == m)
    return int(np.random.choice(best))

def board_hash(board):
    return int(np.int64(np.abs(hash(bytes(board.tobytes()))) % (10**9 + 7)))

# -------------------------------------------------------
# 6) Agente Heurístico (sin árbol)
# -------------------------------------------------------
class HeuristicAgent:
    """
    Reglas rápidas (O(#acciones)):
      1) Si hay victoria inmediata -> jugarla.
      2) Si el rival tiene victoria inmediata -> bloquearla.
      3) Preferir centro y columnas cercanas.
      4) Puntuar jugadas que extienden rachas propias / bloquean rachas rivales.
      5) Evitar (si puede) dar victoria inmediata al rival en el siguiente turno.
    """
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng(0)
        self.ROWS = Connect4Env.ROWS
        self.COLS = Connect4Env.COLS

    def choose_action(self, board, current_player, legal_actions):
        opp = 3 - current_player
        # 1) Win inmediato
        for c in legal_actions:
            r = self._drop_row_sim(board, c)
            board[r, c] = current_player
            won = self._is_win(board, r, c, current_player)
            board[r, c] = 0
            if won:
                return c

        # 2) Bloquear win inmediato del rival
        for c in legal_actions:
            r = self._drop_row_sim(board, c)
            board[r, c] = opp
            opp_wins = self._is_win(board, r, c, opp)
            board[r, c] = 0
            if opp_wins:
                return c

        # 3) Scoring heurístico simple
        best_score = -1e18
        best = []
        for c in legal_actions:
            r = self._drop_row_sim(board, c)
            s = self._score_move(board, r, c, current_player)
            # 5) Evitar dar win inmediato al rival
            board[r, c] = current_player
            opp_wins_next = self._rival_can_win_next(board, opp)
            board[r, c] = 0
            if opp_wins_next:
                s -= 5e4  # fuerte penalización

            if s > best_score + 1e-9:
                best_score = s
                best = [c]
            elif abs(s - best_score) <= 1e-9:
                best.append(c)

        if not best:
            best = list(legal_actions)
        # desempate favoreciendo centro
        best.sort(key=lambda cc: abs(cc - 3))
        return int(self.rng.choice(best[:2]))  # pequeño random entre mejores

    # ---------- helpers ----------
    def _drop_row_sim(self, board, col):
        for r in range(self.ROWS - 1, -1, -1):
            if board[r, col] == 0:
                return r
        raise RuntimeError("columna llena")

    def _is_win(self, board, r, c, p):
        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in dirs:
            cnt = 1
            cnt += self._count_dir(board, r, c, dr, dc, p)
            cnt += self._count_dir(board, r, c, -dr, -dc, p)
            if cnt >= 4:
                return True
        return False

    def _count_dir(self, board, r, c, dr, dc, p):
        R, C = self.ROWS, self.COLS
        k = 0
        rr, cc = r + dr, c + dc
        while 0 <= rr < R and 0 <= cc < C and board[rr, cc] == p:
            k += 1
            rr += dr
            cc += dc
        return k

    def _rival_can_win_next(self, board, opp):
        legal = [c for c in range(self.COLS) if board[0, c] == 0]
        for c in legal:
            r = self._drop_row_sim(board, c)
            board[r, c] = opp
            win = self._is_win(board, r, c, opp)
            board[r, c] = 0
            if win:
                return True
        return False

    def _score_move(self, board, r, c, p):
        """Heurística local: centro, adyacencias y 'ventanas' 4-celdas."""
        opp = 3 - p
        score = 0.0

        # Centro y cercanía a centro
        score += 120 - 10 * abs(c - 3)

        # Simular poner ficha
        board[r, c] = p

        # Bonus por formar 3-en-línea con hueco y 2-en-línea
        score += 300 * self._count_windows(board, p, need=3, empties=1)
        score += 40  * self._count_windows(board, p, need=2, empties=2)

        # Penalizar amenazas del rival (equilibrado para no volverse puramente defensivo)
        score -= 350 * self._count_windows(board, opp, need=3, empties=1)
        score -= 45  * self._count_windows(board, opp, need=2, empties=2)

        # Ligerísimo sesgo por “apilar” sobre fichas propias (vertical)
        if r < self.ROWS - 1 and board[r + 1, c] == p:
            score += 25

        board[r, c] = 0
        return score

    def _count_windows(self, board, player, need=3, empties=1):
        """Cuenta ventanas de 4 celdas con 'need' fichas de player y 'empties' vacías."""
        R, C = self.ROWS, self.COLS
        cnt = 0

        # Horizontal
        for rr in range(R):
            row = board[rr, :]
            for cc in range(C - 3):
                w = row[cc:cc + 4]
                if (w == player).sum() == need and (w == 0).sum() == empties and (w == 3 - player).sum() == 0:
                    cnt += 1
        # Vertical
        for cc in range(C):
            col = board[:, cc]
            for rr in range(R - 3):
                w = col[rr:rr + 4]
                if (w == player).sum() == need and (w == 0).sum() == empties and (w == 3 - player).sum() == 0:
                    cnt += 1
        # Diagonales
        for rr in range(R - 3):
            for cc in range(C - 3):
                w = np.array([board[rr + i, cc + i] for i in range(4)])
                if (w == player).sum() == need and (w == 0).sum() == empties and (w == 3 - player).sum() == 0:
                    cnt += 1
        for rr in range(3, R):
            for cc in range(C - 3):
                w = np.array([board[rr - i, cc + i] for i in range(4)])
                if (w == player).sum() == need and (w == 0).sum() == empties and (w == 3 - player).sum() == 0:
                    cnt += 1
        return cnt

# -------------------------------------------------------
# 7) Entrenamiento DQN (vs HeuristicAgent)
# -------------------------------------------------------
def train_dqn_connect4_vs_heuristic(
    num_epochs=50,
    episodes_per_epoch=100,
    max_steps_per_ep=84,
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
    eps_decay_steps=50_000
):
    """
    Entrena contra un oponente heurístico rápido.
    Se registran transiciones del agente y del oponente (como en tu baseline).
    """
    set_global_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_actions = 7
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_episodes = num_epochs * episodes_per_epoch
    base = f"connect4_dqn_vs_heuristic_{timestamp}_episodes_{total_episodes}"
    os.makedirs("datos_output", exist_ok=True)
    os.makedirs("modelos", exist_ok=True)

    # Tablas
    acciones_df = pd.DataFrame(columns=["Epoch", "Episodio", "Paso", "StateIdx", "Accion", "NextStateIdx", "Reward", "Done", "Epsilon"])
    computo_df = pd.DataFrame(columns=["Epoch", "Episodio", "Tiempo(s)", "CPU(%)", "RAM(MB)", "GPU_mem(MB)"])
    victorias_df = pd.DataFrame(columns=["Epoch", "Exitos", "Fracasos", "WinRate(%)"])
    resumen_df = pd.DataFrame(columns=["Métrica", "Valor"])

    # Redes + buffer + optim (DDQN)
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
    opp = HeuristicAgent(rng=np.random.default_rng(seed))

    def starter_for(ep_idx):
        return 1 if (ep_idx % 2 == 0) else 2

    for epoch in range(num_epochs):
        success_epoch, fail_epoch = 0, 0

        for ep in range(episodes_per_epoch):
            t0 = time.perf_counter()
            env.reset(start_player=starter_for(epoch * episodes_per_epoch + ep))
            done = False

            (board, cur) = env._obs()
            s_enc = encode_obs_connect4((board, cur))
            s_idx = board_hash(board)

            for t in range(1, max_steps_per_ep + 1):
                # --- TURNO DEL JUGADOR ACTUAL (policy)
                eps = epsilon_by_step(step_count, eps_start, eps_end, eps_decay_steps)
                legal = env.available_actions()

                if np.random.rand() < eps:
                    a = int(np.random.choice(legal))
                else:
                    with torch.no_grad():
                        x = torch.from_numpy(s_enc).unsqueeze(0).to(device)
                        q = policy(x).squeeze(0).cpu().numpy()
                        a = select_action(q, legal)

                next_obs, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                (board2, cur2) = next_obs
                s2_enc = encode_obs_connect4((board2, cur2))
                ns_idx = board_hash(board2)

                buffer.push(s_enc, a, r, s2_enc, float(done))
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

                if step_count % target_update_every == 0:
                    target.load_state_dict(policy.state_dict())

                step_count += 1
                s_enc = s2_enc
                s_idx = ns_idx

                if done:
                    start_p = starter_for(epoch * episodes_per_epoch + ep)
                    if env.winner == start_p:
                        successes_total += 1
                        success_epoch += 1
                    else:
                        fails_total += 1
                        fail_epoch += 1
                    break

                # --- TURNO DEL OPONENTE: Heurístico (sin árbol)
                legal = env.available_actions()
                if len(legal) == 0:
                    continue

                a_opp = opp.choose_action(env.board, env.current_player, legal)

                next_obs2, r2, term2, trunc2, _ = env.step(a_opp)
                done = term2 or trunc2
                (board3, cur3) = next_obs2
                s3_enc = encode_obs_connect4((board3, cur3))
                ns2_idx = board_hash(board3)

                buffer.push(s2_enc, a_opp, r2, s3_enc, float(done))
                acciones_df.loc[len(acciones_df)] = [epoch + 1, ep + 1, t, ns_idx, a_opp, ns2_idx, r2, done, 0.0]

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
        "eps_decay_steps": eps_decay_steps
    }
    with open(f"modelos/meta_{base}.json", "w") as f:
        json.dump(meta, f)

    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    print(f"policy: modelos/policy_{base}.pth")
    print(f"target: modelos/target_{base}.pth")
    print(f"meta:   modelos/meta_{base}.json")
    return base

# -------------------------------------------------------
# 8) Evaluación (greedy) vs HeuristicAgent
# -------------------------------------------------------
@torch.no_grad()
def evaluate_agent_dqn_vs_heuristic(
    policy_path,
    episodes=200,
    max_steps=84,
    seed=123
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = QNetC4(in_channels=4, n_actions=7).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()

    env = Connect4Env(seed=seed)
    opp = HeuristicAgent(rng=np.random.default_rng(seed))
    wins = 0

    for ep in range(episodes):
        start_player = 1 if (ep % 2 == 0) else 2
        env.reset(start_player=start_player)

        (b, cur) = env._obs()
        s_enc = encode_obs_connect4((b, cur))

        for _ in range(1, max_steps + 1):
            legal = env.available_actions()
            if cur == start_player:
                # turno del que inició: política evaluada
                x = torch.from_numpy(s_enc).unsqueeze(0).to(device)
                q = policy(x).squeeze(0).cpu().numpy()
                a = select_action(q, legal)
            else:
                # turno del oponente heurístico
                a = opp.choose_action(env.board, env.current_player, legal)

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
    base = train_dqn_connect4_vs_heuristic(
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
    )

    wr = evaluate_agent_dqn_vs_heuristic(
        policy_path=f"modelos/policy_{base}.pth",
        episodes=200,
        max_steps=84,
        seed=123,
    )
    print(f"\nWinRate evaluación (vs Heurístico): {wr:.2f}%")
