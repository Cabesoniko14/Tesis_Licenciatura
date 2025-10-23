# ======================  DQN: Connect4 (no Gym) — recompensa SOLO final (win=1, draw/loss=0)  ======================
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

# =====================================================
# 0) Utils reproducibilidad (idéntico)
# =====================================================
def set_global_seeds(seed=None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =====================================================
# 1) Conecta4: entorno mínimo (sin Gym)
# =====================================================
class Connect4:
    """
    Tablero 6x7. Representación interna:
      0 = vacío, 1 = agente (P1), 2 = oponente (P2).
    El paso 'step(a)' SIEMPRE coloca primero el agente.
    Si no termina, mueve el oponente (random).
    """
    H, W = 6, 7

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((self.H, self.W), dtype=np.int8)
        self.current_player = 1  # 1=agente, 2=oponente (gestionado internamente)

    def reset(self):
        self.board[:, :] = 0
        self.current_player = 1
        return self._obs()

    def _obs(self):
        # Devolvemos copia del estado "lógico" para el encoder
        return (self.board.copy(), self.current_player)

    def valid_actions(self):
        # columnas no llenas
        return [c for c in range(self.W) if self.board[0, c] == 0]

    def _drop(self, col, player):
        # Colocar ficha en la fila más baja disponible de la columna
        for r in range(self.H - 1, -1, -1):
            if self.board[r, col] == 0:
                self.board[r, col] = player
                return r, col
        return None  # columna llena (no debería pasar si validamos)

    def _check_winner_from(self, r, c):
        player = self.board[r, c]
        if player == 0:
            return False

        directions = [(1,0),(0,1),(1,1),(1,-1)]
        for dr, dc in directions:
            cnt = 1
            # forward
            rr, cc = r+dr, c+dc
            while 0 <= rr < self.H and 0 <= cc < self.W and self.board[rr, cc] == player:
                cnt += 1; rr += dr; cc += dc
            # backward
            rr, cc = r-dr, c-dc
            while 0 <= rr < self.H and 0 <= cc < self.W and self.board[rr, cc] == player:
                cnt += 1; rr -= dr; cc -= dc
            if cnt >= 4:
                return True
        return False

    def _is_draw(self):
        return np.all(self.board[0, :] != 0)

    def step(self, action_agent):
        """
        Devuelve: next_obs, env_reward, terminated, truncated, info
        - env_reward: 1.0 si el AGENTE gana; 0.0 en cualquier otro caso.
        """
        # 1) Jugada del agente (si acción inválida, sustituimos por aleatoria válida)
        valids = self.valid_actions()
        if not valids:
            # empate por tablero lleno (raro llegar aquí sin detectarlo antes)
            return self._obs(), 0.0, True, False, {}
        if action_agent not in valids:
            action_agent = random.choice(valids)

        pos = self._drop(action_agent, 1)
        r, c = pos
        # ¿Ganó el agente?
        if self._check_winner_from(r, c):
            return self._obs(), 1.0, True, False, {}  # win del agente
        if self._is_draw():
            return self._obs(), 0.0, True, False, {}  # empate

        # 2) Mueve el oponente (aleatorio)
        opp_valids = self.valid_actions()
        if not opp_valids:
            return self._obs(), 0.0, True, False, {}  # empate
        opp_action = random.choice(opp_valids)
        pos2 = self._drop(opp_action, 2)
        r2, c2 = pos2
        if self._check_winner_from(r2, c2):
            return self._obs(), 0.0, True, False, {}  # derrota del agente -> reward 0
        if self._is_draw():
            return self._obs(), 0.0, True, False, {}  # empate

        return self._obs(), 0.0, False, False, {}

# =====================================================
# 1.5) Observación -> tensor (MISMA forma lógica que FL)
#     Salida: (C=4, H=8, W=8)
#     canales: [AGENT, OPP, EMPTY, WHOSE_TURN]
# =====================================================
def encode_obs_from_board(board_np, current_player):
    """
    board_np: (6,7) con {0,1,2}
    current_player: 1 ó 2 (para el plano WHOSE_TURN)
    Rellenamos/‘pad’ a 8x8 (superior-izquierda el 6x7 real).
    """
    H, W = 8, 8
    obs = np.zeros((4, H, W), dtype=np.float32)

    # canales base sobre 6x7
    agent = (board_np == 1).astype(np.float32)
    opp   = (board_np == 2).astype(np.float32)
    empty = (board_np == 0).astype(np.float32)

    # pegamos en la esquina superior-izquierda
    obs[0, :6, :7] = agent
    obs[1, :6, :7] = opp
    obs[2, :6, :7] = empty

    # plano de "quién mueve": siempre mueve el agente en nuestra API (antes de step)
    # lo ponemos completo (8x8) para que la red lo vea como sesgo fijo por turno
    obs[3, :, :] = 1.0 if current_player == 1 else 0.0
    return obs

# =====================================================
# 1.6) Estado -> entero (para mantener columnas del CSV)
#      Codificación base-3 determinista del 6x7
# =====================================================
def board_to_index(board_np):
    # mapa: 0->0, 1->1, 2->2; base-3: sum(cell * 3^k)
    # k recorre row-major del 6x7 real.
    v = 0
    pow3 = 1
    for r in range(6):
        for c in range(7):
            v += int(board_np[r, c]) * pow3
            pow3 *= 3
    return int(v)

# =====================================================
# 2) Red y Replay Buffer (idénticos)
# =====================================================
class QNetCNN(nn.Module):
    """CNN compacta. Head definida para 8x8 (igual que en el estándar)."""
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
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        # x: (B, C=4, H=8, W=8) — aquí SIEMPRE 8x8 por nuestro padding
        feats = self.features(x)
        out = self.head(feats)
        return out

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

# =====================================================
# 3) Entrenamiento DQN (estructura de guardado idéntica)
# =====================================================
def train_dqn_connect4(
    num_epochs=50,
    episodes_per_epoch=100,
    max_steps_per_ep=84,  # 6*7*2 ~ movimientos con oponente; 84 seguro alcanza
    seed=42,
    # Hiperparámetros DQN (idénticos estilo FrozenLake)
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
    set_global_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_actions = 7  # columnas
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_episodes = num_epochs * episodes_per_epoch
    base = f"connect4_dqn_standard_random_{timestamp}_episodes_{total_episodes}"

    os.makedirs("datos_output", exist_ok=True)
    os.makedirs("modelos", exist_ok=True)

    # Tablas (mismo esquema de nombres que tu estándar)
    acciones_df  = pd.DataFrame(columns=["Epoch","Episodio","Paso","StateIdx","Accion","NextStateIdx","Reward","Done","Epsilon"])
    computo_df   = pd.DataFrame(columns=["Epoch","Episodio","Tiempo(s)","CPU(%)","RAM(MB)","GPU_mem(MB)"])
    victorias_df = pd.DataFrame(columns=["Epoch","Exitos","Fracasos","WinRate(%)"])
    resumen_df   = pd.DataFrame(columns=["Métrica","Valor"])

    # Redes + optim + buffer
    policy = QNetCNN(in_channels=4, n_actions=n_actions).to(device)
    target = QNetCNN(in_channels=4, n_actions=n_actions).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=buffer_capacity)
    step_count = 0

    # Epsilon schedule lineal
    def get_epsilon(t):
        if eps_decay_steps <= 0:
            return eps_end
        frac = min(1.0, t / eps_decay_steps)
        return eps_start + (eps_end - eps_start) * frac

    total_time = 0.0
    successes_total, fails_total = 0, 0

    for epoch in range(num_epochs):
        success_epoch, fail_epoch = 0, 0

        for ep in range(episodes_per_epoch):
            t0 = time.perf_counter()

            env = Connect4(seed=seed + epoch*episodes_per_epoch + ep)
            (board, cur) = env.reset()
            obs_tensor = encode_obs_from_board(board, cur)
            done = False

            for t in range(max_steps_per_ep):
                eps = get_epsilon(step_count)

                # Máscara de acciones inválidas
                valids = env.valid_actions()
                if np.random.rand() < eps:
                    a = random.choice(valids)
                else:
                    with torch.no_grad():
                        x = torch.from_numpy(obs_tensor).unsqueeze(0).to(device)
                        q = policy(x).squeeze(0).cpu().numpy()
                        # enmascarar con -inf en inválidas
                        mask = np.full_like(q, -1e9, dtype=np.float32)
                        mask[valids] = 0.0
                        q_masked = q + mask
                        best = np.flatnonzero(q_masked == q_masked.max())
                        a = int(np.random.choice(best))

                (next_board, next_cur), r_env, terminated, truncated, _ = env.step(a)
                done = terminated or truncated

                # Recompensa SOLO final: 1 si gana agente, else 0
                r = float(r_env) if done else 0.0

                next_obs_tensor = encode_obs_from_board(next_board, next_cur)

                # indices enteros para CSV (mantener esquema)
                s_idx  = board_to_index(board)
                ns_idx = board_to_index(next_board)

                buffer.push(obs_tensor, a, r, next_obs_tensor, float(done))
                acciones_df.loc[len(acciones_df)] = [epoch+1, ep+1, t+1, s_idx, a, ns_idx, r, done, eps]

                # aprender
                if len(buffer) >= start_learning_after and step_count % train_every == 0:
                    s_batch, a_batch, r_batch, s2_batch, d_batch = buffer.sample(batch_size)

                    s_batch_t  = torch.from_numpy(s_batch).to(device)
                    a_batch_t  = torch.from_numpy(a_batch).long().to(device)
                    r_batch_t  = torch.from_numpy(r_batch).float().to(device)
                    s2_batch_t = torch.from_numpy(s2_batch).to(device)
                    d_batch_t  = torch.from_numpy(d_batch).float().to(device)

                    qsa = policy(s_batch_t).gather(1, a_batch_t.view(-1,1)).squeeze(1)
                    with torch.no_grad():
                        next_actions = policy(s2_batch_t).argmax(dim=1)
                        q_next = target(s2_batch_t).gather(1, next_actions.view(-1,1)).squeeze(1)
                        target_q = r_batch_t + gamma * (1.0 - d_batch_t) * q_next

                    loss = nn.MSELoss()(qsa, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                    optimizer.step()

                if step_count % target_update_every == 0:
                    target.load_state_dict(policy.state_dict())

                step_count += 1
                obs_tensor = next_obs_tensor
                board = next_board  # para log de idx en el próximo paso

                if done:
                    # contabilizar éxito según r_env (1 si ganó el AGENTE)
                    if r_env > 0:
                        successes_total += 1; success_epoch += 1
                    else:
                        fails_total += 1; fail_epoch += 1
                    break

            elapsed = time.perf_counter() - t0
            total_time += elapsed
            computo_df.loc[len(computo_df)] = [
                epoch+1, ep+1, elapsed,
                psutil.cpu_percent(),
                psutil.Process(os.getpid()).memory_info().rss/1024/1024,
                torch.cuda.memory_allocated()/1024/1024 if torch.cuda.is_available() else 0
            ]

        wr = 100.0 * success_epoch / episodes_per_epoch
        victorias_df.loc[len(victorias_df)] = [epoch+1, success_epoch, fail_epoch, wr]
        print(f"=== Epoch {epoch+1}/{num_epochs} terminado | winrate={wr:.2f}% ===")

    resumen_df.loc[len(resumen_df)] = ["Éxitos", successes_total]
    resumen_df.loc[len(resumen_df)] = ["Fracasos", fails_total]
    resumen_df.loc[len(resumen_df)] = ["Tiempo total (s)", total_time]
    resumen_df.loc[len(resumen_df)] = ["GPU usada", torch.cuda.is_available()]
    resumen_df.loc[len(resumen_df)] = ["Modo", "connect4 | reward final: win=1, draw/loss=0"]
    resumen_df.loc[len(resumen_df)] = ["Episodios totales", total_episodes]

    acciones_df.to_csv(f"datos_output/acciones_{base}.csv", index=False)
    computo_df.to_csv(f"datos_output/computo_{base}.csv", index=False)
    victorias_df.to_csv(f"datos_output/victorias_{base}.csv", index=False)
    resumen_df.to_csv(f"datos_output/resumen_{base}.csv", index=False)

    torch.save(policy.state_dict(), f"modelos/policy_{base}.pth")
    torch.save(target.state_dict(), f"modelos/target_{base}.pth")

    meta = {
        "game": "connect4",
        "board_shape": [6, 7],
        "obs_shape": [4, 8, 8],
        "n_actions": int(n_actions),
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
    }
    with open(f"modelos/meta_{base}.json", "w") as f:
        json.dump(meta, f)

    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    print(f"policy: modelos/policy_{base}.pth")
    print(f"target: modelos/target_{base}.pth")
    print(f"meta:   modelos/meta_{base}.json")
    return base

# =====================================================
# 4) Evaluación (greedy con desempate aleatorio) — igual idea
# =====================================================
@torch.no_grad()
def evaluate_agent_dqn(
    policy_path,
    episodes=1000,
    max_steps=84,
    seed=123,
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = QNetCNN(in_channels=4, n_actions=7).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()

    wins = 0
    rng = np.random.default_rng(seed)

    for ep in range(episodes):
        env = Connect4(seed=seed + ep)
        (board, cur) = env.reset()
        for _ in range(max_steps):
            obs = torch.from_numpy(encode_obs_from_board(board, cur)).unsqueeze(0).to(device)
            q = policy(obs).squeeze(0).cpu().numpy()
            valids = env.valid_actions()
            mask = np.full_like(q, -1e9, dtype=np.float32)
            mask[valids] = 0.0
            q_masked = q + mask
            best = np.flatnonzero(q_masked == q_masked.max())
            a = int(np.random.choice(best))

            (board, cur), r_env, terminated, truncated, _ = env.step(a)
            if terminated or truncated:
                if r_env > 0:
                    wins += 1
                break

    return 100.0 * wins / episodes

# =====================================================
# 5) Main de ejemplo (idéntico estilo)
# =====================================================
if __name__ == "__main__":
    base = train_dqn_connect4(
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

    wr = evaluate_agent_dqn(
        policy_path=f"modelos/policy_{base}.pth",
    )
    print(f"\nWinRate de evaluación (vs oponente aleatorio): {wr:.2f}%")
