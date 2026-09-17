# ======================  Comparativa: QLearning std vs QLearning + LLM (DeepInfra)  ======================
# Entrena los dos modelos episodio por episodio, forzando la MISMA semilla aleatoria antes de
# cada episodio de cada modelo -> mismas decisiones del oponente y misma exploración epsilon-greedy
# en ambos, para que la comparación sea lo más pareja posible.
import os
import re
import time
import random
import argparse
import psutil
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
import torch
from openai import OpenAI

# --------------------  0. Metadatos --------------------
OPONENTE = "random"
SEED = 42

WIN_REWARD = 1.0
LOSS_REWARD = -1.0
DRAW_REWARD = 0.0
BLOCK_BONUS = 0.3

LLM_PROVIDER = "DeepInfra"
LLM_BASE_URL = "https://api.deepinfra.com/v1/openai"
LLM_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
LLM_TEMPERATURE = 0.0
LLM_MAX_RETRIES = 3
LLM_LOSS_REWARD = -1.0  # aplicado en código cuando la jugada del agente permite la derrota (el LLM no lo ve)

# --------------------  1. Juego --------------------------
class TicTacToe:
    def __init__(self):
        self.board = [' ' for _ in range(9)]
        self.current_winner = None

    def reset(self):
        self.board = [' ' for _ in range(9)]
        self.current_winner = None
        return self.board

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def winner(self, square, letter):
        row_ind = square // 3
        row = self.board[row_ind * 3:(row_ind + 1) * 3]
        if all([spot == letter for spot in row]): return True
        col_ind = square % 3
        column = [self.board[col_ind + i * 3] for i in range(3)]
        if all([spot == letter for spot in column]): return True
        if square % 2 == 0:
            if all([self.board[i] == letter for i in [0, 4, 8]]): return True
            if all([self.board[i] == letter for i in [2, 4, 6]]): return True
        return False

    def is_draw(self):
        return ' ' not in self.board

def winning_moves_for(board_list, letter):
    wins = set()
    for i, spot in enumerate(board_list):
        if spot != ' ':
            continue
        b = board_list.copy()
        b[i] = letter
        if _is_win_after_place(b, i, letter):
            wins.add(i)
    return wins

def _is_win_after_place(board, square, letter):
    r0 = (square // 3) * 3
    if board[r0] == board[r0+1] == board[r0+2] == letter: return True
    c = square % 3
    if board[c] == board[c+3] == board[c+6] == letter: return True
    if square % 2 == 0:
        if board[0] == board[4] == board[8] == letter: return True
        if board[2] == board[4] == board[6] == letter: return True
    return False

# --------------------  2. Agente --------------------------
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q_table[(tuple(state), action)]

    def choose_action(self, state, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)
        q_values = [self.get_q(state, a) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_actions):
        max_q_next = max([self.get_q(next_state, a) for a in next_actions], default=0.0)
        old_value = self.q_table[(tuple(state), action)]
        new_value = old_value + self.alpha * (reward + self.gamma * max_q_next - old_value)
        self.q_table[(tuple(state), action)] = new_value

# --------------------  3. Recompensa vía LLM --------------------------
_PROMPT_TEMPLATE = """Evalúas UNA jugada de Tic Tac Toe para Q-Learning. Responde solo con: 1, 0.3, 0 o -1. Sin texto extra.

1    = la jugada gana la partida de inmediato
0.3  = la jugada bloquea una victoria inmediata del oponente
0    = cualquier otra jugada
-1   = el oponente tenía una victoria inmediata disponible y la jugada NO la bloqueó

Antes: {prev_state}
Agente: {agent_letter}
Acción: {action}
Después: {next_state}

Número:
"""

def _parse_reward_value(s: str):
    m = re.search(r"[-+]?\d*\.?\d+", s.strip())
    if not m:
        return None
    try:
        v = float(m.group(0))
        if v < -1: v = -1.0
        if v > 1:  v = 1.0
        return v
    except:
        return None

def llm_reward(client, prev_state, action, next_state, agent_letter, stats,
               model_name=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE, max_retries=LLM_MAX_RETRIES):
    stats["invocaciones_totales"] += 1
    for _ in range(max_retries):
        stats["llamadas_api_totales"] += 1
        try:
            message = _PROMPT_TEMPLATE.format(
                prev_state=prev_state, action=action, next_state=next_state, agent_letter=agent_letter
            )
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                messages=[{"role": "user", "content": message}],
                model=model_name,
                temperature=temperature,
                max_tokens=20,
            )
            stats["tiempo_respuesta_total_segundos"] += time.perf_counter() - t0
            content = (resp.choices[0].message.content or "").strip()
            parsed = _parse_reward_value(content)
            if parsed is not None:
                return parsed, False
        except Exception as e:
            stats["errores_api_totales"] += 1
            print(f"[LLM Error] {e}")
    stats["fallbacks_totales"] += 1
    return 0.0, True

# --------------------  4. Un episodio de cada modelo --------------------------
def jugar_episodio_std(env, agent, agent_letter, opponent_letter, epoch, ep, filas_acciones):
    state = env.reset()
    reward_total = 0.0
    done = False

    while not done:
        prev_board = state.copy()
        available_actions = env.available_moves()
        action = agent.choose_action(state, available_actions)
        env.make_move(action, agent_letter)
        next_state = env.board.copy()

        if env.current_winner == agent_letter:
            reward, done = WIN_REWARD, True
        elif env.is_draw():
            reward, done = DRAW_REWARD, True
        else:
            opp_wins_prev = winning_moves_for(prev_board, opponent_letter)
            reward = BLOCK_BONUS if action in opp_wins_prev else 0.0
            done = False
            opp_actions = env.available_moves()
            if opp_actions:
                opp_action = random.choice(opp_actions)
                env.make_move(opp_action, opponent_letter)
                next_state = env.board.copy()
                if env.current_winner == opponent_letter:
                    reward = LOSS_REWARD
                    done = True
                elif env.is_draw():
                    done = True

        next_actions = env.available_moves()
        agent.update(state, action, reward, next_state, next_actions)
        reward_total += reward

        filas_acciones.append([
            "QLearning_std", epoch + 1, ep + 1, agent_letter, action,
            next_state.copy(), reward, reward_total, False
        ])
        state = env.board.copy()

    if env.current_winner == agent_letter:
        return "win"
    elif env.current_winner == opponent_letter:
        return "loss"
    return "draw"


def jugar_episodio_llm(env, agent, agent_letter, opponent_letter, client, llm_stats, epoch, ep, filas_acciones):
    state = env.reset()
    reward_total = 0.0
    done = False

    while not done:
        available_actions = env.available_moves()
        action = agent.choose_action(state, available_actions)
        env.make_move(action, agent_letter)
        next_state = env.board.copy()

        reward, fue_fallback = llm_reward(client, state, action, next_state, agent_letter, llm_stats)

        if env.current_winner == agent_letter or env.is_draw():
            done = True
        else:
            done = False
            opp_actions = env.available_moves()
            if opp_actions:
                opp_action = random.choice(opp_actions)
                env.make_move(opp_action, opponent_letter)
                next_state = env.board.copy()
                if env.current_winner == opponent_letter:
                    reward = LLM_LOSS_REWARD
                    done = True
                elif env.is_draw():
                    done = True

        reward_total += reward
        next_actions = env.available_moves()
        agent.update(state, action, reward, next_state, next_actions)

        filas_acciones.append([
            "QLearning_LLM", epoch + 1, ep + 1, agent_letter, action,
            next_state.copy(), reward, reward_total, fue_fallback
        ])
        state = env.board.copy()

    if env.current_winner == agent_letter:
        return "win"
    elif env.current_winner == opponent_letter:
        return "loss"
    return "draw"

# --------------------  5. Comparativa --------------------------
def correr_comparativa(client, num_epochs=10, episodes_per_epoch=100):
    env_std, env_llm = TicTacToe(), TicTacToe()
    agent_std, agent_llm = QLearningAgent(), QLearningAgent()

    llm_stats = {
        "invocaciones_totales": 0,
        "llamadas_api_totales": 0,
        "fallbacks_totales": 0,
        "errores_api_totales": 0,
        "tiempo_respuesta_total_segundos": 0.0,
    }

    filas_acciones = []
    filas_computo = []
    filas_victorias = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_episodes = num_epochs * episodes_per_epoch
    base = f"comparativa_std_vs_llm_{OPONENTE}_{timestamp}_ep{total_episodes}"
    os.makedirs("datos_output", exist_ok=True)

    for epoch in range(num_epochs):
        wins_std = draws_std = losses_std = 0
        wins_llm = draws_llm = losses_llm = 0

        for ep in range(episodes_per_epoch):
            idx_global = epoch * episodes_per_epoch + ep
            agent_letter, opponent_letter = ('X', 'O') if idx_global % 2 == 0 else ('O', 'X')
            episode_seed = SEED + idx_global

            random.seed(episode_seed)
            t0 = time.perf_counter()
            resultado_std = jugar_episodio_std(
                env_std, agent_std, agent_letter, opponent_letter, epoch, ep, filas_acciones
            )
            t_std = time.perf_counter() - t0

            random.seed(episode_seed)  # mismo seed -> mismas decisiones aleatorias del oponente
            t0 = time.perf_counter()
            resultado_llm = jugar_episodio_llm(
                env_llm, agent_llm, agent_letter, opponent_letter, client, llm_stats, epoch, ep, filas_acciones
            )
            t_llm = time.perf_counter() - t0

            filas_computo.append(["QLearning_std", epoch+1, ep+1, t_std, psutil.cpu_percent(),
                                   psutil.Process(os.getpid()).memory_info().rss/1024/1024,
                                   torch.cuda.memory_allocated()/1024/1024 if torch.cuda.is_available() else 0])
            filas_computo.append(["QLearning_LLM", epoch+1, ep+1, t_llm, psutil.cpu_percent(),
                                   psutil.Process(os.getpid()).memory_info().rss/1024/1024,
                                   torch.cuda.memory_allocated()/1024/1024 if torch.cuda.is_available() else 0])

            if resultado_std == "win": wins_std += 1
            elif resultado_std == "loss": losses_std += 1
            else: draws_std += 1

            if resultado_llm == "win": wins_llm += 1
            elif resultado_llm == "loss": losses_llm += 1
            else: draws_llm += 1

        wr_std = wins_std / episodes_per_epoch * 100
        wr_llm = wins_llm / episodes_per_epoch * 100
        filas_victorias.append(["QLearning_std", epoch+1, wins_std, draws_std, losses_std, wr_std])
        filas_victorias.append(["QLearning_LLM", epoch+1, wins_llm, draws_llm, losses_llm, wr_llm])

        print(f"=== Epoch {epoch+1}/{num_epochs} ===")
        print(f"  std : winrate={wr_std:5.1f}%  (V{wins_std} E{draws_std} D{losses_std})")
        print(f"  LLM : winrate={wr_llm:5.1f}%  (V{wins_llm} E{draws_llm} D{losses_llm}) "
              f"| fallbacks acumulados={llm_stats['fallbacks_totales']}")

    acciones_df = pd.DataFrame(filas_acciones, columns=[
        "Modelo", "Epoch", "Episodio", "Agente", "Acción", "Board", "Reward", "RewardAcum", "LLM_Fallback"
    ])
    computo_df = pd.DataFrame(filas_computo, columns=[
        "Modelo", "Epoch", "Episodio", "Tiempo(s)", "CPU(%)", "RAM(MB)", "GPU_mem(MB)"
    ])
    victorias_df = pd.DataFrame(filas_victorias, columns=[
        "Modelo", "Epoch", "Victorias", "Empates", "Derrotas", "WinRate(%)"
    ])

    acciones_df.to_csv(f"datos_output/acciones_{base}.csv", index=False)
    computo_df.to_csv(f"datos_output/computo_{base}.csv", index=False)
    victorias_df.to_csv(f"datos_output/victorias_{base}.csv", index=False)

    print("\n=== RESUMEN FINAL ===")
    for modelo in ["QLearning_std", "QLearning_LLM"]:
        sub = victorias_df[victorias_df["Modelo"] == modelo]
        print(f"{modelo}: winrate promedio de los {num_epochs} epochs = {sub['WinRate(%)'].mean():.2f}%")
    if llm_stats["llamadas_api_totales"]:
        print(f"\nLLM — tiempo de respuesta promedio: "
              f"{llm_stats['tiempo_respuesta_total_segundos']/llm_stats['llamadas_api_totales']:.4f}s "
              f"| fallback rate: {100*llm_stats['fallbacks_totales']/llm_stats['invocaciones_totales']:.2f}%")

    print(f"\n[OK] CSVs guardados en datos_output/ con base: {base}")
    return acciones_df, computo_df, victorias_df

# --------------------  6. API key + Main --------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepinfra-api-key", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--episodes-per-epoch", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    deepinfra_api_key = (
        args.deepinfra_api_key
        or os.environ.get("DEEPINFRA_API_KEY")
        or os.environ.get("DEEPINFRA")
    )
    if not deepinfra_api_key:
        raise RuntimeError(
            "Falta la API key de DeepInfra. Pásala con DEEPINFRA_API_KEY=tu_key antes del comando, "
            "o con --deepinfra-api-key tu_key."
        )

    client = OpenAI(api_key=deepinfra_api_key, base_url=LLM_BASE_URL)

    acciones, computo, victorias = correr_comparativa(
        client=client,
        num_epochs=args.num_epochs,
        episodes_per_epoch=args.episodes_per_epoch,
    )