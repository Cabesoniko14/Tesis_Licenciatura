# ======================  Traza: 5 juegos, std vs LLM  ======================
import os
import re
import random
from collections import defaultdict
from datetime import datetime
from openai import OpenAI

NUM_JUEGOS = 5
SEED = 42

WIN_REWARD = 1.0
LOSS_REWARD = -1.0
DRAW_REWARD = 0.0
BLOCK_BONUS = 0.3

LLM_BASE_URL = "https://api.deepinfra.com/v1/openai"
LLM_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
LLM_TEMPERATURE = 0.0
LLM_MAX_RETRIES = 3
LLM_LOSS_REWARD = -1.0

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

def render_board(board):
    def c(i):
        return board[i] if board[i] != ' ' else str(i)
    return (f" {c(0)} | {c(1)} | {c(2)} \n"
            f"---+---+---\n"
            f" {c(3)} | {c(4)} | {c(5)} \n"
            f"---+---+---\n"
            f" {c(6)} | {c(7)} | {c(8)} ")

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

def llm_reward(client, prev_state, action, next_state, agent_letter, max_retries=LLM_MAX_RETRIES):
    for _ in range(max_retries):
        try:
            message = _PROMPT_TEMPLATE.format(
                prev_state=prev_state, action=action, next_state=next_state, agent_letter=agent_letter
            )
            resp = client.chat.completions.create(
                messages=[{"role": "user", "content": message}],
                model=LLM_MODEL_NAME,
                temperature=LLM_TEMPERATURE,
                max_tokens=20,
            )
            content = (resp.choices[0].message.content or "").strip()
            parsed = _parse_reward_value(content)
            if parsed is not None:
                return parsed, content
        except Exception as e:
            print(f"[LLM Error] {e}")
    return 0.0, "(sin respuesta válida)"

def jugar_std(agent, agent_letter, opponent_letter, log):
    env = TicTacToe()
    state = env.reset()
    reward_total = 0.0
    done = False

    while not done:
        prev_board = state.copy()
        available_actions = env.available_moves()
        action = agent.choose_action(state, available_actions)
        env.make_move(action, agent_letter)
        next_state = env.board.copy()

        fue_bloqueo = action in winning_moves_for(prev_board, opponent_letter)

        if env.current_winner == agent_letter:
            reward, done = WIN_REWARD, True
        elif env.is_draw():
            reward, done = DRAW_REWARD, True
        else:
            reward = BLOCK_BONUS if fue_bloqueo else 0.0
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

        log.append(f"Agente ({agent_letter}) -> casilla {action}  |  reward={reward:+.2f}  acumulado={reward_total:+.2f}")
        log.append(render_board(next_state))
        log.append("")

        state = env.board.copy()

    if env.current_winner == agent_letter:
        return "VICTORIA", reward_total
    elif env.current_winner == opponent_letter:
        return "DERROTA", reward_total
    return "EMPATE", reward_total

def jugar_llm(agent, agent_letter, opponent_letter, client, log):
    env = TicTacToe()
    state = env.reset()
    reward_total = 0.0
    done = False

    while not done:
        available_actions = env.available_moves()
        action = agent.choose_action(state, available_actions)
        env.make_move(action, agent_letter)
        next_state = env.board.copy()

        reward, respuesta_cruda = llm_reward(client, state, action, next_state, agent_letter)

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

        next_actions = env.available_moves()
        agent.update(state, action, reward, next_state, next_actions)
        reward_total += reward

        log.append(f"Agente ({agent_letter}) -> casilla {action}  |  LLM respondió '{respuesta_cruda}'  |  reward={reward:+.2f}  acumulado={reward_total:+.2f}")
        log.append(render_board(next_state))
        log.append("")

        state = env.board.copy()

    if env.current_winner == agent_letter:
        return "VICTORIA", reward_total
    elif env.current_winner == opponent_letter:
        return "DERROTA", reward_total
    return "EMPATE", reward_total

def encabezado():
    return (
        "ESTRATEGIA STD\n"
        f"WIN_REWARD={WIN_REWARD:+.2f}  DRAW_REWARD={DRAW_REWARD:+.2f}  "
        f"BLOCK_BONUS={BLOCK_BONUS:+.2f}  LOSS_REWARD={LOSS_REWARD:+.2f}\n"
        "Bloqueo = el agente ocupa una casilla que el oponente podía usar para ganar "
        "en su siguiente turno (calculado con winning_moves_for sobre el tablero previo).\n\n"
        "PROMPT DEL LLM\n"
        f"{_PROMPT_TEMPLATE}\n"
    )

def correr_traza(client, num_juegos=NUM_JUEGOS):
    agent_std = QLearningAgent()
    agent_llm = QLearningAgent()
    log = [encabezado()]

    for i in range(num_juegos):
        seed_juego = SEED + i
        agent_letter, opponent_letter = ('X', 'O') if i % 2 == 0 else ('O', 'X')

        log.append(f"--- Juego {i+1} | STD ---")
        random.seed(seed_juego)
        resultado_std, reward_std = jugar_std(agent_std, agent_letter, opponent_letter, log)
        log.append(f">>> {resultado_std} | reward total={reward_std:+.2f}\n")

        log.append(f"--- Juego {i+1} | LLM ---")
        random.seed(seed_juego)
        resultado_llm, reward_llm = jugar_llm(agent_llm, agent_letter, opponent_letter, client, log)
        log.append(f">>> {resultado_llm} | reward total={reward_llm:+.2f}\n")

    texto_completo = "\n".join(log)
    print(texto_completo)

    os.makedirs("datos_output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = f"datos_output/traza_5_juegos_std_vs_llm_{timestamp}.txt"
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(texto_completo)

    print(f"\n[OK] guardado en: {ruta}")
    return ruta


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepinfra-api-key", type=str, default=None)
    parser.add_argument("--num-juegos", type=int, default=NUM_JUEGOS)
    args = parser.parse_args()

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
    correr_traza(client, num_juegos=args.num_juegos)