# -----------------------  TIC TAC TOE: Q-LEARNING STANDARD -----------------------------

# Hecho con ayuda de ChatGPT
# Objetivo: entrenar un agente con Q-learning para que aprenda a jugar tic tac toe
# Funcionamiento: El archivo entrena durante un número de episodios seleccionados jugando contra otro agente,
# y cada 1% del total de episodios evalúa contra un oponente aleatorio (100 juegos) y guarda el porcentaje de victorias y empates.

# -----------------------  1. Librerías -----------------------------

import os
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

# --------------------  2. Clase del juego --------------------------

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
        if all([spot == letter for spot in row]):
            return True
        col_ind = square % 3
        column = [self.board[col_ind + i * 3] for i in range(3)]
        if all([spot == letter for spot in column]):
            return True
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all([spot == letter for spot in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all([spot == letter for spot in diagonal2]):
                return True
        return False

    def is_draw(self):
        return ' ' not in self.board

# --------------------  3. Clase del agente --------------------------

class QLearningAgent:
    def __init__(self, alpha=0.001, gamma=0.9, epsilon=0.5):
        self.q_table = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state, available_actions):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)
        q_values = [self.q_table[(state, a)] for a in available_actions]
        max_q_value = max(q_values)
        max_actions = [a for a in available_actions if self.q_table[(state, a)] == max_q_value]
        return random.choice(max_actions)

    def update_q_value(self, state, action, reward, next_state, next_available_actions):
        max_next_q = max([self.q_table[(next_state, a)] for a in next_available_actions], default=0)
        self.q_table[(state, action)] += self.alpha * (reward + self.gamma * max_next_q - self.q_table[(state, action)])

# ----------------  4. Clase del entreno y evaluación periódica del agente -------------------

def train_and_evaluate_agent(episodes=10000):
    env = TicTacToe()
    agent_X = QLearningAgent(alpha=0.001, gamma=0.9, epsilon=0.5)
    agent_O = QLearningAgent(alpha=0.001, gamma=0.9, epsilon=0.5)
    
    eval_interval = int(round(episodes * 0.01))  # Evaluación cada 1% de los episodios
    evaluation_episodes = eval_interval  # Usar el mismo valor para eval_interval y evaluation_episodes
    win_percentages = []
    draw_percentages = []

    for episode in range(episodes):
        state = ''.join(env.reset())
        done = False

        # Alternar quién empieza (agente X o agente O)
        if episode % 2 == 0:
            current_agent, other_agent = agent_X, agent_O
            current_letter, other_letter = 'X', 'O'
        else:
            current_agent, other_agent = agent_O, agent_X
            current_letter, other_letter = 'O', 'X'

        while not done:
            action = current_agent.choose_action(state, env.available_moves())
            env.make_move(action, current_letter)
            next_state = ''.join(env.board)
            
            if env.current_winner == current_letter:
                reward = 100
                current_agent.update_q_value(state, action, reward, next_state, [])
                other_agent.update_q_value(state, action, -100, next_state, [])  # Punishment for losing
                done = True
            elif env.is_draw():
                reward = 50
                current_agent.update_q_value(state, action, reward, next_state, [])
                other_agent.update_q_value(state, action, reward, next_state, [])  # Both agents get same reward
                done = True
            else:
                # Switch to the other agent
                state = next_state
                current_agent, other_agent = other_agent, current_agent
                current_letter, other_letter = other_letter, current_letter
                continue  # Continue to the next move

        # Evaluar el rendimiento del agente cada eval_interval episodios
        if (episode + 1) % eval_interval == 0:
            wins, draws, losses = evaluate_agent(agent_X, games=evaluation_episodes)  # Evaluación del agente X
            win_percentage = (wins / evaluation_episodes) * 100
            draw_percentage = (draws / evaluation_episodes) * 100
            win_percentages.append(win_percentage)
            draw_percentages.append(draw_percentage)

    return win_percentages, draw_percentages, eval_interval

# ---------------  5. Evaluación del rendimiento ---------------------

def evaluate_agent(agent, games=100):
    env = TicTacToe()
    wins, draws, losses = 0, 0, 0

    for _ in range(games):
        state = ''.join(env.reset())
        done = False

        # Alternar quién empieza (agente o oponente aleatorio)
        if _ % 2 == 0:
            agent_letter, opponent_letter = 'X', 'O'
        else:
            agent_letter, opponent_letter = 'O', 'X'

        while not done:
            if agent_letter == 'X':
                action = agent.choose_action(state, env.available_moves())
                env.make_move(action, 'X')
                if env.current_winner == 'X':
                    wins += 1
                    done = True
                elif env.is_draw():
                    draws += 1
                    done = True
                else:
                    opponent_action = random.choice(env.available_moves())
                    env.make_move(opponent_action, 'O')
                    if env.current_winner == 'O':
                        losses += 1
                        done = True
                    elif env.is_draw():
                        draws += 1
                        done = True
            else:
                opponent_action = random.choice(env.available_moves())
                env.make_move(opponent_action, 'X')
                if env.current_winner == 'X':
                    losses += 1
                    done = True
                elif env.is_draw():
                    draws += 1
                    done = True
                else:
                    action = agent.choose_action(state, env.available_moves())
                    env.make_move(action, 'O')
                    if env.current_winner == 'O':
                        wins += 1
                        done = True
                    elif env.is_draw():
                        draws += 1
                        done = True

            state = ''.join(env.board)
    
    return wins, draws, losses

# ------------------  6. Invocar main con Seaborn ------------------------

if __name__ == "__main__":
    episodes = 100000  # Cambia el número según sea necesario
    win_percentages, draw_percentages, eval_interval = train_and_evaluate_agent(episodes)

    # Gráfico del porcentaje de victorias y empates en cada intervalo de evaluación
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Gráfico del porcentaje de victorias
    ax = sns.lineplot(x=range(eval_interval, episodes + 1, eval_interval), y=win_percentages, color="blue", marker="o", linestyle="--", label="Porcentaje de victorias")
    
    # Gráfico del porcentaje de empates
    ax = sns.lineplot(x=range(eval_interval, episodes + 1, eval_interval), y=draw_percentages, color="green", marker="o", linestyle="--", label="Porcentaje de empates")
    
    ax.set_xlabel("Episodios", fontsize=12)
    ax.set_ylabel("Porcentaje", fontsize=12)

    # Añadir título y subtítulo
    plt.title(f"Tic Tac Toe Estándar | Porcentaje de victorias y empates del agente", fontsize=14, fontweight='bold')
    plt.suptitle(f"Evaluación cada {eval_interval} episodios", fontsize=12, style='italic')

    # Guardar la gráfica en lugar de mostrarla
    base_dir = os.path.dirname(os.path.abspath(__file__))
    resultados_dir = os.path.join(base_dir, "resultados")
    if not os.path.exists(resultados_dir):
        os.makedirs(resultados_dir)

    image_path = os.path.join(resultados_dir, f"tictactoe_winpct_drawpct_selftrain_{episodes}_episodes.png")
    plt.savefig(image_path)
    plt.close()

    print(f"Gráfica guardada como {image_path}")
