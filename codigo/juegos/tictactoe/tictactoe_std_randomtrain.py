# -----------------------  TIC TAC TOE: Q-LEARNING STANDARD -----------------------------

# Hecho con ayuda de ChatGPT
# Objetivo: entrenar un agente con Q-learning para que aprenda a jugar tic tac toe
# Funcionamiento: El archivo entrena durante un número de episodios seleccionados jugando contra elección aleatoria,
# y cada 1% del total de episodios evalúa contra un oponente aleatorio (100 juegos) y guarda el porcentaje de victorias.

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

def train_q_learning_agent_random_opponent(episodes=10000):
    env = TicTacToe()
    agent = QLearningAgent(alpha=0.001, gamma=0.9, epsilon=0.5)
    rewards = []
    win_percentages = []
    eval_interval = int(round(episodes * 0.01))  # Evaluación cada 1% de los episodios
    evaluation_episodes = eval_interval  # Usar el mismo valor para eval_interval y evaluation_episodes

    for episode in range(episodes):
        state = ''.join(env.reset())
        done = False
        episode_reward = 0

        while not done:
            # Agente como 'X'
            action = agent.choose_action(state, env.available_moves())
            env.make_move(action, 'X')
            next_state = ''.join(env.board)
            
            if env.current_winner == 'X':
                reward = 100
                agent.update_q_value(state, action, reward, next_state, [])
                done = True
            elif env.is_draw():
                reward = 50
                agent.update_q_value(state, action, reward, next_state, [])
                done = True
            else:
                # Oponente aleatorio como 'O'
                opponent_action = random.choice(env.available_moves())
                env.make_move(opponent_action, 'O')
                if env.current_winner == 'O':
                    reward = -100
                    agent.update_q_value(state, action, reward, next_state, [])
                    done = True
                elif env.is_draw():
                    reward = 50
                    agent.update_q_value(state, action, reward, next_state, [])
                    done = True
                else:
                    state = next_state
                    continue

            episode_reward += reward

        rewards.append(episode_reward)

        # Evaluar el rendimiento del agente cada eval_interval episodios
        if (episode + 1) % eval_interval == 0:
            wins, draws, losses = evaluate_agent(agent, games=evaluation_episodes)  # Evaluación del agente
            win_percentage = (wins / evaluation_episodes) * 100
            win_percentages.append(win_percentage)

    return win_percentages, eval_interval

# ---------------  5. Evaluación del rendimiento ---------------------

def evaluate_agent(agent, games=100):
    env = TicTacToe()
    wins, draws, losses = 0, 0, 0

    for _ in range(games):
        state = ''.join(env.reset())
        done = False

        while not done:
            # Agente juega como 'X'
            action = agent.choose_action(state, env.available_moves())
            env.make_move(action, 'X')
            if env.current_winner == 'X':
                wins += 1
                done = True
            elif env.is_draw():
                draws += 1
                done = True
            else:
                # Oponente aleatorio juega como 'O'
                opponent_action = random.choice(env.available_moves())
                env.make_move(opponent_action, 'O')
                if env.current_winner == 'O':
                    losses += 1
                    done = True
                elif env.is_draw():
                    draws += 1
                    done = True

            state = ''.join(env.board)
    
    return wins, draws, losses

# ------------------  6. Invocar main con Seaborn ------------------------

if __name__ == "__main__":
    episodes = 100000  # Cambia el número según sea necesario
    win_percentages, eval_interval = train_q_learning_agent_random_opponent(episodes)

    # Gráfico del porcentaje de victorias en cada intervalo de evaluación
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Gráfico del porcentaje de victorias
    ax = sns.lineplot(x=range(eval_interval, episodes + 1, eval_interval), y=win_percentages, color="blue", marker="o", linestyle="--", label="Porcentaje de victorias")
    ax.set_xlabel("Episodios", fontsize=12)
    ax.set_ylabel("Porcentaje de victorias", fontsize=12)

    # Añadir título y subtítulo
    plt.title(f"Tic Tac Toe Estándar | Porcentaje de victorias del agente vs elección aleatoria", fontsize=14, fontweight='bold')
    plt.suptitle(f"Evaluación cada {eval_interval} episodios", fontsize=12, style='italic')

    # Guardar la gráfica en lugar de mostrarla
    base_dir = os.path.dirname(os.path.abspath(__file__))
    resultados_dir = os.path.join(base_dir, "resultados")
    if not os.path.exists(resultados_dir):
        os.makedirs(resultados_dir)

    image_path = os.path.join(resultados_dir, f"tictactoe_winpct_random_{episodes}_episodes.png")
    plt.savefig(image_path)
    plt.close()

    print(f"Gráfica guardada como {image_path}")


















