import os
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from groq import Groq

# --------------------  1. Clase del juego --------------------------

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

# --------------------  2. Clase del agente con LLM --------------------------

class QLearningAgentWithLLM:
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

    def get_llm_evaluation(self, state, agent_letter):
        # Use Groq LLM to get the qualitative evaluation of the state
        client = Groq(
            api_key = "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"
        )

        response = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"Evaluate the following Tic Tac Toe board state and respond with one of the following: SUPER BAD, BAD, REGULAR, GOOD, SUPER GOOD. Board: {state}. The user you are evaluating is using {agent_letter}. Remember to JUST reply one of the options. Not any more words not any explanation."
            }],
            model="llama3-8b-8192"
        )
        
        llm_value = response.choices[0].message.content.strip()
        print(f"LLM Response: {llm_value}")  # Print the LLM's response
        return self.transcribe_llm_value(llm_value)

    def transcribe_llm_value(self, llm_value):
        # Convert qualitative response to numeric value
        transcription = {
            "SUPER BAD": -200,
            "BAD": -100,
            "REGULAR": 0,
            "GOOD": 100,
            "SUPER GOOD": 200
        }
        return transcription.get(llm_value, 0)  # Default to 0 if not recognized

    def update_q_value(self, state, action, next_state, agent_letter, next_available_actions):
        # Use LLM to get the qualitative evaluation of the next state
        reward = self.get_llm_evaluation(next_state, agent_letter)
        max_next_q = max([self.q_table[(next_state, a)] for a in next_available_actions], default=0)
        self.q_table[(state, action)] += self.alpha * (reward + self.gamma * max_next_q - self.q_table[(state, action)])

# ----------------  3. Clase del entreno y evaluación periódica del agente -------------------

def train_q_learning_agent_with_llm(episodes=10000):
    env = TicTacToe()
    agent = QLearningAgentWithLLM(alpha=0.001, gamma=0.9, epsilon=0.5)
    rewards = []
    win_percentages = []
    draw_percentages = []
    eval_interval = int(round(episodes * 0.01))  # Evaluación cada 1% de los episodios
    evaluation_episodes = 100  # Usar el mismo valor para eval_interval y evaluation_episodes

    for episode in range(episodes):
        print(f"Starting Episode {episode + 1}")  # Print when a new episode starts
        state = ''.join(env.reset())
        done = False
        episode_reward = 0

        # Alternar entre empezar primero y segundo
        if episode % 2 == 0:
            agent_letter, opponent_letter = 'X', 'O'
        else:
            agent_letter, opponent_letter = 'O', 'X'

        while not done:
            next_state = ''.join(env.board)

            if agent_letter == 'X':
                action = agent.choose_action(state, env.available_moves())
                env.make_move(action, 'X')
                next_state = ''.join(env.board)
                if env.current_winner == 'X' or env.is_draw():
                    done = True
                else:
                    opponent_action = random.choice(env.available_moves())
                    env.make_move(opponent_action, 'O')
                    next_state = ''.join(env.board)
                    if env.current_winner == 'O' or env.is_draw():
                        done = True
            else:
                opponent_action = random.choice(env.available_moves())
                env.make_move(opponent_action, 'X')
                next_state = ''.join(env.board)
                if env.current_winner == 'X' or env.is_draw():
                    done = True
                else:
                    action = agent.choose_action(state, env.available_moves())
                    env.make_move(action, 'O')
                    next_state = ''.join(env.board)
                    if env.current_winner == 'O' or env.is_draw():
                        done = True

            # Update Q-value using LLM's assessment
            agent.update_q_value(state, action, next_state, agent_letter, env.available_moves())
            state = next_state

        # Evaluar el rendimiento del agente cada eval_interval episodios
        if (episode + 1) % eval_interval == 0:
            wins, draws, losses = evaluate_agent(agent, games=evaluation_episodes)
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
    episodes = 1000  # Change the number according to your needs
    win_percentages, draw_percentages, eval_interval = train_q_learning_agent_with_llm(episodes)

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
    plt.title(f"Tic Tac Toe con LLM (Groq) | Porcentaje de victorias y empates del agente vs elección aleatoria", fontsize=14, fontweight='bold')
    plt.suptitle(f"Evaluación cada {eval_interval} episodios", fontsize=12, style='italic')

    # Guardar la gráfica en lugar de mostrarla
    base_dir = os.path.dirname(os.path.abspath(__file__))
    resultados_dir = os.path.join(base_dir, "resultados")
    if not os.path.exists(resultados_dir):
        os.makedirs(resultados_dir)

    image_path = os.path.join(resultados_dir, f"tictactoe_winpct_groq_random_{episodes}_episodes.png")
    plt.savefig(image_path)
    plt.close()

    print(f"Gráfica guardada como {image_path}")
