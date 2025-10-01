# ======================  Q-Learning: Tic Tac Toe (Tabular, según paper)  ======================
import os
import time
import random
import psutil
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
import torch

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

# --------------------  2. Agente Q-Learning (igual al del paper) --------------------------
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = defaultdict(float)  # Q(s,a)
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

# --------------------  3. Entrenamiento --------------------------
def train_qlearning_tictactoe(num_epochs=100, episodes_per_epoch=100):
    env = TicTacToe()
    agent = QLearningAgent()

    total_rewards, total_time = [], 0.0
    wins, draws, losses = 0, 0, 0

    # Tablas
    acciones_df = pd.DataFrame(columns=["Epoch","Episodio","Agente","Acción","Board","Reward","RewardAcum"])
    computo_df = pd.DataFrame(columns=["Epoch","Episodio","Tiempo(s)","CPU(%)","RAM(MB)","GPU_mem(MB)"])
    victorias_df = pd.DataFrame(columns=["Epoch","Victorias","Empates","Derrotas","WinRate(%)"])
    resumen_df = pd.DataFrame(columns=["Métrica","Valor"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_episodes = num_epochs * episodes_per_epoch
    base = f"tictactoe_qlearning_std_{timestamp}_episodes_{total_episodes}"
    os.makedirs("datos_output", exist_ok=True)

    for epoch in range(num_epochs):
        wins_epoch, draws_epoch, losses_epoch = 0, 0, 0

        for ep in range(episodes_per_epoch):
            start = time.perf_counter()
            board = env.reset()
            state = board.copy()
            reward_total = 0
            done = False

            # alternar quién empieza
            agent_letter, opponent_letter = ('X','O') if ((epoch*episodes_per_epoch+ep) % 2 == 0) else ('O','X')

            while not done:
                available_actions = env.available_moves()
                action = agent.choose_action(state, available_actions)
                env.make_move(action, agent_letter)
                next_state = env.board.copy()

                # Recompensa (igual que paper: +1 win, -1 loss, 0 otherwise)
                if env.current_winner == agent_letter:
                    reward, done = 1, True
                elif env.current_winner == opponent_letter:
                    reward, done = -1, True
                elif env.is_draw():
                    reward, done = 0, True
                else:
                    reward, done = 0, False

                next_actions = env.available_moves()
                agent.update(state, action, reward, next_state, next_actions)

                reward_total += reward
                acciones_df.loc[len(acciones_df)] = [epoch+1, ep+1, agent_letter, action, next_state.copy(), reward, reward_total]

                if not done:
                    opp_actions = env.available_moves()
                    if opp_actions:
                        opp_action = random.choice(opp_actions)
                        env.make_move(opp_action, opponent_letter)
                        if env.current_winner == opponent_letter:
                            reward, done = -1, True
                        elif env.is_draw():
                            reward, done = 0, True

                state = env.board.copy()

            if env.current_winner == agent_letter:
                wins += 1; wins_epoch += 1
            elif env.current_winner == opponent_letter:
                losses += 1; losses_epoch += 1
            else:
                draws += 1; draws_epoch += 1

            total_rewards.append(reward_total)
            total_time += time.perf_counter() - start

            computo_df.loc[len(computo_df)] = [
                epoch+1, ep+1, time.perf_counter()-start,
                psutil.cpu_percent(),
                psutil.Process(os.getpid()).memory_info().rss/1024/1024,
                torch.cuda.memory_allocated()/1024/1024 if torch.cuda.is_available() else 0
            ]

        win_rate_epoch = (wins_epoch / episodes_per_epoch) * 100
        victorias_df.loc[len(victorias_df)] = [epoch+1, wins_epoch, draws_epoch, losses_epoch, win_rate_epoch]
        print(f"=== Epoch {epoch+1}/{num_epochs} terminado | winrate={win_rate_epoch:.2f}% ===")

    # Resumen global
    resumen_df.loc[len(resumen_df)] = ["Victorias", wins]
    resumen_df.loc[len(resumen_df)] = ["Empates", draws]
    resumen_df.loc[len(resumen_df)] = ["Derrotas", losses]
    resumen_df.loc[len(resumen_df)] = ["Reward promedio", np.mean(total_rewards) if total_rewards else 0]
    resumen_df.loc[len(resumen_df)] = ["Tiempo total (s)", total_time]
    resumen_df.loc[len(resumen_df)] = ["GPU usada", torch.cuda.is_available()]

    # Guardar resultados
    acciones_df.to_csv(f"datos_output/acciones_{base}.csv", index=False)
    computo_df.to_csv(f"datos_output/computo_{base}.csv", index=False)
    victorias_df.to_csv(f"datos_output/victorias_{base}.csv", index=False)
    resumen_df.to_csv(f"datos_output/resumen_{base}.csv", index=False)

    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    return acciones_df, computo_df, victorias_df, resumen_df

# --------------------  Main --------------------------
if __name__ == "__main__":
    acciones, computo, victorias, resumen = train_qlearning_tictactoe(
        num_epochs=100, episodes_per_epoch=100
    )
