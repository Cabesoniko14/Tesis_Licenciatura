# -----------------------  TIC TAC TOE: DQN + LLM vs Aleatorio (acción incluida) -----------------------------
import os
import time
import random
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
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

# --------------------  2. Clase de la red neuronal --------------------------
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )

    def forward(self, x):
        return self.layers(x)

# --------------------  3. Agente DQN + LLM --------------------------
class DQNAgentWithLLM:
    def __init__(self, alpha=0.001, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_decay, self.epsilon_min = epsilon, epsilon_decay, epsilon_min
        self.memory, self.batch_size = deque(maxlen=2000), 32
        self.model = DQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

        # Conexión Groq
        self.client = Groq(api_key="")

    def choose_action(self, state, available_actions):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor).squeeze().detach().numpy()
        best_action = np.argmax(q_values)
        return best_action if best_action in available_actions else random.choice(available_actions)

    def get_llm_evaluation(self, prev_state, action, next_state, agent_letter):
        try:
            response = self.client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": f"El tablero antes de la acción: {prev_state}\n"
                               f"El agente ({agent_letter}) coloca en la posición {action}\n"
                               f"El tablero resultante: {next_state}\n"
                               f"Evalúa esta acción y responde solo con: SUPER BAD, BAD, REGULAR, GOOD, SUPER GOOD."
                }],
                model="llama-3.1-8b-instant"
            )
            llm_value = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLM Error] {e} → fallback REGULAR")
            llm_value = "REGULAR"

        transcription = {"SUPER BAD": -200, "BAD": -100, "REGULAR": 0, "GOOD": 100, "SUPER GOOD": 200}
        return transcription.get(llm_value, 0)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size: return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state_tensor).detach().numpy()
            target_f[0][action] = target
            self.optimizer.zero_grad()
            predictions = self.model(state_tensor)
            loss = self.loss_fn(predictions, torch.FloatTensor(target_f))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay

# --------------------  4. Entrenamiento con LLM + oponente aleatorio --------------------------
def train_dqn_agent_vs_random_llm(num_epochs=5, episodes_per_epoch=10):
    env, agent = TicTacToe(), DQNAgentWithLLM()
    total_rewards, total_time = [], 0
    wins, draws, losses = 0, 0, 0
    total_episodes = num_epochs*episodes_per_epoch

    acciones_df = pd.DataFrame(columns=["Epoch","Episodio","Agente","Acción","Board","Reward","RewardAcum"])
    computo_df = pd.DataFrame(columns=["Epoch","Episodio","Tiempo(s)","CPU(%)","RAM(MB)","GPU_mem(MB)"])
    resumen_df = pd.DataFrame(columns=["Métrica","Valor"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"tictactoe_llm_random_{timestamp}_episodes_{total_episodes}"
    os.makedirs("datos_output", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("modelos", exist_ok=True)
    paths = {
        "acciones": f"datos_output/acciones_{base}.csv",
        "computo": f"datos_output/computo_{base}.csv",
        "resumen": f"datos_output/resumen_{base}.csv",
        "log": f"logs/log_{base}.txt",
        "policy": f"modelos/policy_{base}.pth"
    }

    with open(paths["log"], "w") as logf:
        for epoch in range(num_epochs):
            print(f"\n=== Inicio Epoch {epoch+1}/{num_epochs} ===")
            logf.write(f"\n=== Epoch {epoch+1}/{num_epochs} ===\n")
            for ep in range(episodes_per_epoch):
                start = time.perf_counter()
                board_state = env.reset()
                state = np.array([1 if x == 'X' else -1 if x == 'O' else 0 for x in board_state])
                done, reward_total, steps = False, 0, 0
                agent_letter, opponent_letter = ('X','O') if ep % 2 == 0 else ('O','X')
                print(f"[Epoch {epoch+1} | Ep {ep+1}] Agente: {agent_letter}")
                logf.write(f"[Epoch {epoch+1} | Ep {ep+1}] Agente: {agent_letter}\n")

                while not done:
                    steps += 1
                    prev_state = state.copy()
                    available_actions = env.available_moves()
                    action = agent.choose_action(state, available_actions)
                    env.make_move(action, agent_letter)
                    next_state_list = env.board.copy()
                    next_state = np.array([1 if x == 'X' else -1 if x == 'O' else 0 for x in next_state_list])

                    # Reward basado en LLM evaluando acción
                    reward = agent.get_llm_evaluation(prev_state.tolist(), action, next_state_list, agent_letter)
                    reward_total += reward
                    agent.remember(prev_state, action, reward, next_state, done)

                    acciones_df.loc[len(acciones_df)] = [epoch+1, ep+1, agent_letter, action, next_state_list, reward, reward_total]
                    print(f"  Acción {action} | Reward {reward} | Acum {reward_total}")
                    logf.write(f"  Acción {action} | Reward {reward} | Acum {reward_total}\n")

                    # Movimiento aleatorio del oponente
                    if not env.current_winner and not env.is_draw():
                        opponent_action = random.choice(env.available_moves())
                        env.make_move(opponent_action, opponent_letter)

                    done = env.current_winner is not None or env.is_draw()
                    state = next_state

                    agent.replay()

                # Contabilizar resultados
                if env.current_winner == agent_letter: wins += 1
                elif env.current_winner == opponent_letter: losses += 1
                else: draws += 1

                elapsed = time.perf_counter()-start
                total_time += elapsed
                computo_df.loc[len(computo_df)] = [
                    epoch+1, ep+1, elapsed, psutil.cpu_percent(),
                    psutil.Process(os.getpid()).memory_info().rss/1024/1024,
                    torch.cuda.memory_allocated()/1024/1024 if torch.cuda.is_available() else 0
                ]
                total_rewards.append(reward_total)

    resumen_df.loc[len(resumen_df)] = ["Victorias", wins]
    resumen_df.loc[len(resumen_df)] = ["Empates", draws]
    resumen_df.loc[len(resumen_df)] = ["Derrotas", losses]
    resumen_df.loc[len(resumen_df)] = ["Reward promedio", np.mean(total_rewards)]
    resumen_df.loc[len(resumen_df)] = ["Tiempo total (s)", total_time]
    resumen_df.loc[len(resumen_df)] = ["GPU usada", torch.cuda.is_available()]

    acciones_df.to_csv(paths["acciones"], index=False)
    computo_df.to_csv(paths["computo"], index=False)
    resumen_df.to_csv(paths["resumen"], index=False)
    torch.save(agent.model.state_dict(), paths["policy"])

    print(f"\n=== Entrenamiento completado: logs en {paths['log']} ===")
    return acciones_df, computo_df, resumen_df

# ------------------  5. Main ------------------------
if __name__ == "__main__":
    acciones, computo, resumen = train_dqn_agent_vs_random_llm(num_epochs=10, episodes_per_epoch=10)
