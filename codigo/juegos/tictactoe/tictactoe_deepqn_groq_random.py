# -----------------------  TIC TAC TOE: DQN + LLM + Target Net (Groq) -----------------------------
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

# --------------------  3. Agente DQN + Target Net + Groq LLM --------------------------
class DQNAgentWithLLM:
    def __init__(self, alpha=0.001, gamma=0.9, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.1, tau=0.01):
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_decay, self.epsilon_min = epsilon, epsilon_decay, epsilon_min
        self.memory, self.batch_size = deque(maxlen=2000), 32

        # Redes local (policy) y target
        self.model = DQN()
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.tau = tau  # factor para soft update (opcional)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

        # Conexión Groq (llena tu api_key si quieres usarla)
        self.client = Groq(api_key="")

    def soft_update(self):
        # actualización suave: target <- tau*local + (1-tau)*target
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def hard_update(self):
        # actualización dura: copiar pesos directamente
        self.target_model.load_state_dict(self.model.state_dict())

    def choose_action(self, state, available_actions):
        # state: vector de 9 valores
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor).squeeze().detach().numpy()
        best_action = int(np.argmax(q_values))
        return best_action if best_action in available_actions else random.choice(available_actions)

    def get_llm_evaluation(self, prev_state, action, next_state, agent_letter):
        # Evalúa la acción con Groq LLM; si falla devuelve "REGULAR"
        try:
            response = self.client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": (
                        f"Eres la función de recompensas de un agente de RL que juega Tic Tac Toe.\n"
                        f"Tablero antes: {prev_state}\n"
                        f"Acción tomada (posición): {action} por {agent_letter}\n"
                        f"Tablero después: {next_state}\n"
                        f"Responde SOLO con una de: SUPER BAD, BAD, REGULAR, GOOD, SUPER GOOD."
                    )
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
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                # usar target_model para estabilidad
                target = reward + self.gamma * torch.max(self.target_model(next_state_tensor)).item()
            target_f = self.model(state_tensor).detach().numpy()
            target_f[0][action] = target
            states.append(state_tensor)
            targets.append(torch.FloatTensor(target_f))
            # entrenamiento por cada muestra (sencillo y explícito)
            self.optimizer.zero_grad()
            predictions = self.model(state_tensor)
            loss = self.loss_fn(predictions, torch.FloatTensor(target_f))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # actualizar target con soft update
        self.soft_update()

# --------------------  4. Entrenamiento por epochs y tracking completo --------------------------
def train_dqn_agent_vs_random_llm(num_epochs=100, episodes_per_epoch=20, hard_update_every_epochs=1):
    env = TicTacToe()
    agent = DQNAgentWithLLM()
    total_rewards = []
    total_time = 0.0

    # conteos globales
    wins, draws, losses = 0, 0, 0

    # DataFrames para guardar resultados
    acciones_df = pd.DataFrame(columns=["Epoch", "Episodio", "Agente", "Acción", "Board", "Reward", "RewardAcum"])
    computo_df = pd.DataFrame(columns=["Epoch", "Episodio", "Tiempo(s)", "CPU(%)", "RAM(MB)", "GPU_mem(MB)"])
    resumen_df = pd.DataFrame(columns=["Métrica", "Valor"])
    victorias_df = pd.DataFrame(columns=["Epoch", "Victorias", "Empates", "Derrotas", "WinRate(%)"])

    # Rutas y directorios
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_episodes = num_epochs * episodes_per_epoch
    base = f"tictactoe_groq_{timestamp}_episodes_{total_episodes}"
    os.makedirs("datos_output", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("modelos", exist_ok=True)
    paths = {
        "acciones": f"datos_output/acciones_{base}.csv",
        "computo": f"datos_output/computo_{base}.csv",
        "resumen": f"datos_output/resumen_{base}.csv",
        "victorias": f"datos_output/victorias_{base}.csv",
        "log": f"logs/log_{base}.txt",
        "policy": f"modelos/policy_{base}.pth",
        "target": f"modelos/target_{base}.pth"
    }

    # archivo de log
    with open(paths["log"], "w") as logf:
        for epoch in range(num_epochs):
            print(f"\n=== Inicio Epoch {epoch+1}/{num_epochs} ===")
            logf.write(f"\n=== Epoch {epoch+1}/{num_epochs} ===\n")
            wins_epoch, draws_epoch, losses_epoch = 0, 0, 0

            for ep in range(episodes_per_epoch):
                start = time.perf_counter()
                board = env.reset()
                state = np.array([1 if x == 'X' else -1 if x == 'O' else 0 for x in board])
                done = False
                reward_total = 0
                # alternar quién empieza por episodio (mejora exploración)
                agent_letter, opponent_letter = ('X', 'O') if ((epoch * episodes_per_epoch + ep) % 2 == 0) else ('O', 'X')

                print(f"[Epoch {epoch+1} | Ep {ep+1}] Agente: {agent_letter}")
                logf.write(f"[Epoch {epoch+1} | Ep {ep+1}] Agente: {agent_letter}\n")

                steps = 0
                while not done:
                    steps += 1
                    prev_state = state.copy()
                    available_actions = env.available_moves()
                    action = agent.choose_action(state, available_actions)
                    env.make_move(action, agent_letter)
                    next_board = env.board.copy()
                    next_state = np.array([1 if x == 'X' else -1 if x == 'O' else 0 for x in next_board])

                    # recompensa por LLM
                    reward = agent.get_llm_evaluation(prev_state.tolist(), action, next_board, agent_letter)
                    reward_total += reward
                    agent.remember(prev_state, action, reward, next_state, done)

                    # registrar acción
                    acciones_df.loc[len(acciones_df)] = [epoch+1, ep+1, agent_letter, action, next_board.copy(), reward, reward_total]
                    logf.write(f"  Acción {action} | Reward {reward} | Acum {reward_total}\n")
                    print(f"  Acción {action} | Reward {reward} | Acum {reward_total}")

                    # oponente aleatorio mueve (si el juego no terminó)
                    if not env.current_winner and not env.is_draw():
                        # comprobar si quedan movimientos
                        opp_moves = env.available_moves()
                        if len(opp_moves) > 0:
                            opp_action = random.choice(opp_moves)
                            env.make_move(opp_action, opponent_letter)

                    done = env.current_winner is not None or env.is_draw()
                    state = next_state
                    agent.replay()

                # contabilizar resultado del episodio
                if env.current_winner == agent_letter:
                    wins += 1
                    wins_epoch += 1
                elif env.current_winner == opponent_letter:
                    losses += 1
                    losses_epoch += 1
                else:
                    draws += 1
                    draws_epoch += 1

                elapsed = time.perf_counter() - start
                total_time += elapsed
                computo_df.loc[len(computo_df)] = [
                    epoch+1, ep+1, elapsed, psutil.cpu_percent(),
                    psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
                    torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
                ]
                total_rewards.append(reward_total)
                logf.write(f"Episodio terminado | Total Reward: {reward_total} | Éxito: {1 if env.current_winner==agent_letter else 0} | Pasos: {steps}\n")
                print(f"Episodio terminado | Total Reward: {reward_total} | Winner: {env.current_winner} | Pasos: {steps}")

            # actualizar target (hard update cada N epochs)
            if hard_update_every_epochs > 0 and ((epoch + 1) % hard_update_every_epochs == 0):
                agent.hard_update()
                logf.write("Target network hard-updated from policy network.\n")
                print("Target network hard-updated from policy network.")

            # guardar métricas por epoch
            win_rate_epoch = (wins_epoch / episodes_per_epoch) * 100
            victorias_df.loc[len(victorias_df)] = [epoch+1, wins_epoch, draws_epoch, losses_epoch, win_rate_epoch]
            logf.write(f"Epoch {epoch+1} summary: wins={wins_epoch}, draws={draws_epoch}, losses={losses_epoch}, win_rate={win_rate_epoch:.2f}%\n")
            print(f"=== Fin Epoch {epoch+1}: win_rate={win_rate_epoch:.2f}% ===")

    # resumen global
    resumen_df.loc[len(resumen_df)] = ["Victorias", wins]
    resumen_df.loc[len(resumen_df)] = ["Empates", draws]
    resumen_df.loc[len(resumen_df)] = ["Derrotas", losses]
    resumen_df.loc[len(resumen_df)] = ["Reward promedio", np.mean(total_rewards) if len(total_rewards) > 0 else 0]
    resumen_df.loc[len(resumen_df)] = ["Tiempo total (s)", total_time]
    resumen_df.loc[len(resumen_df)] = ["GPU usada", torch.cuda.is_available()]

    # guardar CSVs y modelos (policy y target)
    acciones_df.to_csv(paths["acciones"], index=False)
    computo_df.to_csv(paths["computo"], index=False)
    resumen_df.to_csv(paths["resumen"], index=False)
    victorias_df.to_csv(paths["victorias"], index=False)

    # guardar ambos modelos
    torch.save(agent.model.state_dict(), paths["policy"])
    torch.save(agent.target_model.state_dict(), paths["target"])

    print(f"\n=== Entrenamiento completado: logs en {paths['log']} ===")
    print(f"Modelos guardados: {paths['policy']} , {paths['target']}")
    return acciones_df, computo_df, resumen_df, victorias_df

# ------------------  5. Main ------------------------
if __name__ == "__main__":
    # ejemplo: 10 epochs x 10 episodios por epoch, hard update (copia) del target cada epoch
    acciones, computo, resumen, victorias = train_dqn_agent_vs_random_llm(
        num_epochs=500, episodes_per_epoch=20, hard_update_every_epochs=1
    )
