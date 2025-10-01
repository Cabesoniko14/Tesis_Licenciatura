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

# --------------------  2. Clase de la red neuronal (más grande) --------------------------
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(9, 256),  # antes 128
            nn.ReLU(),
            nn.Linear(256, 256),  # antes 64
            nn.ReLU(),
            nn.Linear(256, 9)
        )

    def forward(self, x):
        return self.layers(x)

# --------------------  3. Agente DQN + Target Net + Groq LLM --------------------------
class DQNAgentWithLLM:
    def __init__(self, alpha=0.0005, gamma=0.95, epsilon=1.0,
                 epsilon_decay=0.9995, epsilon_min=0.05, tau=0.01):
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_decay, self.epsilon_min = epsilon, epsilon_decay, epsilon_min
        self.memory, self.batch_size = deque(maxlen=50000), 128

        # Redes local (policy) y target
        self.model = DQN()
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.tau = tau

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()

        # Conexión Groq (rellenar tu api_key)
        self.client = Groq(api_key="")

    def soft_update(self):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def hard_update(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def choose_action(self, state, available_actions):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor).squeeze().detach().numpy()
        best_action = int(np.argmax(q_values))
        return best_action if best_action in available_actions else random.choice(available_actions)

    def get_llm_evaluation(self, prev_state, action, next_state, agent_letter):
        try:
            response = self.client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": (
                        "Eres la función de recompensas de un agente de aprendizaje por refuerzo que juega Tic Tac Toe.\n"
                        "Evalúa la calidad de la acción tomada con base en:\n"
                        "- Recompensar altamente una victoria inmediata.\n"
                        "- Penalizar fuertemente una derrota o una jugada que permite que el oponente gane en el siguiente turno.\n"
                        "- Recompensar jugadas que bloquean al oponente o crean oportunidades de ganar.\n"
                        "- Considerar REGULAR jugadas neutrales sin progreso claro.\n\n"
                        f"Tablero antes: {prev_state}\n"
                        f"Acción del agente ({agent_letter}) en posición: {action}\n"
                        f"Tablero después: {next_state}\n\n"
                        "Responde SOLO con una palabra: SUPER BAD, BAD, REGULAR, GOOD, SUPER GOOD."
                    )
                }],
                model="llama-3.1-8b-instant"
            )
            llm_value = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLM Error] {e} → fallback REGULAR")
            llm_value = "REGULAR"

        transcription = {"SUPER BAD": -500, "BAD": -100, "REGULAR": 0, "GOOD": 100, "SUPER GOOD": 500}
        return transcription.get(llm_value, 0)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Q-values actuales
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN
        next_actions = torch.argmax(self.model(next_states), dim=1)
        next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

        expected = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, expected.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.soft_update()

# --------------------  4. Entrenamiento --------------------------
def train_dqn_agent_vs_random_llm(num_epochs=1000, episodes_per_epoch=100, hard_update_every_epochs=5):
    env = TicTacToe()
    agent = DQNAgentWithLLM()
    total_rewards, total_time = [], 0.0
    wins, draws, losses = 0, 0, 0

    victorias_df = pd.DataFrame(columns=["Epoch", "Victorias", "Empates", "Derrotas", "WinRate(%)"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_episodes = num_epochs * episodes_per_epoch
    base = f"tictactoe_groq_{timestamp}_episodes_{total_episodes}"

    os.makedirs("modelos", exist_ok=True)

    for epoch in range(num_epochs):
        wins_epoch, draws_epoch, losses_epoch = 0, 0, 0
        for ep in range(episodes_per_epoch):
            start = time.perf_counter()
            board = env.reset()
            state = np.array([1 if x == 'X' else -1 if x == 'O' else 0 for x in board])
            done, reward_total = False, 0
            agent_letter, opponent_letter = ('X','O') if ((epoch*episodes_per_epoch+ep) % 2 == 0) else ('O','X')

            while not done:
                prev_state = state.copy()
                available_actions = env.available_moves()
                action = agent.choose_action(state, available_actions)
                env.make_move(action, agent_letter)
                next_board = env.board.copy()
                next_state = np.array([1 if x == 'X' else -1 if x == 'O' else 0 for x in next_board])

                reward = agent.get_llm_evaluation(prev_state.tolist(), action, next_board, agent_letter)
                reward_total += reward
                agent.remember(prev_state, action, reward, next_state, done)

                if not env.current_winner and not env.is_draw():
                    opp_moves = env.available_moves()
                    if opp_moves: env.make_move(random.choice(opp_moves), opponent_letter)

                done = env.current_winner is not None or env.is_draw()
                state = next_state
                agent.replay()

            if env.current_winner == agent_letter:
                wins += 1; wins_epoch += 1
            elif env.current_winner == opponent_letter:
                losses += 1; losses_epoch += 1
            else:
                draws += 1; draws_epoch += 1

            total_rewards.append(reward_total)
            total_time += time.perf_counter()-start

        if (epoch+1) % hard_update_every_epochs == 0:
            agent.hard_update()
            print(f"[Epoch {epoch+1}] Hard update de target network.")

        win_rate_epoch = (wins_epoch / episodes_per_epoch) * 100
        victorias_df.loc[len(victorias_df)] = [epoch+1, wins_epoch, draws_epoch, losses_epoch, win_rate_epoch]
        print(f"=== Fin Epoch {epoch+1}: win_rate={win_rate_epoch:.2f}% | eps={agent.epsilon:.3f}")

    torch.save(agent.model.state_dict(), f"modelos/policy_{base}.pth")
    torch.save(agent.target_model.state_dict(), f"modelos/target_{base}.pth")
    victorias_df.to_csv(f"victorias_{base}.csv", index=False)

    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    return victorias_df

# ------------------  5. Main ------------------------
if __name__ == "__main__":
    victorias = train_dqn_agent_vs_random_llm(
        num_epochs=1000, episodes_per_epoch=100, hard_update_every_epochs=5
    )
