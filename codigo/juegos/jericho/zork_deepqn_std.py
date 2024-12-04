import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from jericho import *
from datetime import datetime
import matplotlib.pyplot as plt

# Configuración
zork_path = "jericho-game-suite/zork1.z5"  # Ruta al archivo Zork
NUM_EPISODES = 1000  # Número de episodios de entrenamiento
GAMMA = 0.99  # Factor de descuento
EPSILON = 1.0  # Probabilidad inicial de tomar una acción aleatoria
EPSILON_DECAY = 0.995  # Decaimiento de epsilon
EPSILON_MIN = 0.1  # Valor mínimo de epsilon
LEARNING_RATE = 1e-3  # Tasa de aprendizaje
BATCH_SIZE = 64  # Tamaño del batch para el entrenamiento
MEMORY_SIZE = 10000  # Tamaño máximo del buffer de experiencia
state_size = 512  # Tamaño del vector que representa el estado (ajustado al modelo)

# Verificar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Crear carpetas de logs, resultados y modelos
log_dir = "logs"
results_dir = "resultados"
models_dir = "modelos"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Generar un nombre único para el archivo de logs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(log_dir, f"log_zork_deepqn_{timestamp}_{NUM_EPISODES}_episodes.txt")

# Crear el entorno
env = FrotzEnv(zork_path)

# Obtener las acciones válidas del estado actual
def get_valid_actions(env):
    valid_actions = env.get_valid_actions()
    return [action['action'] if isinstance(action, dict) and 'action' in action else str(action) for action in valid_actions]

# Definir el modelo DQN
class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Inicializar la red, optimizador y pérdida
policy_net = DQNetwork(state_size, state_size).to(device)
target_net = DQNetwork(state_size, state_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Buffer de experiencia
memory = deque(maxlen=MEMORY_SIZE)

# Función para codificar el estado
def encode_state(state):
    encoded = np.zeros(state_size)
    hash_value = hash(state) % state_size
    encoded[hash_value] = 1
    return encoded

# Función para seleccionar una acción (política epsilon-greedy)
def select_action(state, env, epsilon):
    valid_actions = get_valid_actions(env)
    if len(valid_actions) == 0:
        return None
    if random.random() < epsilon:
        return random.choice(valid_actions)  # Acción aleatoria válida

    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    q_values = policy_net(state_tensor)

    valid_indices = {action: idx for idx, action in enumerate(valid_actions)}
    q_values_dict = {action: q_values[0, valid_indices[action]].item() for action in valid_actions}
    return max(q_values_dict, key=q_values_dict.get)

# Función para entrenar la red DQN
def train_dqn():
    if len(memory) < BATCH_SIZE:
        return
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor(dones).to(device)

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0]
    expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    loss = criterion(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Entrenamiento
total_rewards = []
episode_status = []  # 0: estancado/muerto, 1: exitoso

with open(log_file_path, "w") as log_file:
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        state = encode_state(state)
        total_reward = 0
        done = False
        successful = False

        log_file.write(f"\n--- Inicio del Episodio {episode + 1} ---\n")

        while not done:
            valid_actions = get_valid_actions(env)
            action = select_action(state, env, EPSILON)
            if action is None:
                log_file.write("No hay acciones válidas. Terminando episodio.\n")
                break

            next_state, reward, done, info = env.step(action)
            next_state = encode_state(next_state)

            memory.append((state, valid_actions.index(action), reward, next_state, done))
            state = next_state
            total_reward += reward

            # Registrar estado y acción en el log
            log_file.write(f"Acción: {action}, Recompensa: {reward}, Info: {info}\n")

            if done and reward > 0:
                successful = True

            train_dqn()

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
        total_rewards.append(total_reward)
        episode_status.append(1 if successful else 0)

        # Resumen del episodio
        log_file.write(f"--- Fin del Episodio {episode + 1} ---\n")
        log_file.write(f"Recompensa Total: {total_reward}\n")
        log_file.write(f"Estado final: {'Exitoso' if successful else 'Estancado/Muerto'}\n")
        log_file.write("-" * 50 + "\n")

        print(f"Episode {episode + 1}: Total Reward: {total_reward}, Status: {'Success' if successful else 'Stuck/Dead'}")

# Guardar el modelo entrenado
torch.save(policy_net.state_dict(), os.path.join(models_dir, f"model_zork_deepqn_{timestamp}_{NUM_EPISODES}_episodes.pth"))

# Crear la gráfica de resultados
plt.figure(figsize=(10, 6))
plt.title(f"Resultados del experimento Zork - {NUM_EPISODES} episodios")
plt.xlabel("Episodio")
plt.ylabel("Recompensa Total")

for i, reward in enumerate(total_rewards):
    plt.scatter(i + 1, reward, color="red" if episode_status[i] == 0 else "blue")

plt.plot(range(1, NUM_EPISODES + 1), total_rewards, color="green")
plt.savefig(os.path.join(results_dir, f"viz_zork_deepqn_{timestamp}_{NUM_EPISODES}_episodes.png"))
plt.close()

print(f"Gráfica guardada en {results_dir}.")
