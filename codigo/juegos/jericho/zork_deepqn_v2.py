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
import psutil
import gc

# Configuración
zork_path = "jericho-game-suite/zork1.z5"  # Ruta al archivo Zork
NUM_EPOCHS = 50  # Número de epochs
EPISODES_PER_EPOCH = 20  # Número de episodios por epoch
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
log_file_path = os.path.join(log_dir, f"log_zork_deepqnv2_{timestamp}_epochs_{NUM_EPOCHS}.txt")

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

# Función para registrar memoria usada
def log_memory():
    print(f"Memoria RAM usada: {psutil.virtual_memory().used / 1e9:.2f} GB")
    print(f"Memoria GPU usada: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    gc.collect()

# Entrenamiento
episode_rewards = []
epoch_end_points = []

with open(log_file_path, "w") as log_file:
    for epoch in range(NUM_EPOCHS):
        print(f"\n----------------- EPOCH {epoch + 1} -----------------")
        log_file.write(f"\n----------------- EPOCH {epoch + 1} -----------------\n")

        # Reiniciar el entorno al inicio de cada epoch
        if env is not None:
            env.close()
        env = FrotzEnv(zork_path)

        epoch_total_reward = 0
        for episode in range(EPISODES_PER_EPOCH):
            episode_number = epoch * EPISODES_PER_EPOCH + episode + 1
            print(f"Procesando episodio {episode_number}...")
            log_file.write(f"--- Inicio del Episodio {episode_number} ---\n")

            state, _ = env.reset()
            state = encode_state(state)
            total_reward = 0
            done = False
            steps = 0  # Contador de pasos
            action_history = []  # Historial de acciones

            while not done and steps < 100:  # Límite de pasos
                steps += 1
                valid_actions = get_valid_actions(env)
                action = select_action(state, env, EPSILON)
                if action is None:
                    log_file.write("No hay acciones válidas. Terminando episodio.\n")
                    break

                # Verificar acciones repetitivas
                if len(action_history) > 10 and all(a == action for a in action_history[-10:]):
                    print("Acciones repetitivas detectadas. Terminando episodio.")
                    break
                action_history.append(action)

                next_state, reward, done, info = env.step(action)
                next_state = encode_state(next_state)

                memory.append((state, valid_actions.index(action), reward, next_state, done))
                state = next_state
                total_reward += reward

                log_file.write(f"Acción: {action}, Recompensa: {reward}, Info: {info}\n")

                train_dqn()

            EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
            episode_rewards.append(total_reward)
            epoch_total_reward += total_reward

            log_file.write(f"--- Fin del Episodio {episode_number} ---\n")
            log_file.write(f"Recompensa Total: {total_reward}\n")
            log_file.write("-" * 50 + "\n")

        # Marcar el final del epoch
        epoch_end_points.append(len(episode_rewards))

        epoch_average_reward = epoch_total_reward / EPISODES_PER_EPOCH
        print(f"Epoch {epoch + 1} completado. Recompensa Promedio: {epoch_average_reward}")
        log_file.write(f"--- Fin del EPOCH {epoch + 1} ---\n")
        log_file.write(f"Recompensa Promedio del EPOCH: {epoch_average_reward}\n")
        log_file.write("=" * 50 + "\n")

        # Actualizar red objetivo periódicamente
        if (epoch + 1) % 5 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Liberar memoria
        torch.cuda.empty_cache()
        gc.collect()
        log_memory()  # Registrar uso de memoria

# Guardar el modelo entrenado
torch.save(policy_net.state_dict(), os.path.join(models_dir, f"model__zork_deepqnv2_{timestamp}_epochs_{NUM_EPOCHS}.pth"))

# Crear la gráfica de resultados
plt.figure(figsize=(10, 6))
plt.title(f"Resultados del experimento Zork - {NUM_EPOCHS} epochs")
plt.xlabel("Episodio")
plt.ylabel("Recompensa Total")

plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, color="green", label="Recompensa por Episodio")
for epoch_end in epoch_end_points:
    plt.axvline(x=epoch_end, color="blue", linestyle="--", alpha=0.5, label="Fin de Epoch" if epoch_end == epoch_end_points[0] else "")

plt.legend()
plt.savefig(os.path.join(results_dir, f"viz_zork_deepqnv2_{timestamp}_epochs_{NUM_EPOCHS}.png"))
plt.close()

print(f"Gráfica guardada en {results_dir}.")
