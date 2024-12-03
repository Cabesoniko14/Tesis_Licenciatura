import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from jericho import *

# Configuración
zork_path = "jericho-game-suite/zork1.z5"  # Ruta al archivo Zork
NUM_EPISODES = 10  # Número de episodios de entrenamiento
GAMMA = 0.99  # Factor de descuento
EPSILON = 1.0  # Probabilidad inicial de tomar una acción aleatoria
EPSILON_DECAY = 0.995  # Decaimiento de epsilon
EPSILON_MIN = 0.1  # Valor mínimo de epsilon
LEARNING_RATE = 1e-3  # Tasa de aprendizaje
BATCH_SIZE = 64  # Tamaño del batch para el entrenamiento
MEMORY_SIZE = 10000  # Tamaño máximo del buffer de experiencia
state_size = 512  # Tamaño del vector que representa el estado (ajustado al modelo)

# Crear carpeta de logs
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Nombre del archivo de log
log_file_path = os.path.join(log_dir, f"log_zork_deepqn_standard_{NUM_EPISODES}_episodes.txt")

# Crear el entorno
env = FrotzEnv(zork_path)
vocab = env.get_dictionary()  # Diccionario de palabras válidas

# Generar acciones posibles
def generate_actions(vocab):

    # Combinaciones posibles de acciones
    verbs = ["look", "take", "drop", "open", "close", "go", "attack", "read"]
    directions = ["north", "south", "east", "west", "up", "down"]
    actions = verbs + directions

    # Combinaciones
    for verb in verbs:
        for word in vocab:
            actions.append(f"{verb} {word}")

    return actions

actions = generate_actions(vocab)

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
action_size = len(actions)
policy_net = DQNetwork(state_size, action_size)
target_net = DQNetwork(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())  # Inicializar con los mismos pesos
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Buffer de experiencia
memory = deque(maxlen=MEMORY_SIZE)

# Función para codificar el estado
def encode_state(state):
    encoded = np.zeros(state_size)  # Crea un vector de ceros de tamaño state_size
    hash_value = hash(state) % state_size
    encoded[hash_value] = 1  # Marca una posición basada en el hash
    return encoded

# Función para seleccionar una acción (política epsilon-greedy)
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, action_size - 1)  # Acción aleatoria
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    q_values = policy_net(state_tensor)
    return torch.argmax(q_values).item()

# Función para entrenar la red DQN
def train_dqn():
    if len(memory) < BATCH_SIZE:
        return
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    # Asegurar dimensiones correctas
    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(np.array(next_states))
    dones = torch.FloatTensor(dones)

    # Cálculo de valores Q esperados
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0]
    expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    # Actualización de la red
    loss = criterion(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Entrenamiento
with open(log_file_path, "w") as log_file:  # Abrir archivo de log una sola vez
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        state_text = state  # Guardar el texto inicial del estado
        state = encode_state(state)
        total_reward = 0
        done = False

        log_file.write(f"\n--- Episode {episode + 1} ---\n")

        while not done:
            # Seleccionar una acción
            action_idx = select_action(state, EPSILON)
            action = actions[action_idx]

            # Ejecutar la acción en el entorno
            next_state, reward, done, info = env.step(action)
            next_state_text = next_state  # Guardar el texto del siguiente estado
            next_state = encode_state(next_state)

            # Guardar detalles del episodio en el archivo
            log_file.write(
                f"Action: {action}\nReward: {reward}\nState Text: {next_state_text}\nScore: {info['score']}, Moves: {info['moves']}\n{'=' * 50}\n"
            )

            # Almacenar en el buffer
            memory.append((state, action_idx, reward, next_state, done))
            state = next_state
            state_text = next_state_text
            total_reward += reward

            # Entrenar el modelo
            train_dqn()

        # Actualizar epsilon
        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

        # Actualizar la red objetivo
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Mostrar resumen del episodio
        print(f"Episode {episode + 1}: Total Reward: {total_reward}, Final Score: {info['score']}, Total Moves: {info['moves']}")

        # Guardar resumen del episodio en el archivo de log
        log_file.write(f"Episode {episode + 1} finished. Total Reward: {total_reward}, Final Score: {info['score']}, Total Moves: {info['moves']}\n")

# Guardar el modelo entrenado
torch.save(policy_net.state_dict(), "dqn_zork.pth")
print(f"Entrenamiento completado y modelo guardado en {log_file_path}.")
