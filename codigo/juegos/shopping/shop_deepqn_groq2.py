import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from datetime import datetime
import time
import csv
import psutil
from groq import Groq

# ===== Environment (igual que antes) =====
class ShoppingEnv:
    def __init__(self):
        self.aisles = {
            "Dairy": ["milk", "cheese", "yogurt", "butter"],
            "Bakery": ["bread", "cake", "croissant", "bagel"],
            "Produce": ["apples", "bananas", "carrots", "lettuce"]
        }
        self.all_items = [item for aisle_items in self.aisles.values() for item in aisle_items]
        self.shopping_list = []
        self.current_location = "Entrance"
        self.collected_items = []
        self.step_count = 0
        self.max_steps = 15

    def reset(self):
        self.current_location = "Entrance"
        self.collected_items = []
        self.step_count = 0
        num_items = random.randint(2, 3)
        self.shopping_list = random.sample(self.all_items, num_items)
        return self.get_state()

    def get_state(self):
        return {
            "location": self.current_location,
            "collected_items": self.collected_items,
            "remaining_items": [item for item in self.shopping_list if item not in self.collected_items]
        }

    def step(self, action):
        self.step_count += 1
        reward, done, info = 0, False, ""

        if action.startswith("go to"):
            location = action.split("go to ")[1]
            if location in self.aisles:
                self.current_location = location
                info = f"Moved to {location}"
            else:
                reward, info = -1, "Invalid location"

        elif action.startswith("take"):
            item = action.split("take ")[1]
            if self.current_location in self.aisles and item in self.aisles[self.current_location]:
                if item not in self.collected_items:
                    self.collected_items.append(item)
                    reward, info = 10, f"Collected {item}"
                else:
                    reward, info = -1, f"Already collected {item}"
            else:
                reward, info = -1, f"{item} not found in {self.current_location}"

        elif action == "checkout":
            if set(self.collected_items) == set(self.shopping_list):
                reward, done, info = 50, True, "Checked out successfully"
            else:
                reward, info = -10, "Shopping list not complete"

        else:
            reward, info = -1, "Invalid action"

        if self.step_count >= self.max_steps:
            done, info = True, "Max steps reached"

        return self.get_state(), reward, done, info

    def get_valid_actions(self):
        actions = [f"go to {aisle}" for aisle in self.aisles]
        if self.current_location in self.aisles:
            actions.extend([f"take {item}" for item in self.aisles[self.current_location]])
        actions.append("checkout")
        return actions

# ===== LLM evaluator =====
client = Groq(api_key="YOUR_API_KEY")
def evaluate_with_llm(state, action):
    message = (f"Evaluate action. State: {state}. Action: {action}. Respond with SUPER BAD, BAD, REGULAR, GOOD, SUPER GOOD.")
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": message}],
        model="llama3-8b-8192"
    )
    llm_value = response.choices[0].message.content.strip()
    return {"SUPER BAD": -4, "BAD": -2, "REGULAR": 0, "GOOD": 2, "SUPER GOOD": 4}.get(llm_value, 0)

# ===== DQN =====
class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def encode_state(state):
    state_vector = np.zeros(state_size)
    hash_value = hash(str(state)) % state_size
    state_vector[hash_value] = 1
    return state_vector

def select_action(state, valid_actions, epsilon):
    if random.random() < epsilon:
        return random.choice(valid_actions)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    q_values = policy_net(state_tensor)
    action_q_values = {action: q_values[0, idx].item() for idx, action in enumerate(valid_actions)}
    return max(action_q_values, key=action_q_values.get)

def train_dqn():
    if len(memory) < BATCH_SIZE: return
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

# ===== Config =====
NUM_EPOCHS, EPISODES_PER_EPOCH = 5, 10
TOTAL_EPISODES = NUM_EPOCHS * EPISODES_PER_EPOCH
GAMMA, EPSILON, EPSILON_DECAY, EPSILON_MIN = 0.99, 1.0, 0.995, 0.1
LEARNING_RATE, BATCH_SIZE, MEMORY_SIZE, state_size = 1e-3, 64, 10000, 512

# ===== Directories =====
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_name = f"shopping_{timestamp}_episodes_{TOTAL_EPISODES}"
dirs = {name: os.path.join(name) for name in ["logs", "resultados", "modelos", "datos_output"]}
for d in dirs.values(): os.makedirs(d, exist_ok=True)

# Paths
detail_path = os.path.join(dirs["datos_output"], f"detalle_{base_name}.csv")
compute_path = os.path.join(dirs["datos_output"], f"computo_{base_name}.csv")
summary_path = os.path.join(dirs["datos_output"], f"resumen_{base_name}.csv")

# CSV headers
with open(detail_path, "w", newline="") as f: csv.writer(f).writerow(["Episodio","Epoch","Recompensa","Pasos","Éxito","Epsilon"])
with open(compute_path, "w", newline="") as f: csv.writer(f).writerow(["Episodio","Epoch","Tiempo(s)","CPU(%)","RAM(MB)","GPU_mem(MB)"])
with open(summary_path, "w", newline="") as f: csv.writer(f).writerow(["Métrica","Valor"])

# ===== Setup =====
env, device = ShoppingEnv(), torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net, target_net = DQNetwork(state_size, state_size).to(device), DQNetwork(state_size, state_size).to(device)
target_net.load_state_dict(policy_net.state_dict()); target_net.eval()
optimizer, criterion, memory = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE), nn.MSELoss(), deque(maxlen=MEMORY_SIZE)

# ===== Training =====
total_rewards, total_success, total_time = [], 0, 0
for epoch in range(NUM_EPOCHS):
    for episode in range(EPISODES_PER_EPOCH):
        start_time = time.perf_counter()
        state_text, state, done, total_reward, steps = env.reset(), encode_state(env.reset()), False, 0, 0
        success = 0

        while not done:
            steps += 1
            valid_actions = env.get_valid_actions()
            action = select_action(state, valid_actions, EPSILON)
            next_state, _, done, _ = env.step(action)
            reward = evaluate_with_llm(env.get_state(), action)
            next_state_encoded = encode_state(next_state)
            memory.append((state, valid_actions.index(action), reward, next_state_encoded, done))
            state, total_reward = next_state_encoded, total_reward + reward
            train_dqn()

        if set(env.collected_items) == set(env.shopping_list): success = 1
        total_rewards.append(total_reward); total_success += success
        elapsed = time.perf_counter() - start_time
        total_time += elapsed

        # ===== Save detail =====
        with open(detail_path, "a", newline="") as f:
            csv.writer(f).writerow([episode+1+epoch*EPISODES_PER_EPOCH, epoch+1, total_reward, steps, success, EPSILON])

        # ===== Save compute =====
        process = psutil.Process(os.getpid())
        cpu, ram = psutil.cpu_percent(), process.memory_info().rss / 1024 / 1024
        gpu_mem = torch.cuda.memory_allocated()/1024/1024 if torch.cuda.is_available() else 0
        with open(compute_path, "a", newline="") as f:
            csv.writer(f).writerow([episode+1+epoch*EPISODES_PER_EPOCH, epoch+1, elapsed, cpu, ram, gpu_mem])

    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

# ===== Summary =====
with open(summary_path, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Recompensa Promedio", np.mean(total_rewards)])
    writer.writerow(["Éxito (%)", 100*total_success/TOTAL_EPISODES])
    writer.writerow(["Tiempo Total (s)", total_time])
    writer.writerow(["GPU usada", torch.cuda.is_available()])
