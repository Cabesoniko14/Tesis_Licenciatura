import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
import gc
import csv
import openai
import concurrent.futures  # For timeouts
import time  # For enforcing delays

# Evaluate actions with OpenAI's GPT-4 and timeout

# Environment setup
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
        self.max_steps = 10  # Limit the number of steps per episode

    def reset(self):
        print("[DEBUG] Resetting environment")
        self.current_location = "Entrance"
        self.collected_items = []
        self.step_count = 0
        # Randomize shopping list (min 2, max 3 items)
        num_items = random.randint(2, 3)
        self.shopping_list = random.sample(self.all_items, num_items)
        print(f"[DEBUG] Shopping list: {self.shopping_list}")
        return self.get_state()

    def get_state(self):
        state = {
            "location": self.current_location,
            "collected_items": self.collected_items,
            "remaining_items": [item for item in self.shopping_list if item not in self.collected_items]
        }
        print(f"[DEBUG] Current state: {state}")
        return state

    def step(self, action):
        print(f"[DEBUG] Executing action: {action}")
        self.step_count += 1
        reward = 0
        done = False
        info = ""

        if action.startswith("go to"):
            location = action.split("go to ")[1]
            if location in self.aisles:
                self.current_location = location
                info = f"Moved to {location}"
            else:
                reward = -1  # Penalty for invalid moves
                info = "Invalid location"

        elif action.startswith("take"):
            item = action.split("take ")[1]
            if self.current_location in self.aisles and item in self.aisles[self.current_location]:
                if item not in self.collected_items:
                    self.collected_items.append(item)
                    reward = 10  # Reward for collecting an item
                    info = f"Collected {item}"
                else:
                    reward = -1  # Penalty for taking an already collected item
                    info = f"Already collected {item}"
            else:
                reward = -1  # Penalty for taking an item not in the current location
                info = f"{item} not found in {self.current_location}"

        elif action == "checkout":
            if set(self.collected_items) == set(self.shopping_list):
                reward = 50  # Big reward for completing the shopping list
                done = True
                info = "Checked out successfully"
            else:
                reward = -10  # Penalty for checking out early
                info = "Shopping list not complete"

        else:
            reward = -1  # Penalty for invalid actions
            info = "Invalid action"

        if self.step_count >= self.max_steps:
            done = True
            info = "Max steps reached"

        print(f"[DEBUG] Step result: reward={reward}, done={done}, info={info}")
        return self.get_state(), reward, done, info

    def get_valid_actions(self):
        actions = [f"go to {aisle}" for aisle in self.aisles]
        if self.current_location in self.aisles:
            actions.extend([f"take {item}" for item in self.aisles[self.current_location]])
        actions.append("checkout")
        print(f"[DEBUG] Valid actions: {actions}")
        return actions

# Evaluate actions with OpenAI's GPT-4 and timeout

def evaluate_with_llm(state, action):
    """
    Evaluate the action using GPT-4 and assign a reward.
    """
    print("[DEBUG] Evaluating with LLM")
    start_time = time.time()
    message = (
        f"Evaluate the following action for a state in the shopping game. "
        f"The objective is to go to the corresponding aisles and select the appropriate products depending on the shopping list. "
        f"At the end, you have to checkout with the appropriate items. Respond with one of the following: SUPER BAD, BAD, REGULAR, GOOD, SUPER GOOD.\n\n"
        f"State: {state}\nAction: {action}\n\n"
        f"ONLY respond with one of the mentioned options. Not an explanation or anything, just: SUPER BAD, BAD, REGULAR, GOOD, SUPER GOOD."
    )

    def call_llm():
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",  # Use a valid GPT-4 model
            messages=[{"role": "user", "content": message}]
        )
        return response["choices"][0]["message"]["content"].strip()

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(call_llm)
            llm_value = future.result(timeout=2)  # 10-second timeout
            print(f"[DEBUG] LLM Evaluation: {llm_value}")
            reward = transcribe_llm_value(llm_value)
    except concurrent.futures.TimeoutError:
        print("[DEBUG] LLM evaluation timed out.")
        reward = 0  # Default reward for timeout
    except openai.error.OpenAIError as e:
        print(f"[DEBUG] Error with LLM evaluation: {e}")
        reward = 0  # Default reward if there's an error

    elapsed_time = time.time() - start_time
    if elapsed_time < 3:
        delay = 3 - elapsed_time
        print(f"[DEBUG] Enforcing delay of {delay:.2f} seconds to maintain minimum step time.")
        time.sleep(delay)

    return reward

def transcribe_llm_value(llm_value):
    """
    Convert LLM's qualitative evaluation to a numeric reward.
    """
    transcription = {
        "SUPER BAD": -4,
        "BAD": -2,
        "REGULAR": 0,
        "GOOD": 2,
        "SUPER GOOD": 4
    }
    reward = transcription.get(llm_value, 0)
    print(f"[DEBUG] Transcribed reward: {reward}")
    return reward

# Add debugging everywhere by printing intermediate values and critical events.
# Full training loop and additional logic remain unchanged but include debugging lines.




# DQN setup
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

# Helper functions
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
    if len(memory) < BATCH_SIZE:
        return
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor(dones).to(device)

    # Get Q-values for the actions taken
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Get max Q-values for the next states
    next_q_values = target_net(next_states).max(1)[0]
    
    # Compute expected Q-values
    expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)
    
    # Compute loss
    loss = criterion(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Configuration
NUM_EPOCHS = 20
EPISODES_PER_EPOCH = 20
TOTAL_EPISODES = NUM_EPOCHS * EPISODES_PER_EPOCH
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.1
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
state_size = 512

# Directories
log_dir = "logs"
results_dir = "resultados"
models_dir = "modelos"
data_output_dir = "datos_output"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(data_output_dir, exist_ok=True)

# Logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(log_dir, f"log_shopping_{timestamp}_episodes_{TOTAL_EPISODES}_LLM.txt")
data_file_path = os.path.join(data_output_dir, f"data_shopping_{timestamp}_episodes_{TOTAL_EPISODES}_LLM.csv")

# Create CSV file
with open(data_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Episodio", "Epoch", "Recompensa", "Epsilon"])

# Environment and device
env = ShoppingEnv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN setup
policy_net = DQNetwork(state_size, state_size).to(device)
target_net = DQNetwork(state_size, state_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()
memory = deque(maxlen=MEMORY_SIZE)

# Training loop
epoch_rewards = []
with open(log_file_path, "w") as log_file:
    for epoch in range(NUM_EPOCHS):
        epoch_total_reward = 0
        log_file.write(f"--- Inicio del Epoch {epoch + 1} ---\n")
        print(f"Epoch {epoch + 1}...")

        for episode in range(EPISODES_PER_EPOCH):
            state_text = env.reset()
            state = encode_state(state_text)
            done = False
            total_reward = 0

            shopping_list = env.shopping_list
            log_file.write(f"Episodio {episode + 1}:")
            log_file.write(f"Lista de compras: {shopping_list}\n")
            print(f"  Episodio {episode + 1} - Lista de compras: {shopping_list}...")

            step_count = 0
            while not done:
                valid_actions = env.get_valid_actions()
                action = select_action(state, valid_actions, EPSILON)

                next_state, _, done, info = env.step(action)
                reward = evaluate_with_llm(env.get_state(), action)
                step_count = step_count + 1
                print(f'Evaluación de LLM completada número: {step_count}')

                log_file.write(f"Instrucción: {env.get_state()}, Acción tomada: {action}, Respuesta: {info}, Recompensa: {reward}\n")

                next_state_encoded = encode_state(next_state)

                memory.append((state, valid_actions.index(action), reward, next_state_encoded, done))
                state = next_state_encoded
                total_reward += reward

                train_dqn()

            epoch_total_reward += total_reward

            # Write episode data to CSV
            with open(data_file_path, mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([episode + 1 + epoch * EPISODES_PER_EPOCH, epoch + 1, total_reward, EPSILON])

            log_file.write(f"Recompensa del Episodio: {total_reward}\n")
            log_file.write("-------------------------------\n")

        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
        epoch_average_reward = epoch_total_reward / EPISODES_PER_EPOCH
        epoch_rewards.append(epoch_average_reward)

        log_file.write(f"Recompensa Promedio del Epoch {epoch + 1}: {epoch_average_reward}\n")
        log_file.write("===================================\n")

        # Update target network
        if (epoch + 1) % 5 == 0:
            target_net.load_state_dict(policy_net.state_dict())

# Save model
torch.save(policy_net.state_dict(), os.path.join(models_dir, f"model_shopping_{timestamp}_episodes_{TOTAL_EPISODES}_LLM.pth"))

# Plot average rewards
plt.figure(figsize=(10, 6))
plt.title(f"Promedio de Recompensas por Epoch")
plt.xlabel("Epoch")
plt.ylabel("Recompensa Promedio")
plt.plot(range(1, NUM_EPOCHS + 1), epoch_rewards, marker="o", label="Promedio por Epoch")
plt.legend()
plt.savefig(os.path.join(results_dir, f"shopping_rewards_{timestamp}_episodes_{TOTAL_EPISODES}_LLM.png"))
plt.close()

print("Entrenamiento completado. Modelo y resultados guardados.")
