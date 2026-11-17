import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from datetime import datetime
import time
import psutil
import pandas as pd
import concurrent.futures
from openai import OpenAI   # ✅ nuevo cliente

# ====== OPENAI API KEY ======
client = OpenAI(api_key="")

# ========== ENVIRONMENT ==========
class ShoppingEnv:
    def __init__(self):
        self.aisles = {
            "Dairy": ["milk", "cheese", "yogurt", "butter"],
            "Bakery": ["bread", "cake", "croissant", "bagel"],
            "Produce": ["apples", "bananas", "carrots", "lettuce"]
        }
        self.all_items = [item for aisle_items in self.aisles.values() for item in aisle_items]
        self.shopping_list, self.current_location, self.collected_items = [], "Entrance", []
        self.step_count, self.max_steps = 0, 15

    def reset(self):
        self.current_location, self.collected_items, self.step_count = "Entrance", [], 0
        self.shopping_list = random.sample(self.all_items, random.randint(2, 3))
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
            loc = action.split("go to ")[1]
            if loc in self.aisles: self.current_location, info = loc, f"Moved to {loc}"
            else: reward, info = -1, "Invalid location"

        elif action.startswith("take"):
            item = action.split("take ")[1]
            if self.current_location in self.aisles and item in self.aisles[self.current_location]:
                if item not in self.collected_items:
                    self.collected_items.append(item); reward, info = 10, f"Collected {item}"
                else: reward, info = -1, f"Already collected {item}"
            else: reward, info = -1, f"{item} not in {self.current_location}"

        elif action == "checkout":
            if set(self.collected_items) == set(self.shopping_list):
                reward, done, info = 50, True, "Checked out successfully"
            else: reward, info = -10, "Shopping list not complete"
        else: reward, info = -1, "Invalid action"

        if self.step_count >= self.max_steps: done, info = True, "Max steps reached"
        return self.get_state(), reward, done, info

    def get_valid_actions(self):
        actions = [f"go to {a}" for a in self.aisles]
        if self.current_location in self.aisles:
            actions.extend([f"take {i}" for i in self.aisles[self.current_location]])
        actions.append("checkout")
        return actions

# ========== LLM REWARD ==========
def evaluate_with_llm(state, action):
    msg = (
        "Evaluate the following action for a state in the shopping game. "
        "The objective is to go to the corresponding aisles and select the appropriate products "
        "depending on the shopping list. At the end you must checkout with the appropriate items. "
        "Respond with only one of: SUPER BAD, BAD, REGULAR, GOOD, SUPER GOOD.\n\n"
        f"State: {state}\nAction: {action}"
    )

    def call_llm():
        response = client.chat.completions.create(   # ✅ nuevo formato
            model="gpt-5-mini-2025-08-07",
            messages=[{"role": "user", "content": msg}],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            llm_value = executor.submit(call_llm).result(timeout=5)
        print(f"   [LLM Response] {llm_value}")
    except Exception as e:
        print(f"   [LLM Error] {e} → fallback a REGULAR")
        llm_value = "REGULAR"

    mapping = {"SUPER BAD": -4, "BAD": -2, "REGULAR": 0, "GOOD": 2, "SUPER GOOD": 4}
    return mapping.get(llm_value, 0)

# ========== DQN ==========
class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(input_dim, 256), nn.Linear(256, 256), nn.Linear(256, output_dim)
    def forward(self, x): return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

def encode_state(state):
    vec = np.zeros(state_size); vec[hash(str(state)) % state_size] = 1; return vec

def select_action(state, valid_actions, eps):
    if random.random() < eps: return random.choice(valid_actions)
    st = torch.FloatTensor(state).unsqueeze(0).to(device)
    q_vals = policy_net(st)
    return max(valid_actions, key=lambda a: q_vals[0, valid_actions.index(a)].item())

def train_dqn():
    if len(memory) < BATCH_SIZE: return
    states, actions, rewards, next_states, dones = zip(*random.sample(memory, BATCH_SIZE))
    states, actions, rewards = torch.FloatTensor(states).to(device), torch.LongTensor(actions).to(device), torch.FloatTensor(rewards).to(device)
    next_states, dones = torch.FloatTensor(next_states).to(device), torch.FloatTensor(dones).to(device)
    q_vals = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_vals = target_net(next_states).max(1)[0]
    expected = rewards + GAMMA * next_q_vals * (1 - dones)
    loss = criterion(q_vals, expected); optimizer.zero_grad(); loss.backward(); optimizer.step()

# ========== CONFIG ==========
NUM_EPOCHS, EPISODES_PER_EPOCH = 10, 10
TOTAL_EPISODES  = NUM_EPOCHS * EPISODES_PER_EPOCH
GAMMA, EPSILON, EPSILON_DECAY, EPSILON_MIN = 0.99, 1.0, 0.995, 0.1
LR, BATCH_SIZE, MEMORY_SIZE, state_size = 1e-3, 64, 10000, 512

# ========== PATHS ==========
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base = f"shopping_deepqn_gpt4_{timestamp}_episodes_{TOTAL_EPISODES}"
os.makedirs("datos_output", exist_ok=True); os.makedirs("logs", exist_ok=True); os.makedirs("modelos", exist_ok=True)
paths = {
    "acciones": f"datos_output/acciones_{base}.csv",
    "computo": f"datos_output/computo_{base}.csv",
    "resumen": f"datos_output/resumen_{base}.csv",
    "log": f"logs/log_{base}.txt",
    "policy": f"modelos/policy_{base}.pth",
    "target": f"modelos/target_{base}.pth"
}

# ========== SETUP ==========
env, device = ShoppingEnv(), torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net, target_net = DQNetwork(state_size, state_size).to(device), DQNetwork(state_size, state_size).to(device)
target_net.load_state_dict(policy_net.state_dict()); target_net.eval()
optimizer, criterion, memory = optim.Adam(policy_net.parameters(), lr=LR), nn.MSELoss(), deque(maxlen=MEMORY_SIZE)

acciones_df = pd.DataFrame(columns=["Episodio","Epoch","Acción","CanastaActual","CanastaObjetivo","RecompensaObtenida","RecompensaAcumulada"])
computo_df = pd.DataFrame(columns=["Episodio","Epoch","Tiempo(s)","CPU(%)","RAM(MB)","GPU_mem(MB)"])
resumen_df = pd.DataFrame(columns=["Métrica","Valor"])

# ========== TRAIN ==========
total_rewards, total_success, total_time = [], 0, 0
with open(paths["log"], "w") as logf:
    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Inicio Epoch {epoch+1}/{NUM_EPOCHS} ===")
        epoch_total = 0

        for ep in range(EPISODES_PER_EPOCH):
            print(f"\n[Epoch {epoch+1} | Episodio {ep+1}]")
            start = time.perf_counter()
            state, done, total_r, steps, success = encode_state(env.reset()), False, 0, 0, 0
            print(f"   Lista objetivo: {env.shopping_list}")

            while not done:
                steps += 1
                action = select_action(state, env.get_valid_actions(), EPSILON)
                next_state, _, done, info = env.step(action)
                reward = evaluate_with_llm(env.get_state(), action)

                state, total_r = encode_state(next_state), total_r + reward
                memory.append((state, env.get_valid_actions().index(action), reward, encode_state(next_state), done))
                acciones_df.loc[len(acciones_df)] = [ep+1+epoch*EPISODES_PER_EPOCH, epoch+1, action, list(env.collected_items), list(env.shopping_list), reward, total_r]

                print(f"   Acción: {action} | Recompensa: {reward} | Total acumulado: {total_r}")
                train_dqn()

            if set(env.collected_items) == set(env.shopping_list): success = 1
            epoch_total += total_r; total_rewards.append(total_r); total_success += success
            elapsed = time.perf_counter()-start; total_time += elapsed
            computo_df.loc[len(computo_df)] = [ep+1+epoch*EPISODES_PER_EPOCH, epoch+1, elapsed, psutil.cpu_percent(), psutil.Process(os.getpid()).memory_info().rss/1024/1024, torch.cuda.memory_allocated()/1024/1024 if torch.cuda.is_available() else 0]

            print(f"   Episodio terminado | Total Reward: {total_r} | Éxito: {success} | Pasos: {steps}")

        EPSILON = max(EPSILON_MIN, EPSILON*EPSILON_DECAY)
        avg_reward = epoch_total/EPISODES_PER_EPOCH
        print(f"=== Fin Epoch {epoch+1} | Promedio Recompensa={avg_reward:.2f} | Epsilon={EPSILON:.3f} ===")

# ========== SAVE ==========
resumen_df.loc[len(resumen_df)] = ["Recompensa Promedio", np.mean(total_rewards)]
resumen_df.loc[len(resumen_df)] = ["Éxito (%)", 100*total_success/TOTAL_EPISODES]
resumen_df.loc[len(resumen_df)] = ["Tiempo Total (s)", total_time]
resumen_df.loc[len(resumen_df)] = ["GPU usada", torch.cuda.is_available()]
acciones_df.to_csv(paths["acciones"], index=False); computo_df.to_csv(paths["computo"], index=False); resumen_df.to_csv(paths["resumen"], index=False)
torch.save(policy_net.state_dict(), paths["policy"]); torch.save(target_net.state_dict(), paths["target"])

print("\n=== Entrenamiento completado con ChatGPT (API moderna) como función de recompensas ===")
