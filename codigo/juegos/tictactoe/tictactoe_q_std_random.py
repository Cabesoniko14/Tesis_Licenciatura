# ======================  Q-Learning: Tic Tac Toe (Tabular) + bono por BLOQUEO  ======================
import os
import sys
import json
import time
import random
import psutil
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
import torch

# --------------------  0. Metadatos del modelo (para el rastreo en resumen_modelos) --------------------
# Estas constantes son las que después le van a dar sentido a la nomenclatura de archivos
# y al contenido del resumen. Cámbialas en cada script (QLearning/DeepQN, con/sin LLM).
MODEL_TYPE = "QLearning"     # "QLearning" | "DeepQN"
USA_LLM = False              # True en las variantes con LLM
OPONENTE = "random"          # tipo de oponente durante el entrenamiento
REWARD_LEVEL = "accion"      # "accion" | "episodio" | "epoch" -> a qué nivel se calcula/aplica la recompensa
SEED = 42                    # semilla fija para reproducibilidad

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

# --------------------  Utilidad: jugadas ganadoras inmediatas --------------------------
def winning_moves_for(board_list, letter):
    """Regresa el conjunto de índices que darían victoria a `letter` si jugara AHORA en ese índice."""
    wins = set()
    for i, spot in enumerate(board_list):
        if spot != ' ':
            continue
        # simular
        b = board_list.copy()
        b[i] = letter
        # comprobar victoria con reglas del juego
        if _is_win_after_place(b, i, letter):
            wins.add(i)
    return wins

def _is_win_after_place(board, square, letter):
    # fila
    r0 = (square // 3) * 3
    if board[r0] == board[r0+1] == board[r0+2] == letter: return True
    # columna
    c = square % 3
    if board[c] == board[c+3] == board[c+6] == letter: return True
    # diagonales (solo casillas pares)
    if square % 2 == 0:
        if board[0] == board[4] == board[8] == letter: return True
        if board[2] == board[4] == board[6] == letter: return True
    return False

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

# --------------------  3. Sistema de resumen de modelos (resumen_modelos/) --------------------
def construir_resumen_modelo(
    base_filename,
    timestamp,
    num_epochs,
    episodes_per_epoch,
    total_episodes,
    tiempo_total,
    resultados_finales,
    estructura_recompensas,
    hiperparametros=None,
    red_neuronal=None,   # dict con capas/dims/activaciones -> se llena en los scripts DeepQN
    llm_info=None,        # dict con prompt/cuándo se invoca -> se llena en los scripts con LLM
    entorno=None,          # versiones de Python/librerías usadas en la corrida
):
    """
    Arma un dict estandarizado con todo lo necesario para rastrear un modelo entre experimentos.
    Pensado para reusarse igual en QLearning/DeepQN y con/sin LLM: los campos que no aplican
    quedan en None y así el resumen es diagnóstico por sí solo (se ve a qué variante corresponde).
    """
    return {
        "archivo_base": base_filename,
        "timestamp": timestamp,
        "modelo": {
            "tipo": MODEL_TYPE,       # "QLearning" o "DeepQN"
            "usa_llm": USA_LLM,
            "oponente": OPONENTE,
        },
        "entrenamiento": {
            "num_epochs": num_epochs,
            "episodes_per_epoch": episodes_per_epoch,
            "total_episodes": total_episodes,
            "tiempo_total_segundos": round(tiempo_total, 4),
            "tiempo_promedio_por_episodio_segundos": (
                round(tiempo_total / total_episodes, 6) if total_episodes else None
            ),
            "hiperparametros": hiperparametros or {},
        },
        "resultados_finales": resultados_finales,
        "estructura_recompensas": estructura_recompensas,
        "red_neuronal": red_neuronal,   # None si es QLearning tabular
        "llm": llm_info,                # None si no se usó LLM
        "entorno": entorno or {},
    }


def guardar_resumen_modelo(resumen, carpeta="resumen_modelos"):
    """Guarda el resumen en JSON (para agregarlo luego entre experimentos) y en TXT (para leerlo rápido)."""
    os.makedirs(carpeta, exist_ok=True)
    nombre_archivo = f"resumen_{resumen['archivo_base']}"

    ruta_json = os.path.join(carpeta, f"{nombre_archivo}.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    ruta_txt = os.path.join(carpeta, f"{nombre_archivo}.txt")
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write(f"RESUMEN DE MODELO: {resumen['archivo_base']}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Tipo de modelo : {resumen['modelo']['tipo']}\n")
        f.write(f"Usa LLM        : {'Sí' if resumen['modelo']['usa_llm'] else 'No'}\n")
        f.write(f"Oponente       : {resumen['modelo']['oponente']}\n")
        f.write(f"Timestamp      : {resumen['timestamp']}\n\n")

        f.write("-- Entrenamiento --\n")
        for k, v in resumen["entrenamiento"].items():
            if k == "hiperparametros":
                f.write("  hiperparametros:\n")
                for hk, hv in v.items():
                    f.write(f"    {hk}: {hv}\n")
            else:
                f.write(f"  {k}: {v}\n")

        f.write("\n-- Resultados finales --\n")
        for k, v in resumen["resultados_finales"].items():
            f.write(f"  {k}: {v}\n")

        f.write("\n-- Estructura de recompensas --\n")
        f.write(f"  nivel_aplicacion: {resumen['estructura_recompensas'].get('nivel')}\n")
        f.write("  componentes:\n")
        for comp, val in resumen["estructura_recompensas"].get("componentes", {}).items():
            f.write(f"    {comp}: {val}\n")

        f.write("\n-- Red neuronal --\n")
        if resumen["red_neuronal"]:
            for k, v in resumen["red_neuronal"].items():
                f.write(f"  {k}: {v}\n")
        else:
            f.write("  N/A (modelo tabular, no aplica)\n")

        f.write("\n-- LLM --\n")
        if resumen["llm"]:
            for k, v in resumen["llm"].items():
                if k == "prompt_template":
                    f.write(f"  prompt_template:\n{v}\n")
                else:
                    f.write(f"  {k}: {v}\n")
        else:
            f.write("  N/A (no se usó LLM en este experimento)\n")

        if resumen.get("entorno"):
            f.write("\n-- Entorno --\n")
            for k, v in resumen["entorno"].items():
                f.write(f"  {k}: {v}\n")

    print(f"[OK] Resumen de modelo guardado en:\n  {ruta_json}\n  {ruta_txt}")
    return ruta_json, ruta_txt

# --------------------  4. Entrenamiento --------------------------
def train_qlearning_tictactoe(num_epochs=100, episodes_per_epoch=100):
    random.seed(SEED)
    np.random.seed(SEED)

    env = TicTacToe()
    agent = QLearningAgent()

    # ======= ÚNICO CAMBIO DE RECOMPENSAS =======
    WIN_REWARD     = 1.0     # victoria inmediata (igual al paper)
    LOSS_REWARD    = -1.0    # derrota inmediata (igual al paper)
    DRAW_REWARD    = 0.0     # empate (igual al paper)
    BLOCK_BONUS    = 0.3     # **BONO por bloquear una victoria inmediata del rival**

    total_rewards, total_time = [], 0.0
    wins, draws, losses = 0, 0, 0

    # Tablas
    acciones_df = pd.DataFrame(columns=["Epoch","Episodio","Agente","Acción","Board","Reward","RewardAcum"])
    computo_df  = pd.DataFrame(columns=["Epoch","Episodio","Tiempo(s)","CPU(%)","RAM(MB)","GPU_mem(MB)"])
    victorias_df= pd.DataFrame(columns=["Epoch","Victorias","Empates","Derrotas","WinRate(%)"])
    resumen_df  = pd.DataFrame(columns=["Métrica","Valor"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_episodes = num_epochs * episodes_per_epoch

    # Nomenclatura mejorada: deja explícito tipo de modelo, si usa LLM, oponente y bono activo,
    # para que cualquier archivo (csv, npy, json de resumen) se identifique solo con el nombre.
    llm_tag = "conLLM" if USA_LLM else "sinLLM"
    base = f"tictactoe_{MODEL_TYPE}_{llm_tag}_{OPONENTE}_blockBonus_{timestamp}_ep{total_episodes}"

    os.makedirs("datos_output", exist_ok=True)
    os.makedirs("modelos", exist_ok=True)

    for epoch in range(num_epochs):
        wins_epoch, draws_epoch, losses_epoch = 0, 0, 0

        for ep in range(episodes_per_epoch):
            t0 = time.perf_counter()
            board = env.reset()
            state = board.copy()
            reward_total = 0.0
            done = False

            # alternar quién empieza
            agent_letter, opponent_letter = ('X','O') if ((epoch*episodes_per_epoch+ep) % 2 == 0) else ('O','X')

            while not done:
                prev_board = state.copy()
                available_actions = env.available_moves()
                action = agent.choose_action(state, available_actions)

                # --- el agente juega ---
                env.make_move(action, agent_letter)
                next_state = env.board.copy()

                # ----- RECOMPENSA (se calcula y aplica por ACCIÓN, no por episodio ni por epoch) -----
                # 1) Terminal inmediata para el agente
                if env.current_winner == agent_letter:
                    reward, done = WIN_REWARD, True
                elif env.is_draw():
                    reward, done = DRAW_REWARD, True
                else:
                    # 2) BONO por BLOQUEO: si en el estado previo el oponente tenía
                    #    una jugada ganadora inmediata y el agente ocupó ese lugar.
                    opp_wins_prev = winning_moves_for(prev_board, opponent_letter)
                    reward = BLOCK_BONUS if action in opp_wins_prev else 0.0
                    done = False

                    # FIX: el oponente responde AQUÍ, antes de actualizar Q, para poder
                    # castigar con LOSS_REWARD la misma jugada del agente que permitió
                    # (o no evitó) la derrota. Antes, si el oponente ganaba después,
                    # el episodio se cerraba sin que LOSS_REWARD se aplicara nunca.
                    opp_actions = env.available_moves()
                    if opp_actions:
                        opp_action = random.choice(opp_actions)
                        env.make_move(opp_action, opponent_letter)
                        next_state = env.board.copy()
                        if env.current_winner == opponent_letter:
                            reward = LOSS_REWARD
                            done = True
                        elif env.is_draw():
                            done = True
                # ----------------------

                next_actions = env.available_moves()
                agent.update(state, action, reward, next_state, next_actions)

                reward_total += reward
                acciones_df.loc[len(acciones_df)] = [
                    epoch+1, ep+1, agent_letter, action, next_state.copy(), reward, reward_total
                ]

                state = env.board.copy()

            # métricas del episodio
            if env.current_winner == agent_letter:
                wins += 1; wins_epoch += 1
            elif env.current_winner == opponent_letter:
                losses += 1; losses_epoch += 1
            else:
                draws += 1; draws_epoch += 1

            total_rewards.append(reward_total)
            elapsed = time.perf_counter() - t0
            total_time += elapsed

            computo_df.loc[len(computo_df)] = [
                epoch+1, ep+1, elapsed,
                psutil.cpu_percent(),
                psutil.Process(os.getpid()).memory_info().rss/1024/1024,
                torch.cuda.memory_allocated()/1024/1024 if torch.cuda.is_available() else 0
            ]

        win_rate_epoch = (wins_epoch / episodes_per_epoch) * 100
        victorias_df.loc[len(victorias_df)] = [epoch+1, wins_epoch, draws_epoch, losses_epoch, win_rate_epoch]
        print(f"=== Epoch {epoch+1}/{num_epochs} terminado | winrate={win_rate_epoch:.2f}% ===")

    # Resumen global (csv de siempre)
    reward_promedio = float(np.mean(total_rewards)) if total_rewards else 0.0
    win_rate_global = (wins / total_episodes) * 100 if total_episodes else 0.0

    resumen_df.loc[len(resumen_df)] = ["Victorias", wins]
    resumen_df.loc[len(resumen_df)] = ["Empates", draws]
    resumen_df.loc[len(resumen_df)] = ["Derrotas", losses]
    resumen_df.loc[len(resumen_df)] = ["Reward promedio", reward_promedio]
    resumen_df.loc[len(resumen_df)] = ["Tiempo total (s)", total_time]
    resumen_df.loc[len(resumen_df)] = ["GPU usada", torch.cuda.is_available()]

    # Guardar resultados (como ya lo hacías)
    acciones_df.to_csv(f"datos_output/acciones_{base}.csv", index=False)
    computo_df.to_csv(f"datos_output/computo_{base}.csv", index=False)
    victorias_df.to_csv(f"datos_output/victorias_{base}.csv", index=False)
    resumen_df.to_csv(f"datos_output/resumen_{base}.csv", index=False)
    np.save(f"modelos/qtable_{base}.npy", dict(agent.q_table))

    # --------------------  NUEVO: resumen de modelo en resumen_modelos/ --------------------
    resultados_finales = {
        "victorias": wins,
        "empates": draws,
        "derrotas": losses,
        "win_rate_%": round(win_rate_global, 2),
        "reward_promedio": round(reward_promedio, 4),
    }

    estructura_recompensas = {
        "nivel": REWARD_LEVEL,  # "accion": la recompensa se calcula y se usa para actualizar Q en cada movimiento
        "componentes": {
            "WIN_REWARD": WIN_REWARD,
            "LOSS_REWARD": LOSS_REWARD,
            "DRAW_REWARD": DRAW_REWARD,
            "BLOCK_BONUS": BLOCK_BONUS,
            "descripcion": (
                "Recompensa asignada a la última acción del agente en el turno: +1 si esa jugada "
                "gana de inmediato, 0 si empata de inmediato, +0.3 si bloquea una jugada ganadora "
                "inmediata del oponente, 0 si es una jugada intermedia sin efecto, y -1 (LOSS_REWARD) "
                "si, tras la respuesta del oponente, esa misma jugada terminó permitiendo la derrota."
            ),
        },
    }

    hiperparametros = {
        "alpha": agent.alpha,
        "gamma": agent.gamma,
        "epsilon": agent.epsilon,
        "seed": SEED,
    }

    entorno = {
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "torch_version": torch.__version__,
    }

    resumen_modelo = construir_resumen_modelo(
        base_filename=base,
        timestamp=timestamp,
        num_epochs=num_epochs,
        episodes_per_epoch=episodes_per_epoch,
        total_episodes=total_episodes,
        tiempo_total=total_time,
        resultados_finales=resultados_finales,
        estructura_recompensas=estructura_recompensas,
        hiperparametros=hiperparametros,
        red_neuronal=None,   # este script es QLearning tabular, no aplica
        llm_info=None,        # este script no usa LLM
        entorno=entorno,
    )
    guardar_resumen_modelo(resumen_modelo)
    # -----------------------------------------------------------------------------------------

    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    return acciones_df, computo_df, victorias_df, resumen_df

# --------------------  Main --------------------------
if __name__ == "__main__":
    acciones, computo, victorias, resumen = train_qlearning_tictactoe(
        num_epochs=100, episodes_per_epoch=100
    )