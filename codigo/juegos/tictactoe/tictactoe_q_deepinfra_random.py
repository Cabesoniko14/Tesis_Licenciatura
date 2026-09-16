# ======================  Q-Learning: Tic Tac Toe + LLM Rewards (DeepInfra)  ======================
import os
import sys
import re
import json
import time
import random
import argparse
import psutil
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
import torch
import openai
from openai import OpenAI

# --------------------  0. Metadatos del modelo --------------------
MODEL_TYPE = "QLearning"     # "QLearning" | "DeepQN"
USA_LLM = True
OPONENTE = "random"
REWARD_LEVEL = "accion"      # "accion" | "episodio" | "epoch"
LLM_REWARD_STRATEGY = "replica_std_numerica"  # el prompt reproduce los mismos números del script std
REWARD_TAG = "llmReplicaStd"
SEED = 42                    # semilla fija para reproducibilidad (afecta elección del oponente/epsilon)

LLM_PROVIDER = "DeepInfra"
LLM_BASE_URL = "https://api.deepinfra.com/v1/openai"
LLM_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"  # no-reasoning real; mismo modelo que ya usabas en Groq antes de la deprecación
LLM_TEMPERATURE = 0.0
LLM_MAX_RETRIES = 3

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

# --------------------  2. Agente Q-Learning --------------------------
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

# --------------------  3. Recompensa vía LLM --------------------------
# Por acción. Evalúa tablero antes/después de la jugada del agente, no ve al oponente.
_PROMPT_TEMPLATE = """Evalúas UNA jugada de Tic Tac Toe para Q-Learning. Responde solo con: 1, 0.3, 0 o -1. Sin texto extra.

1    = la jugada gana la partida de inmediato
0.3  = la jugada bloquea una victoria inmediata del oponente
0    = cualquier otra jugada
-1   = el oponente tenía una victoria inmediata disponible y la jugada NO la bloqueó

Antes: {prev_state}
Agente: {agent_letter}
Acción: {action}
Después: {next_state}

Número:
"""

def _parse_reward_value(s: str):
    m = re.search(r"[-+]?\d*\.?\d+", s.strip())
    if not m:
        return None
    try:
        v = float(m.group(0))
        if v < -1: v = -1.0
        if v > 1:  v = 1.0
        return v
    except:
        return None

def llm_reward(client, prev_state, action, next_state, agent_letter, stats,
               model_name=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE, max_retries=LLM_MAX_RETRIES):
    stats["invocaciones_totales"] += 1
    for _ in range(max_retries):
        stats["llamadas_api_totales"] += 1
        try:
            message = _PROMPT_TEMPLATE.format(
                prev_state=prev_state, action=action, next_state=next_state, agent_letter=agent_letter
            )
            resp = client.chat.completions.create(
                messages=[{"role": "user", "content": message}],
                model=model_name,
                temperature=temperature,
                max_tokens=20,  # sin reasoning_effort: qwen3.8-27b responde directo, no piensa por dentro
            )
            raw_message = resp.choices[0].message
            content = (raw_message.content or "").strip()
            parsed = _parse_reward_value(content)
            if parsed is not None:
                return parsed, False
        except Exception as e:
            stats["errores_api_totales"] += 1
            print(f"[LLM Error] {e}")

    stats["fallbacks_totales"] += 1
    print(f"[LLM WARN] Respuesta inválida tras {max_retries} intentos → uso 0.0 de fallback")
    return 0.0, True

# --------------------  4. Resumen de modelos (resumen_modelos/) --------------------
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
    red_neuronal=None,
    llm_info=None,
    entorno=None,
):
    return {
        "archivo_base": base_filename,
        "timestamp": timestamp,
        "modelo": {
            "tipo": MODEL_TYPE,
            "usa_llm": USA_LLM,
            "oponente": OPONENTE,
            "estrategia_recompensa_llm": LLM_REWARD_STRATEGY if USA_LLM else None,
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
        "red_neuronal": red_neuronal,
        "llm": llm_info,
        "entorno": entorno or {},
    }


def guardar_resumen_modelo(resumen, carpeta="resumen_modelos"):
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
        if resumen["modelo"].get("estrategia_recompensa_llm"):
            f.write(f"Estrategia LLM : {resumen['modelo']['estrategia_recompensa_llm']}\n")
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
                elif k == "estadisticas_de_llamadas":
                    f.write("  estadisticas_de_llamadas:\n")
                    for sk, sv in v.items():
                        f.write(f"    {sk}: {sv}\n")
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

# --------------------  5. Entrenamiento --------------------------
def train_qlearning_tictactoe_llm(client, num_epochs=100, episodes_per_epoch=100):
    random.seed(SEED)
    np.random.seed(SEED)

    env = TicTacToe()
    agent = QLearningAgent()

    total_rewards, total_time = [], 0.0
    wins, draws, losses = 0, 0, 0

    llm_stats = {
        "invocaciones_totales": 0,
        "llamadas_api_totales": 0,
        "fallbacks_totales": 0,
        "errores_api_totales": 0,
    }

    acciones_df = pd.DataFrame(columns=["Epoch","Episodio","Agente","Acción","Board","Reward","RewardAcum","LLM_Fallback"])
    computo_df  = pd.DataFrame(columns=["Epoch","Episodio","Tiempo(s)","CPU(%)","RAM(MB)","GPU_mem(MB)"])
    victorias_df= pd.DataFrame(columns=["Epoch","Victorias","Empates","Derrotas","WinRate(%)"])
    resumen_df  = pd.DataFrame(columns=["Métrica","Valor"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_episodes = num_epochs * episodes_per_epoch

    llm_tag = "conLLM" if USA_LLM else "sinLLM"
    base = f"tictactoe_{MODEL_TYPE}_{llm_tag}_{OPONENTE}_{REWARD_TAG}_{timestamp}_ep{total_episodes}"

    os.makedirs("datos_output", exist_ok=True)
    os.makedirs("modelos", exist_ok=True)

    for epoch in range(num_epochs):
        wins_epoch, draws_epoch, losses_epoch = 0, 0, 0

        for ep in range(episodes_per_epoch):
            start = time.perf_counter()
            board = env.reset()
            state = board.copy()
            reward_total = 0.0
            done = False

            agent_letter, opponent_letter = ('X','O') if ((epoch*episodes_per_epoch+ep) % 2 == 0) else ('O','X')

            while not done:
                available_actions = env.available_moves()
                action = agent.choose_action(state, available_actions)
                env.make_move(action, agent_letter)
                next_state = env.board.copy()

                reward, fue_fallback = llm_reward(client, state, action, next_state, agent_letter, llm_stats)
                reward_total += reward

                next_actions = env.available_moves()
                agent.update(state, action, reward, next_state, next_actions)

                acciones_df.loc[len(acciones_df)] = [
                    epoch+1, ep+1, agent_letter, action, next_state.copy(), reward, reward_total, fue_fallback
                ]

                if not env.current_winner and not env.is_draw():
                    opp_actions = env.available_moves()
                    if opp_actions:
                        env.make_move(random.choice(opp_actions), opponent_letter)

                done = env.current_winner is not None or env.is_draw()
                state = env.board.copy()

            if env.current_winner == agent_letter:
                wins += 1; wins_epoch += 1
            elif env.current_winner == opponent_letter:
                losses += 1; losses_epoch += 1
            else:
                draws += 1; draws_epoch += 1

            total_rewards.append(reward_total)
            total_time += time.perf_counter() - start

            computo_df.loc[len(computo_df)] = [
                epoch+1, ep+1, time.perf_counter()-start,
                psutil.cpu_percent(),
                psutil.Process(os.getpid()).memory_info().rss/1024/1024,
                torch.cuda.memory_allocated()/1024/1024 if torch.cuda.is_available() else 0
            ]

        win_rate_epoch = (wins_epoch / episodes_per_epoch) * 100
        victorias_df.loc[len(victorias_df)] = [epoch+1, wins_epoch, draws_epoch, losses_epoch, win_rate_epoch]
        print(f"=== Epoch {epoch+1}/{num_epochs} terminado | winrate={win_rate_epoch:.2f}% "
              f"| LLM fallbacks acumulados={llm_stats['fallbacks_totales']} ===")

    reward_promedio = float(np.mean(total_rewards)) if total_rewards else 0.0
    win_rate_global = (wins / total_episodes) * 100 if total_episodes else 0.0

    resumen_df.loc[len(resumen_df)] = ["Victorias", wins]
    resumen_df.loc[len(resumen_df)] = ["Empates", draws]
    resumen_df.loc[len(resumen_df)] = ["Derrotas", losses]
    resumen_df.loc[len(resumen_df)] = ["Reward promedio", reward_promedio]
    resumen_df.loc[len(resumen_df)] = ["Tiempo total (s)", total_time]
    resumen_df.loc[len(resumen_df)] = ["GPU usada", torch.cuda.is_available()]
    resumen_df.loc[len(resumen_df)] = ["LLM fallbacks totales", llm_stats["fallbacks_totales"]]
    resumen_df.loc[len(resumen_df)] = ["LLM invocaciones totales", llm_stats["invocaciones_totales"]]

    acciones_df.to_csv(f"datos_output/acciones_{base}.csv", index=False)
    computo_df.to_csv(f"datos_output/computo_{base}.csv", index=False)
    victorias_df.to_csv(f"datos_output/victorias_{base}.csv", index=False)
    resumen_df.to_csv(f"datos_output/resumen_{base}.csv", index=False)
    np.save(f"modelos/qtable_{base}.npy", dict(agent.q_table))

    resultados_finales = {
        "victorias": wins,
        "empates": draws,
        "derrotas": losses,
        "win_rate_%": round(win_rate_global, 2),
        "reward_promedio": round(reward_promedio, 4),
    }

    estructura_recompensas = {
        "nivel": REWARD_LEVEL,
        "componentes": {
            "fuente": "LLM (DeepInfra) — ver bloque 'llm' de este mismo resumen para el prompt exacto",
            "rango_valores": "{1, 0.3, 0, -1} — replica WIN_REWARD/BLOCK_BONUS/DRAW/LOSS del script sin LLM",
            "objetivo_declarado_en_el_prompt": (
                "1 si la jugada gana de inmediato, 0.3 si bloquea una victoria inmediata del "
                "oponente, 0 en cualquier otra jugada neutral, -1 si había una victoria inmediata "
                "del oponente disponible y la jugada no la bloqueó."
            ),
            "valor_si_el_llm_falla_o_no_responde_un_numero": 0.0,
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
        "openai_version": getattr(openai, "__version__", "desconocida"),
    }

    llm_info = {
        "proveedor": LLM_PROVIDER,
        "modelo_llm": LLM_MODEL_NAME,
        "temperature": LLM_TEMPERATURE,
        "estrategia_recompensa": LLM_REWARD_STRATEGY,
        "estrategia_descripcion": (
            "El prompt le pide al LLM que reproduzca, con los mismos 4 números, la función de "
            "recompensa del script sin LLM (WIN_REWARD, BLOCK_BONUS, DRAW_REWARD, y -1 como proxy "
            "de LOSS_REWARD)."
        ),
        "api_key_source": "variable de entorno DEEPINFRA_API_KEY (o DEEPINFRA) o --deepinfra-api-key",
        "cuando_se_invoca": (
            "Por acción: una llamada al LLM por cada movimiento del agente, justo después de "
            "jugarlo y antes de actualizar la Q-table. No ve la respuesta del oponente."
        ),
        "max_retries_por_accion": LLM_MAX_RETRIES,
        "nota_seleccion_modelo": (
            "Se probó openai/gpt-oss-20b (Groq) con reasoning_effort='low' (su mínimo posible): en "
            "una corrida corta consumió ~35K tokens de salida en razonamiento oculto no visible "
            "para una tarea que solo requiere devolver un número. Se probó también qwen/qwen3.8-27b "
            "(Groq, no-reasoning por default), que sí evitaba ese gasto pero era ~16-50x más caro por "
            "token que el llama-3.1-8b-instant original. Se migró a DeepInfra, que sigue hospedando "
            "el mismo Meta-Llama-3.1-8B-Instruct (deprecado por Groq el 16-ago-2026) a fracción del "
            "costo, sin razonamiento oculto."
        ),
        "parseo_respuesta": (
            "Se extrae el primer número (entero o decimal) de la respuesta con regex y se recorta "
            "a [-1, 1]; si no hay número parseable, se reintenta."
        ),
        "valor_fallback_si_falla": 0.0,
        "prompt_template": _PROMPT_TEMPLATE,
        "estadisticas_de_llamadas": {
            "invocaciones_totales": llm_stats["invocaciones_totales"],
            "llamadas_api_totales": llm_stats["llamadas_api_totales"],
            "fallbacks_totales": llm_stats["fallbacks_totales"],
            "fallback_rate_%": (
                round(100 * llm_stats["fallbacks_totales"] / llm_stats["invocaciones_totales"], 2)
                if llm_stats["invocaciones_totales"] else 0.0
            ),
            "errores_api_totales": llm_stats["errores_api_totales"],
        },
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
        red_neuronal=None,
        llm_info=llm_info,
        entorno=entorno,
    )
    guardar_resumen_modelo(resumen_modelo)

    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    return acciones_df, computo_df, victorias_df, resumen_df

# --------------------  6. API key + Main --------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepinfra-api-key", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--episodes-per-epoch", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Acepta DEEPINFRA_API_KEY (nombre recomendado) o DEEPINFRA (por si guardaste el Secret así)
    deepinfra_api_key = (
        args.deepinfra_api_key
        or os.environ.get("DEEPINFRA_API_KEY")
        or os.environ.get("DEEPINFRA")
    )
    if not deepinfra_api_key:
        raise RuntimeError(
            "Falta la API key de DeepInfra. Pásala con DEEPINFRA_API_KEY=tu_key antes del comando, "
            "o con --deepinfra-api-key tu_key."
        )

    client = OpenAI(api_key=deepinfra_api_key, base_url=LLM_BASE_URL)

    acciones, computo, victorias, resumen = train_qlearning_tictactoe_llm(
        client=client,
        num_epochs=args.num_epochs,
        episodes_per_epoch=args.episodes_per_epoch,
    )