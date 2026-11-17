# -----------------------  ZORK: RANDOM-CHIOCE -----------------------------

from jericho import *
import random

# --------------------  1. Iniciar ambiente --------------------------

# Ruta al archivo Zork
zork_path = "jericho-game-suite/zork1.z5"

# Crear el entorno
env = FrotzEnv(zork_path)

# Reiniciar el entorno y obtener el estado inicial
state, info = env.reset()

print("Initial State:")
print(state)  # Mostrar el texto inicial
print("=" * 50)

# Obtener el vocabulario permitido por el juego
vocab = env.get_dictionary()

# ------------  2. Función para generar acciones aleatorias -----------------------

# Función para generar posibles acciones (simple ejemplo basado en vocabulario)
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

# Generar todas las acciones posibles
possible_actions = generate_actions(vocab)

# ----------------------------  3. Iniciar juego ----------------------------

done = False
while not done:
    # Elegir una acción aleatoria
    action = random.choice(possible_actions)
    print(f"Action: {action}")

    # Ejecutar la acción
    try:
        state, reward, done, info = env.step(action)
    except:
        # Si ocurre un error, simplemente pasa a la siguiente iteración
        print("Invalid action or error. Skipping...")
        continue

    # Mostrar el estado actual y la recompensa obtenida
    print("State:")
    print(state)
    print(f"Score: {info['score']}, Moves: {info['moves']}")
    print("=" * 50)

# ----------------------------  4. Resultados ----------------------------

# Mostrar el puntaje final al terminar el juego
print("Game Over")
print(f"Final Score: {info['score']} out of {env.get_max_score()}")
