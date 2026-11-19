# ======================  Frozen Lake: prompt del LLM como función de recompensas  ======================
from common import ACCION_NOMBRE, desc_a_filas, idx_to_rc, tile_at

NOMBRE_CASILLA = {"S": "inicio", "F": "hielo seguro", "H": "hoyo", "G": "meta"}

# --------------------  1. Directivas --------------------
# Lo único que cambia entre experimentos con LLM. El resto del prompt es igual para todos.
DIRECTIVAS = {
    "replica": (
        "Da 1 si el agente llegó a la meta 'G'. "
        "Da 0 en cualquier otro caso (cayó en un hoyo o se acabaron los pasos)."
    ),
    "borde": (
        "Si el agente no llegó a la meta 'G', da 0. "
        "Si llegó, da una calificación alta. Prefiere las rutas que van por el borde del mapa "
        "(primera o última fila, primera o última columna): da 1 si casi toda la ruta va por el borde "
        "y resta un poco mientras más celdas del interior use."
    ),
    "segura": (
        "Si el agente no llegó a la meta 'G', da 0. "
        "Si llegó, da una calificación alta. Prefiere las rutas que pasan lejos de los hoyos 'H': "
        "da 1 si la ruta nunca pasa junto a un hoyo y resta un poco por cada celda de la ruta "
        "que esté junto a uno (arriba, abajo, izquierda o derecha)."
    ),
}

# --------------------  2. Plantilla --------------------
# Sin líneas numeradas (00:, 01:, ...): con ellas Llama 3.1 8B respondía 0 de reflejo, incluso en victorias.
_PROMPT_TEMPLATE = """Eres una función de recompensas para un episodio de FrozenLake.

REGLAS DEL ENTORNO:
- Mapa de {H}x{W}. Celdas: 'S' inicio, 'F' hielo seguro, 'H' hoyo, 'G' meta.
- El episodio termina si el agente llega a 'G' o cae en un 'H'. Si se acaban los pasos, se trunca.

MAPA (fila por fila, de arriba hacia abajo):
{mapa}

CAMINO DEL AGENTE:
{camino}

FIN DEL EPISODIO:
- Terminó: {termino}. Se truncó por límite de pasos: {trunco}.
- Casilla final: '{casilla_final}' ({nombre_final})

DIRECTIVA:
{directiva}

Responde solo con un número entre 0 y 1, sin texto extra.
"""

# --------------------  3. Construcción del prompt --------------------
def construir_prompt(desc, estados, acciones, terminated, truncated, directiva):
    """estados: [s0, ..., sT]; acciones: [a0, ..., a(T-1)]; directiva: llave de DIRECTIVAS."""
    H, W = desc.shape
    r0, c0 = idx_to_rc(estados[0], W)
    camino = [f"Inicio: fila {r0}, columna {c0} ('{tile_at(desc, estados[0])}')"]
    for i, (a, s) in enumerate(zip(acciones, estados[1:]), start=1):
        r, c = idx_to_rc(s, W)
        camino.append(f"Paso {i}: {ACCION_NOMBRE[int(a)]} -> fila {r}, columna {c} ('{tile_at(desc, s)}')")
    casilla_final = tile_at(desc, estados[-1])

    return _PROMPT_TEMPLATE.format(
        H=H, W=W, mapa="\n".join(desc_a_filas(desc)), camino="\n".join(camino),
        termino="sí" if terminated else "no", trunco="sí" if truncated else "no",
        casilla_final=casilla_final, nombre_final=NOMBRE_CASILLA[casilla_final],
        directiva=DIRECTIVAS[directiva],
    )
