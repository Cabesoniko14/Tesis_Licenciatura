# ======================  Frozen Lake: prompt del LLM como función de recompensas  ======================
from common import ACCION_NOMBRE, desc_a_filas, idx_to_rc, tile_at

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
_PROMPT_TEMPLATE = """Eres una función de recompensas para un episodio de FrozenLake.

REGLAS DEL ENTORNO:
- Mapa de {H}x{W}. Celdas: 'S' inicio, 'F' hielo seguro, 'H' hoyo, 'G' meta.
- Acciones: 0=Left, 1=Down, 2=Right, 3=Up.
- El episodio termina si el agente llega a 'G' o cae en un 'H'. Si se acaban los pasos, se trunca.

MAPA (por filas):
{mapa}

ÍNDICES DE CADA CELDA (por filas):
{indices}

CAMINO DEL AGENTE:
{camino}

FIN DEL EPISODIO:
- terminated={terminated}, truncated={truncated}
- casilla final: '{casilla_final}'

DIRECTIVA:
{directiva}

Responde solo con un número entre 0 y 1, sin texto extra.
"""

# --------------------  3. Construcción del prompt --------------------
def construir_prompt(desc, estados, acciones, terminated, truncated, directiva):
    """estados: [s0, ..., sT]; acciones: [a0, ..., a(T-1)]; directiva: llave de DIRECTIVAS."""
    H, W = desc.shape
    mapa = "\n".join(f"{r:02d}: {fila}" for r, fila in enumerate(desc_a_filas(desc)))
    indices = "\n".join(" ".join(f"{r * W + c:02d}" for c in range(W)) for r in range(H))

    r0, c0 = idx_to_rc(estados[0], W)
    camino = [f"00: inicio -> idx={estados[0]:02d} (r={r0},c={c0}) '{tile_at(desc, estados[0])}'"]
    for i, (a, s) in enumerate(zip(acciones, estados[1:]), start=1):
        r, c = idx_to_rc(s, W)
        camino.append(f"{i:02d}: {ACCION_NOMBRE[int(a)]} -> idx={s:02d} (r={r},c={c}) '{tile_at(desc, s)}'")

    return _PROMPT_TEMPLATE.format(
        H=H, W=W, mapa=mapa, indices=indices, camino="\n".join(camino),
        terminated=bool(terminated), truncated=bool(truncated),
        casilla_final=tile_at(desc, estados[-1]), directiva=DIRECTIVAS[directiva],
    )
