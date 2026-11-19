# ======================  Frozen Lake: prompt del LLM como función de recompensas  ======================
from common import ACCION_NOMBRE, desc_a_filas, idx_to_rc, tile_at

NOMBRE_CASILLA = {"S": "inicio", "F": "hielo seguro", "H": "hoyo", "G": "meta"}

# --------------------  1. Directivas --------------------
# Lo único que cambia entre experimentos con LLM. El LLM responde con una palabra de "categorias",
# ordenadas de menos a más valiosa. La primera vale 0 (no llegó a la meta) y las demás se reparten en
# partes iguales entre "minimo_exito" y 1.0. Con respuestas numéricas, Llama 3.1 8B tendía a responder 0
# aunque el agente ganara (ver runs/estudio_prompts_*).
DIRECTIVAS = {
    "replica": {
        "texto": (
            "Responde SI si el agente llegó a la meta 'G'. "
            "Responde NO en cualquier otro caso (cayó en un hoyo o se acabaron los pasos)."
        ),
        "categorias": ["NO", "SI"],
        "minimo_exito": 1.0,
    },
    "borde": {
        "texto": (
            "Califica la ruta del agente con una de estas palabras:\n"
            "- FALLO: no llegó a la meta 'G' (cayó en un hoyo o se acabaron los pasos).\n"
            "- REGULAR: llegó a la meta, pero casi toda la ruta va por el interior del mapa.\n"
            "- BIEN: llegó a la meta y buena parte de la ruta va por el borde del mapa "
            "(primera o última fila, primera o última columna).\n"
            "- EXCELENTE: llegó a la meta y casi toda la ruta va por el borde del mapa."
        ),
        "categorias": ["FALLO", "REGULAR", "BIEN", "EXCELENTE"],
        "minimo_exito": 0.6,
    },
    "segura": {
        "texto": (
            "Califica la ruta del agente con una de estas palabras:\n"
            "- FALLO: no llegó a la meta 'G' (cayó en un hoyo o se acabaron los pasos).\n"
            "- REGULAR: llegó a la meta, pero pasó junto a hoyos 'H' muchas veces "
            "(arriba, abajo, izquierda o derecha de un hoyo).\n"
            "- BIEN: llegó a la meta y pasó junto a hoyos pocas veces.\n"
            "- EXCELENTE: llegó a la meta sin pasar nunca junto a un hoyo."
        ),
        "categorias": ["FALLO", "REGULAR", "BIEN", "EXCELENTE"],
        "minimo_exito": 0.6,
    },
}

def categorias(directiva):
    """{palabra: recompensa}. La primera categoría vale 0; las demás van de minimo_exito a 1.0."""
    lista = DIRECTIVAS[directiva]["categorias"]
    minimo = DIRECTIVAS[directiva]["minimo_exito"]
    exitos = lista[1:]
    paso = (1.0 - minimo) / (len(exitos) - 1) if len(exitos) > 1 else 0.0
    return {lista[0]: 0.0, **{c: round(minimo + i * paso, 4) for i, c in enumerate(exitos)}}

# --------------------  2. Plantilla --------------------
# Sin líneas numeradas (00:, 01:, ...): con ellas el LLM se equivocaba más, con cualquier formato de respuesta.
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

Responde solo con una palabra: {opciones}.
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
    palabras = DIRECTIVAS[directiva]["categorias"][::-1]   # en el prompt van de más a menos valiosa

    return _PROMPT_TEMPLATE.format(
        H=H, W=W, mapa="\n".join(desc_a_filas(desc)), camino="\n".join(camino),
        termino="sí" if terminated else "no", trunco="sí" if truncated else "no",
        casilla_final=casilla_final, nombre_final=NOMBRE_CASILLA[casilla_final],
        directiva=DIRECTIVAS[directiva]["texto"],
        opciones=", ".join(palabras[:-1]) + " o " + palabras[-1],
    )
