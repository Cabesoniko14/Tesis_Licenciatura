# ======================  Frozen Lake: directivas del estudio y su respuesta correcta  ======================
# Cada variante es una forma de pedirle al LLM que evalúe un episodio. Para cada una se puede calcular en código
# la respuesta correcta ("esperado"), así que también sirve de oráculo para entrenar con la misma regla sin LLM.
# Las variantes se agrupan en familias; las variantes de una familia ven exactamente los mismos episodios.
from common import ACCION_NOMBRE, desc_a_filas, idx_to_rc, tile_at

NOMBRE_CASILLA = {"S": "inicio", "F": "hielo seguro", "H": "hoyo", "G": "meta"}

# --------------------  1. Propiedades de un episodio --------------------
# ep = (desc, estados, acciones, terminated, truncated, clase)
def llego(ep):
    return bool(ep[3]) and tile_at(ep[0], ep[1][-1]) == "G"

def es_borde(r, c, H, W):
    return r in (0, H - 1) or c in (0, W - 1)

def casillas_interior(ep):
    """Número de casillas distintas del interior que pisó el agente."""
    desc, estados = ep[0], ep[1]
    H, W = desc.shape
    return len({s for s in estados if not es_borde(*idx_to_rc(s, W), H, W)})

def repite_casilla(ep):
    return len(set(ep[1])) < len(ep[1])

# --------------------  2. Familias: qué episodios se usan y cómo se clasifican --------------------
def _clase_si_no(cumple):
    return lambda ep: ("META_SI" if cumple(ep) else "META_NO") if llego(ep) else "NO_META"

def _clase_borde(ep):
    if not llego(ep):
        return "NO_META"
    n = casillas_interior(ep)
    return "META_BORDE" if n == 0 else ("META_1_INTERIOR" if n == 1 else "META_2+_INTERIOR")

FAMILIAS = {
    "resultado": {"clases": ["META", "NO_META"], "clase": lambda ep: "META" if llego(ep) else "NO_META"},
    "pasos":     {"clases": ["META_SI", "META_NO", "NO_META"], "clase": _clase_si_no(lambda ep: len(ep[2]) <= 8)},
    "repetir":   {"clases": ["META_SI", "META_NO", "NO_META"], "clase": _clase_si_no(lambda ep: not repite_casilla(ep))},
    "borde":     {"clases": ["META_BORDE", "META_1_INTERIOR", "META_2+_INTERIOR", "NO_META"], "clase": _clase_borde},
}

# --------------------  3. Variantes --------------------
BORDE = "(primera o última fila, primera o última columna; todas las demás casillas son el interior)"
FINAL_PALABRA = "Responde solo con una palabra: {opciones}."
FINAL_RAZONAR = ("Primero escribe en una línea las casillas del interior del mapa que pisó el agente, o 'ninguna'. "
                 "Después, en otra línea, escribe RESPUESTA: SI o RESPUESTA: NO.")

def _binaria(si, no, cumple):
    return lambda ep: si if (llego(ep) and cumple(ep)) else no

def _graduada_borde(ep):
    if not llego(ep):
        return "FALLO"
    n = casillas_interior(ep)
    return "EXCELENTE" if n == 0 else ("BIEN" if n == 1 else "REGULAR")

VARIANTES = {
    "V1 réplica": {
        "familia": "resultado",
        "directiva": ("Responde SI si el agente llegó a la meta 'G'. "
                      "Responde NO en cualquier otro caso (cayó en un hoyo o se acabaron los pasos)."),
        "categorias": ["NO", "SI"], "exito": ["SI"],
        "esperado": _binaria("SI", "NO", lambda ep: True),
    },
    "V2a ≤8 pasos, pasos numerados": {
        "familia": "pasos",
        "directiva": "Responde SI solo si el agente llegó a la meta 'G' en 8 pasos o menos. Responde NO en cualquier otro caso.",
        "categorias": ["NO", "SI"], "exito": ["SI"],
        "esperado": _binaria("SI", "NO", lambda ep: len(ep[2]) <= 8),
    },
    "V2b ≤8 pasos, sin numerar": {
        "familia": "pasos", "numerado": False,
        "directiva": "Responde SI solo si el agente llegó a la meta 'G' en 8 pasos o menos. Responde NO en cualquier otro caso.",
        "categorias": ["NO", "SI"], "exito": ["SI"],
        "esperado": _binaria("SI", "NO", lambda ep: len(ep[2]) <= 8),
    },
    "V3 sin repetir casillas": {
        "familia": "repetir",
        "directiva": ("Responde SI solo si el agente llegó a la meta 'G' sin volver nunca a una casilla en la que ya había estado. "
                      "Responde NO en cualquier otro caso: si no llegó a la meta, o si pisó alguna casilla más de una vez."),
        "categorias": ["NO", "SI"], "exito": ["SI"],
        "esperado": _binaria("SI", "NO", lambda ep: not repite_casilla(ep)),
    },
    "V4a borde graduado": {
        "familia": "borde",
        "directiva": ("Califica la ruta del agente con una de estas palabras:\n"
                      "- FALLO: no llegó a la meta 'G' (cayó en un hoyo o se acabaron los pasos).\n"
                      "- REGULAR: llegó a la meta, pero pisó dos o más casillas del interior del mapa.\n"
                      "- BIEN: llegó a la meta y pisó exactamente una casilla del interior del mapa.\n"
                      f"- EXCELENTE: llegó a la meta y todas las casillas de su camino están en el borde del mapa {BORDE}."),
        "categorias": ["FALLO", "REGULAR", "BIEN", "EXCELENTE"], "exito": ["EXCELENTE"],
        "esperado": _graduada_borde,
    },
    "V4b borde SI/NO": {
        "familia": "borde",
        "directiva": (f"Responde SI solo si el agente llegó a la meta 'G' y todas las casillas de su camino están en el borde del mapa {BORDE}. "
                      "Responde NO en cualquier otro caso: si no llegó a la meta, o si pisó alguna casilla del interior."),
        "categorias": ["NO", "SI"], "exito": ["SI"],
        "esperado": _binaria("SI", "NO", lambda ep: casillas_interior(ep) == 0),
    },
    "V4c borde BIEN/MAL": {
        "familia": "borde",
        "directiva": (f"Responde BIEN solo si el agente llegó a la meta 'G' y todas las casillas de su camino están en el borde del mapa {BORDE}. "
                      "Responde MAL en cualquier otro caso: si no llegó a la meta, o si pisó alguna casilla del interior."),
        "categorias": ["MAL", "BIEN"], "exito": ["BIEN"],
        "esperado": _binaria("BIEN", "MAL", lambda ep: casillas_interior(ep) == 0),
    },
    "V4d borde razonando, SI/NO": {
        "familia": "borde", "final": FINAL_RAZONAR, "marcador": "RESPUESTA", "max_tokens": 150,
        "directiva": (f"Responde SI solo si el agente llegó a la meta 'G' y todas las casillas de su camino están en el borde del mapa {BORDE}. "
                      "Responde NO en cualquier otro caso: si no llegó a la meta, o si pisó alguna casilla del interior."),
        "categorias": ["NO", "SI"], "exito": ["SI"],
        "esperado": _binaria("SI", "NO", lambda ep: casillas_interior(ep) == 0),
    },
}

def categorias(variante):
    """{palabra: recompensa}. Binaria: 0 y 1. Graduada: la primera vale 0 y las demás van de 0.6 a 1.0 (igual que prompts.py)."""
    lista = VARIANTES[variante]["categorias"]
    exitos = lista[1:]
    if len(exitos) == 1:
        return {lista[0]: 0.0, exitos[0]: 1.0}
    paso = 0.4 / (len(exitos) - 1)
    return {lista[0]: 0.0, **{c: round(0.6 + i * paso, 4) for i, c in enumerate(exitos)}}

# --------------------  4. Prompt (misma plantilla que prompts.py) --------------------
_PROMPT = """Eres una función de recompensas para un episodio de FrozenLake.

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

{final}
"""

def construir_prompt(ep, variante):
    v = VARIANTES[variante]
    desc, estados, acciones, terminated, truncated = ep[:5]
    H, W = desc.shape
    r0, c0 = idx_to_rc(estados[0], W)
    camino = [f"Inicio: fila {r0}, columna {c0} ('{tile_at(desc, estados[0])}')"]
    for i, (a, s) in enumerate(zip(acciones, estados[1:]), start=1):
        r, c = idx_to_rc(s, W)
        prefijo = f"Paso {i}:" if v.get("numerado", True) else "-"
        camino.append(f"{prefijo} {ACCION_NOMBRE[int(a)]} -> fila {r}, columna {c} ('{tile_at(desc, s)}')")
    palabras = v["categorias"][::-1]
    opciones = ", ".join(palabras[:-1]) + " o " + palabras[-1]
    casilla_final = tile_at(desc, estados[-1])
    return _PROMPT.format(H=H, W=W, mapa="\n".join(desc_a_filas(desc)), camino="\n".join(camino),
                          termino="sí" if terminated else "no", trunco="sí" if truncated else "no",
                          casilla_final=casilla_final, nombre_final=NOMBRE_CASILLA[casilla_final],
                          directiva=v["directiva"], final=v.get("final", FINAL_PALABRA).format(opciones=opciones))
