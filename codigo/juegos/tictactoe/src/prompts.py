# ======================  Tic-Tac-Toe: prompt del LLM como función de recompensas  ======================
from common import casilla_rc, rival_de

# --------------------  1. Directivas --------------------
# Lo único que cambia entre experimentos con LLM. El LLM responde con una palabra de "categorias"; cada palabra
# corresponde a una llave de "valores" en el config, que da la recompensa. Mismo formato que funcionó en
# Frozen Lake: respuesta categórica y sin líneas numeradas.
DIRECTIVAS = {
    "replica_jugada": {
        "tipo": "jugada",
        "texto": (
            "Califica el turno del agente con una de estas palabras. Si aplican varias, usa la primera de la lista:\n"
            "- GANA: con su jugada, el agente completó tres en línea y ganó la partida.\n"
            "- PIERDE: después de la jugada del agente, el rival completó tres en línea y ganó la partida.\n"
            "- BLOQUEA: antes de la jugada, el rival tenía dos fichas en línea con la tercera casilla vacía, "
            "y el agente jugó justo en esa casilla.\n"
            "- EMPATE: al final del turno el tablero quedó lleno sin ganador.\n"
            "- OTRA: cualquier otra jugada."
        ),
        "categorias": {"GANA": "gana", "PIERDE": "pierde", "BLOQUEA": "bloquea", "EMPATE": "empate", "OTRA": "otra"},
    },
    "replica_resultado": {
        "tipo": "partida",
        "texto": "Responde GANA si la partida la ganó el agente, PIERDE si la ganó el rival y EMPATE si terminó sin ganador.",
        "categorias": {"GANA": "gana", "EMPATE": "empate", "PIERDE": "pierde"},
    },
}

def categorias(directiva, valores):
    """{palabra: recompensa}, con la recompensa de cada palabra tomada de los valores del config."""
    return {palabra: valores[llave] for palabra, llave in DIRECTIVAS[directiva]["categorias"].items()}

def palabra_de(directiva, llave):
    """Llave de la recompensa estándar ('bloquea') -> palabra que debería responder el LLM ('BLOQUEA')."""
    return {v: k for k, v in DIRECTIVAS[directiva]["categorias"].items()}[llave]

# --------------------  2. Plantillas --------------------
_REGLAS = """Eres una función de recompensas para una partida de Tic-Tac-Toe (gato).

REGLAS DEL JUEGO:
- Tablero de 3x3. '{agente}' son las fichas del agente, '{rival}' las del rival y '.' una casilla vacía.
- Las filas y las columnas se cuentan desde 0, empezando arriba a la izquierda.
- Gana quien completa tres fichas en línea (fila, columna o diagonal). Si el tablero se llena sin ganador, es empate.
"""

_PROMPT_JUGADA = _REGLAS + """
TABLERO ANTES DE LA JUGADA DEL AGENTE:
{antes}

JUGADA DEL AGENTE: {jugada}

TABLERO DESPUÉS DE LA JUGADA DEL AGENTE:
{despues}

RESPUESTA DEL RIVAL: {respuesta}

TABLERO AL FINAL DEL TURNO:
{final}

FIN DEL TURNO:
- La partida terminó: {termino}. Ganador: {ganador}.

DIRECTIVA:
{directiva}

Responde solo con una palabra: {opciones}.
"""

_PROMPT_PARTIDA = _REGLAS + """
JUGADAS DE LA PARTIDA, EN ORDEN:
{jugadas}

TABLERO FINAL:
{final}

FIN DE LA PARTIDA:
- Ganador: {ganador}.

DIRECTIVA:
{directiva}

Responde solo con una palabra: {opciones}.
"""

# --------------------  3. Construcción del prompt --------------------
def tablero_texto(board):
    return "\n".join(" ".join(c if c != ' ' else '.' for c in board[r * 3:(r + 1) * 3]) for r in range(3))

def casilla_texto(i):
    r, c = casilla_rc(i)
    return f"fila {r}, columna {c}"

def _ganador_texto(ganador, letra_agente):
    if ganador is None:
        return "ninguno"
    return f"el agente ('{ganador}')" if ganador == letra_agente else f"el rival ('{ganador}')"

def _opciones(directiva, valores):
    palabras = sorted(categorias(directiva, valores).items(), key=lambda kv: -kv[1])
    palabras = [p for p, _ in palabras]
    return ", ".join(palabras[:-1]) + " o " + palabras[-1]

def construir_prompt_jugada(board_antes, accion, board_agente, accion_rival, board_final, letra_agente,
                            terminada, ganador, directiva, valores):
    """Un turno completo: jugada del agente y respuesta del rival (si la partida no terminó antes)."""
    return _PROMPT_JUGADA.format(
        agente=letra_agente, rival=rival_de(letra_agente),
        antes=tablero_texto(board_antes), jugada=casilla_texto(accion), despues=tablero_texto(board_agente),
        respuesta=casilla_texto(accion_rival) if accion_rival is not None else "no jugó, la partida ya había terminado",
        final=tablero_texto(board_final), termino="sí" if terminada else "no",
        ganador=_ganador_texto(ganador, letra_agente),
        directiva=DIRECTIVAS[directiva]["texto"], opciones=_opciones(directiva, valores),
    )

def construir_prompt_partida(historial, board_final, letra_agente, ganador, directiva, valores):
    """historial: [(letra, casilla), ...] en el orden en que se jugaron."""
    jugadas = [f"- {'Agente' if letra == letra_agente else 'Rival'} ('{letra}'): {casilla_texto(i)}" for letra, i in historial]
    return _PROMPT_PARTIDA.format(
        agente=letra_agente, rival=rival_de(letra_agente), jugadas="\n".join(jugadas),
        final=tablero_texto(board_final), ganador=_ganador_texto(ganador, letra_agente),
        directiva=DIRECTIVAS[directiva]["texto"], opciones=_opciones(directiva, valores),
    )
