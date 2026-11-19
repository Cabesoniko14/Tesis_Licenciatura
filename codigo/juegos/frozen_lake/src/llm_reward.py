# ======================  Frozen Lake: recompensa vía LLM (DeepInfra)  ======================
import os
import re
import time
import unicodedata

from openai import OpenAI

# --------------------  1. Cliente y estadísticas --------------------
def crear_cliente(cfg_llm):
    return OpenAI(api_key=os.environ[cfg_llm["api_key_env"]], base_url=cfg_llm["base_url"])

def nuevas_stats():
    return {
        "invocaciones_totales": 0,
        "llamadas_api_totales": 0,
        "fallbacks_totales": 0,
        "errores_api_totales": 0,
        "tiempo_respuesta_total_segundos": 0.0,
        "tokens_entrada_totales": 0,
        "tokens_salida_totales": 0,
    }

# --------------------  2. Regla de conversión --------------------
def _palabras(s):
    """'Sí.' -> ['SI']; quita acentos y signos, pasa a mayúsculas."""
    s = unicodedata.normalize("NFD", s.upper())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.findall(r"[A-Z]+", s)

def _parse_categoria(s, categorias):
    """Primera palabra de la respuesta que sea una categoría válida. None si no hay ninguna."""
    for palabra in _palabras(s):
        if palabra in categorias:
            return palabra, categorias[palabra]
    return None, None

# --------------------  3. Llamada --------------------
def llm_reward(client, prompt, cfg_llm, stats, categorias):
    """Regresa (r, registro). Si no hay categoría válida tras los reintentos, r = fallback."""
    stats["invocaciones_totales"] += 1
    registro = {"respuesta": None, "categoria": None, "r_llm": None, "fallback": False, "intentos": 0,
                "tokens_entrada": 0, "tokens_salida": 0, "latencia_s": 0.0}

    for _ in range(cfg_llm["reintentos"]):
        stats["llamadas_api_totales"] += 1
        registro["intentos"] += 1
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=cfg_llm["modelo"],
                temperature=cfg_llm["temperatura"],
                max_tokens=cfg_llm["max_tokens"],
            )
            latencia = time.perf_counter() - t0
            stats["tiempo_respuesta_total_segundos"] += latencia
            registro["latencia_s"] += latencia

            if resp.usage:
                registro["tokens_entrada"] += resp.usage.prompt_tokens
                registro["tokens_salida"] += resp.usage.completion_tokens
                stats["tokens_entrada_totales"] += resp.usage.prompt_tokens
                stats["tokens_salida_totales"] += resp.usage.completion_tokens

            content = (resp.choices[0].message.content or "").strip()
            registro["respuesta"] = content
            categoria, valor = _parse_categoria(content, categorias)
            if categoria is not None:
                registro["categoria"] = categoria
                registro["r_llm"] = valor
                return valor, registro
        except Exception as e:
            stats["errores_api_totales"] += 1
            print(f"[LLM Error] {e}")

    stats["fallbacks_totales"] += 1
    registro["fallback"] = True
    registro["r_llm"] = cfg_llm["fallback"]
    print(f"[LLM WARN] Respuesta inválida tras {cfg_llm['reintentos']} intentos -> uso {cfg_llm['fallback']} de fallback")
    return cfg_llm["fallback"], registro
