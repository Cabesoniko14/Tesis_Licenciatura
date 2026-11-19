# ======================  Frozen Lake: recompensa vía LLM (DeepInfra)  ======================
import os
import re
import time

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
def _parse_reward_value(s):
    """Primer número de la respuesta, recortado a [0, 1]. None si no hay número."""
    m = re.search(r"[-+]?\d*\.?\d+", s.strip())
    if not m:
        return None
    return max(0.0, min(1.0, float(m.group(0))))

# --------------------  3. Llamada --------------------
def llm_reward(client, prompt, cfg_llm, stats):
    """Regresa (r, registro). Si no hay número tras los reintentos, r = fallback."""
    stats["invocaciones_totales"] += 1
    registro = {"respuesta": None, "r_llm": None, "fallback": False, "intentos": 0,
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
            parsed = _parse_reward_value(content)
            if parsed is not None:
                registro["r_llm"] = parsed
                return parsed, registro
        except Exception as e:
            stats["errores_api_totales"] += 1
            print(f"[LLM Error] {e}")

    stats["fallbacks_totales"] += 1
    registro["fallback"] = True
    registro["r_llm"] = cfg_llm["fallback"]
    print(f"[LLM WARN] Respuesta inválida tras {cfg_llm['reintentos']} intentos -> uso {cfg_llm['fallback']} de fallback")
    return cfg_llm["fallback"], registro
