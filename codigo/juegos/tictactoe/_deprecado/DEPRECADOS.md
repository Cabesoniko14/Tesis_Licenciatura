# Deprecados — Tic-Tac-Toe

Movido el 2026-11-17 a `_deprecado/2026-11-17/`. Nada se borró: todo sigue en el repo y en el historial de git.
Las corridas se reemplazan por la matriz T del plan de acción. Las marcadas como fuente de figuras pueden citarse como exploratorias.

Los scripts que generaron las Figs. 4.1–4.4 ya no estaban en la carpeta; están en el commit `da7c1ab` (2025-10-21): `git show da7c1ab:codigo/juegos/tictactoe/tictactoe_q_std_random.py`.

## Código

| Archivo | Qué era | Por qué se depreca | Lo reemplaza |
| --- | --- | --- | --- |
| `tictactoe_q_std_random.py` | Q-Learning estándar corregido (sep 2026) | Base del nuevo código; el agente siempre mueve primero | `src/train_qlearning.py --config T0` |
| `tictactoe_q_deepinfra_random.py` | Q-Learning + DeepInfra, "réplica" por jugada | El −1 lo pone el código, no el LLM; el prompt define −1 distinto al estándar | `src/train_qlearning.py --config T1` |
| `tictactoe_q_groq_random.py` | Q-Learning + Groq `openai/gpt-oss-20b` | Otro modelo; sin corridas guardadas | — |
| `tictactoe_q_std_vs_llm_random.py` | Std y LLM lado a lado con la misma semilla | Sin corridas guardadas; la idea pasa al registro r_std / r_llm | `src/train_qlearning.py` |
| `tictactoe_q_std_vs_llm_action_rwd_mapping.py` | Traza de 5 partidas; el LLM ve el turno completo | Prototipo del diseño nuevo | `src/trace_prompt.py` |
| `tictactoe_deepqn_std_random.py` | **Copia exacta de `tictactoe_q_std_random.py`** (solo cambian saltos de línea) | No hay DQN estándar para TTT | — |
| `tictactoe_deepqn_groq_random.py` | DQN + Groq `llama-3.1-8b-instant` | `api_key` vacía, modelo deprecado por Groq; sin corridas | — |
| `tictactoe_deepqn_deepinfra_random.py` | Vacío (0 líneas) | — | — |
| `testing_ui/streamlit_tictactoe.py` | Streamlit para jugar contra la Q-table | Busca archivos con nombres viejos | Se adapta a `runs/` si se necesita |

## Corridas

| Corrida (patrón en el nombre) | Qué era | Nota | Reemplazo | Archivos |
| --- | --- | --- | --- | --- |
| `qlearning_std_random_block_20251021_015440` | Q-Learning, recompensa estándar (oct 2025) | **Alimenta las Figs. 4.1–4.4.** 76.9% vic. Bugs: nunca aplicó −1 en derrotas (0 de 1,183), Q se actualizaba con el tablero a medio turno, el agente siempre movía primero | T0 | 5 |
| `qlearning_groq_random_block_20251021_020328` | Q-Learning + Groq `llama-3.1-8b-instant` (oct 2025) | **Alimenta las Figs. 4.1–4.4.** 58.2% vic. No es réplica: prompt de "bloquear al oponente", solo enteros (nunca dio 0.3), sin `temperature`, fallback sin registrar | T1 | 5 |
| `sinLLM_random_blockBonus_20260916_183819` | Q-Learning estándar corregido (sep 2026) | 81.8% vic.; recompensas 100% consistentes con su función | T0 | 7 |
| `sinLLM_random_blockBonus_20260916_185300` | Q-Learning estándar corregido (sep 2026) | Repetición de la anterior (80.9%) | — | 7 |
| `conLLM_random_llmReplicaStd_20260916_194112` | Q-Learning + LLM, 5 episodios | Prueba de humo; 22 de 22 llamadas en fallback | — | 7 |
| `conLLM_random_llmReplicaStd_20260916_224632` | Q-Learning + LLM, 200 episodios | Prueba; modelo no registrado en el nombre | — | 7 |
| `conLLM_random_llmReplicaStd_20260916_225510` | Q-Learning + LLM, 100 episodios | Prueba; modelo no registrado en el nombre | — | 7 |
| `Instruct-Turbo_random_llmReplicaStd_20260916_233534` | Q-Learning + DeepInfra Llama 3.1 8B, 100 ep. | Prueba | — | 7 |
| `Instruct-Turbo_random_llmReplicaStd_20260917_031742` | Q-Learning + DeepInfra Llama 3.1 8B, réplica | Incompleta (50 de 100 épocas). Acuerdo con la recompensa estándar: 23% de 21,051 jugadas; dio 0.3 a 10,934 jugadas que valían 0. Win rate plano ~50% | T1 | 4 |
| `traza_5_juegos` | Trazas de 5 partidas (Llama 3.1 8B y DeepSeek V4.1 Flash) | Ningún modelo reprodujo la recompensa estándar por jugada | `trace_prompt.py` | 6 |

## Lista completa de archivos

**`qlearning_std_random_block_20251021_015440`**
- `datos_output/acciones_tictactoe_qlearning_std_random_block_20251021_015440_episodes_10000.csv`
- `datos_output/computo_tictactoe_qlearning_std_random_block_20251021_015440_episodes_10000.csv`
- `datos_output/resumen_tictactoe_qlearning_std_random_block_20251021_015440_episodes_10000.csv`
- `datos_output/victorias_tictactoe_qlearning_std_random_block_20251021_015440_episodes_10000.csv`
- `modelos/qtable_tictactoe_qlearning_std_random_block_20251021_015440_episodes_10000.npy`

**`qlearning_groq_random_block_20251021_020328`**
- `datos_output/acciones_tictactoe_qlearning_groq_random_block_20251021_020328_episodes_10000.csv`
- `datos_output/computo_tictactoe_qlearning_groq_random_block_20251021_020328_episodes_10000.csv`
- `datos_output/resumen_tictactoe_qlearning_groq_random_block_20251021_020328_episodes_10000.csv`
- `datos_output/victorias_tictactoe_qlearning_groq_random_block_20251021_020328_episodes_10000.csv`
- `modelos/qtable_tictactoe_qlearning_groq_random_block_20251021_020328_episodes_10000.npy`

**`sinLLM_random_blockBonus_20260916_183819`**
- `datos_output/acciones_tictactoe_QLearning_sinLLM_random_blockBonus_20260916_183819_ep10000.csv`
- `datos_output/computo_tictactoe_QLearning_sinLLM_random_blockBonus_20260916_183819_ep10000.csv`
- `datos_output/resumen_tictactoe_QLearning_sinLLM_random_blockBonus_20260916_183819_ep10000.csv`
- `datos_output/victorias_tictactoe_QLearning_sinLLM_random_blockBonus_20260916_183819_ep10000.csv`
- `modelos/qtable_tictactoe_QLearning_sinLLM_random_blockBonus_20260916_183819_ep10000.npy`
- `resumen_modelos/resumen_tictactoe_QLearning_sinLLM_random_blockBonus_20260916_183819_ep10000.json`
- `resumen_modelos/resumen_tictactoe_QLearning_sinLLM_random_blockBonus_20260916_183819_ep10000.txt`

**`sinLLM_random_blockBonus_20260916_185300`**
- `datos_output/acciones_tictactoe_QLearning_sinLLM_random_blockBonus_20260916_185300_ep10000.csv`
- `datos_output/computo_tictactoe_QLearning_sinLLM_random_blockBonus_20260916_185300_ep10000.csv`
- `datos_output/resumen_tictactoe_QLearning_sinLLM_random_blockBonus_20260916_185300_ep10000.csv`
- `datos_output/victorias_tictactoe_QLearning_sinLLM_random_blockBonus_20260916_185300_ep10000.csv`
- `modelos/qtable_tictactoe_QLearning_sinLLM_random_blockBonus_20260916_185300_ep10000.npy`
- `resumen_modelos/resumen_tictactoe_QLearning_sinLLM_random_blockBonus_20260916_185300_ep10000.json`
- `resumen_modelos/resumen_tictactoe_QLearning_sinLLM_random_blockBonus_20260916_185300_ep10000.txt`

**`conLLM_random_llmReplicaStd_20260916_194112`**
- `datos_output/acciones_tictactoe_QLearning_conLLM_random_llmReplicaStd_20260916_194112_ep5.csv`
- `datos_output/computo_tictactoe_QLearning_conLLM_random_llmReplicaStd_20260916_194112_ep5.csv`
- `datos_output/resumen_tictactoe_QLearning_conLLM_random_llmReplicaStd_20260916_194112_ep5.csv`
- `datos_output/victorias_tictactoe_QLearning_conLLM_random_llmReplicaStd_20260916_194112_ep5.csv`
- `modelos/qtable_tictactoe_QLearning_conLLM_random_llmReplicaStd_20260916_194112_ep5.npy`
- `resumen_modelos/resumen_tictactoe_QLearning_conLLM_random_llmReplicaStd_20260916_194112_ep5.json`
- `resumen_modelos/resumen_tictactoe_QLearning_conLLM_random_llmReplicaStd_20260916_194112_ep5.txt`

**`conLLM_random_llmReplicaStd_20260916_224632`**
- `datos_output/acciones_tictactoe_QLearning_conLLM_random_llmReplicaStd_20260916_224632_ep200.csv`
- `datos_output/computo_tictactoe_QLearning_conLLM_random_llmReplicaStd_20260916_224632_ep200.csv`
- `datos_output/resumen_tictactoe_QLearning_conLLM_random_llmReplicaStd_20260916_224632_ep200.csv`
- `datos_output/victorias_tictactoe_QLearning_conLLM_random_llmReplicaStd_20260916_224632_ep200.csv`
- `modelos/qtable_tictactoe_QLearning_conLLM_random_llmReplicaStd_20260916_224632_ep200.npy`
- `resumen_modelos/resumen_tictactoe_QLearning_conLLM_random_llmReplicaStd_20260916_224632_ep200.json`
- `resumen_modelos/resumen_tictactoe_QLearning_conLLM_random_llmReplicaStd_20260916_224632_ep200.txt`

**`conLLM_random_llmReplicaStd_20260916_225510`**
- `datos_output/acciones_tictactoe_QLearning_conLLM_random_llmReplicaStd_20260916_225510_ep100.csv`
- `datos_output/computo_tictactoe_QLearning_conLLM_random_llmReplicaStd_20260916_225510_ep100.csv`
- `datos_output/resumen_tictactoe_QLearning_conLLM_random_llmReplicaStd_20260916_225510_ep100.csv`
- `datos_output/victorias_tictactoe_QLearning_conLLM_random_llmReplicaStd_20260916_225510_ep100.csv`
- `modelos/qtable_tictactoe_QLearning_conLLM_random_llmReplicaStd_20260916_225510_ep100.npy`
- `resumen_modelos/resumen_tictactoe_QLearning_conLLM_random_llmReplicaStd_20260916_225510_ep100.json`
- `resumen_modelos/resumen_tictactoe_QLearning_conLLM_random_llmReplicaStd_20260916_225510_ep100.txt`

**`Instruct-Turbo_random_llmReplicaStd_20260916_233534`**
- `datos_output/acciones_tictactoe_QLearning_conLLM_Meta-Llama-3.1-8B-Instruct-Turbo_random_llmReplicaStd_20260916_233534_ep100.csv`
- `datos_output/computo_tictactoe_QLearning_conLLM_Meta-Llama-3.1-8B-Instruct-Turbo_random_llmReplicaStd_20260916_233534_ep100.csv`
- `datos_output/resumen_tictactoe_QLearning_conLLM_Meta-Llama-3.1-8B-Instruct-Turbo_random_llmReplicaStd_20260916_233534_ep100.csv`
- `datos_output/victorias_tictactoe_QLearning_conLLM_Meta-Llama-3.1-8B-Instruct-Turbo_random_llmReplicaStd_20260916_233534_ep100.csv`
- `modelos/qtable_tictactoe_QLearning_conLLM_Meta-Llama-3.1-8B-Instruct-Turbo_random_llmReplicaStd_20260916_233534_ep100.npy`
- `resumen_modelos/resumen_tictactoe_QLearning_conLLM_Meta-Llama-3.1-8B-Instruct-Turbo_random_llmReplicaStd_20260916_233534_ep100.json`
- `resumen_modelos/resumen_tictactoe_QLearning_conLLM_Meta-Llama-3.1-8B-Instruct-Turbo_random_llmReplicaStd_20260916_233534_ep100.txt`

**`Instruct-Turbo_random_llmReplicaStd_20260917_031742`**
- `datos_output/acciones_tictactoe_QLearning_conLLM_Meta-Llama-3.1-8B-Instruct-Turbo_random_llmReplicaStd_20260917_031742_ep10000.csv`
- `datos_output/computo_tictactoe_QLearning_conLLM_Meta-Llama-3.1-8B-Instruct-Turbo_random_llmReplicaStd_20260917_031742_ep10000.csv`
- `datos_output/victorias_tictactoe_QLearning_conLLM_Meta-Llama-3.1-8B-Instruct-Turbo_random_llmReplicaStd_20260917_031742_ep10000.csv`
- `modelos/qtable_tictactoe_QLearning_conLLM_Meta-Llama-3.1-8B-Instruct-Turbo_random_llmReplicaStd_20260917_031742_ep10000.npy`

**`traza_5_juegos`**
- `datos_output/traza_5_juegos_llm_vs_std_sombra_20260917_064447.txt`
- `datos_output/traza_5_juegos_llm_vs_std_sombra_20260917_065821.txt`
- `datos_output/traza_5_juegos_llm_vs_std_sombra_20260917_070907.txt`
- `datos_output/traza_5_juegos_llm_vs_std_sombra_DeepSeek-V4.1-Flash_20260917_071533.txt`
- `datos_output/traza_5_juegos_llm_vs_std_sombra_DeepSeek-V4.1-Flash_20260917_071837.txt`
- `datos_output/traza_5_juegos_std_vs_llm_20260917_062423.txt`
