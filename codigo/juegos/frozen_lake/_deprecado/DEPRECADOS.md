# Deprecados — Frozen Lake

Movido el 2026-11-17 a `_deprecado/2026-11-17/`. Nada se borró: todo sigue en el repo y en el historial de git.
Todas las corridas se reemplazan por la matriz A0–A4 (mapa fijo) y B0–B2 (mapas aleatorios) del plan de acción. Las corridas marcadas con figura pueden citarse como exploratorias.

## Código

| Archivo | Qué era | Por qué se depreca | Lo reemplaza |
| --- | --- | --- | --- |
| `frozenlake_q_std.py` | Q-Learning tabular | Entrena en mapa fijo y evalúa en aleatorio | `src/train_qlearning.py` |
| `frozenlake_deepqn_std.py` | DQN con recompensa de Gym | Base del nuevo código; outputs en formato viejo | `src/train_dqn.py --config A1 / B1` |
| `frozenlake_deepqn_std_gap.py` | DQN con Global Average Pooling | Exploratorio; no está en la tesina | — |
| `frozenlake_deepqn_groq_preferences.py` | DQN + Groq con directivas | `Groq(api_key)` roto; fallback devuelve la recompensa real; no registra fallbacks ni tokens | `src/train_dqn.py --config A2–A4, B2` |
| `frozenlake_deepqn_deepinfra_preferences.py` | DQN + DeepInfra, réplica | Solo réplica; el prompt da el booleano `llego_meta` | `src/train_dqn.py` + `src/llm_reward.py` |
| `frozenlake_deepqn_deepinfra_quickeval_preferences.py` | Traza de 2 episodios | Solo prompt de réplica | `src/trace_prompt.py` |
| `testing_ui/st_frozenlake.py` | Streamlit para probar modelos | Busca modelos por nombre de archivo viejo | Se adapta a `runs/` si se necesita |

## Corridas

| Corrida (patrón en el nombre) | Qué era | Nota | Reemplazo | Archivos |
| --- | --- | --- | --- | --- |
| `qlearning_std_fixedmap_size4_noslip_20251021_052005` | Q-Learning, mapa fijo 4x4 | Usada en Fig. 4.5 contra DQN en mapas aleatorios (no comparable) | A0 | 3 |
| `qlearning_std_fixedmap_size4_slip_20251021_052629` | Q-Learning, mapa fijo, resbaloso | Exploratoria (22% de victorias); fuera de alcance | — | 1 |
| `qlearning_std_fixed_size4_noslip_20251021_063441` | Q-Learning, mapa fijo | Duplicada de la anterior con 5,000 episodios | — | 4 |
| `qlearning_std_per_episode_random_size4_noslip_20251021_060949` | Q-Learning, mapas aleatorios | 36% de victorias (no ve hoyos). Hallazgo útil, se repite limpio | B0 | 6 |
| `dqn_per_episode_random_size4_noslip_20251021_065345` | DQN std, mapas aleatorios | Usada en Fig. 4.5 y 4.6 (75%) | B1 | 7 |
| `dqn_per_episode_random_size4_noslip_20251021_230530` | DQN std, 5,000 ep. | Prueba intermedia | — | 4 |
| `dqn_per_episode_random_size4_noslip_20251022_001006` | DQN std, 5,000 ep. | Prueba intermedia | — | 4 |
| `dqn_gap_` | DQN con Global Average Pooling | Arquitectura exploratoria; no aparece en la tesina | — | 7 |
| `dqn_openai_replicate` | DQN + OpenAI gpt-5-mini, 1 episodio | Script descartado por bugs | — | 8 |
| `dqn_groq_per_episode_random_size4_noslip_20251021_18` | DQN + Groq, 3 y 300 episodios | Pruebas de humo | — | 8 |
| `dqn_groq_replicate_gym_reward` | DQN + Groq, réplica (v1) | Primera versión de la réplica | B2 | 4 |
| `dqn_groq_own_reward_system_fast_route_custom_prompt_adv_directive_nofall` | DQN + Groq, ruta rápida con prompt avanzado | Variante de directiva; no aparece en la tesina | A3–A4 | 4 |
| `dqn_groq_own_reward_system_fast_route_per` | DQN + Groq, ruta rápida (v1) | Primeras pruebas de directiva | A3–A4 | 8 |
| `preferences_basic_finishornot` | DQN + Groq, réplica (v2) | Usada en Fig. 4.6 y 4.7; réplica 100% fiel (2,729/2,729), solo 5,000 ep. | B2 | 7 |
| `preferences_1_shortest_route` | DQN + Groq, ruta más corta | Usada en Fig. 4.8 y 4.9; el LLM dio 0.9 a casi toda victoria | A3 | 7 |
| `preferences_2_safest_route` | DQN + Groq, ruta más segura | No está en figuras; el LLM premió ~730 caídas en hoyos | A4 | 7 |
| `llmReplicaStd_20260917_074103` | DQN + DeepInfra, réplica | Corrida incompleta (sin meta ni target) | — | 4 |
| `llmReplicaStd_20260917_161652` | DQN + DeepInfra, réplica | 74.6%, 9,938 llamadas, 0 fallbacks, fidelidad 100% | B2 | 8 |
| `traza_frozenlake_2_episodios` | Trazas de 2 episodios (quickeval) | Pruebas del prompt de réplica | Herramienta trace_prompt | 3 |

## Lista completa de archivos

**`qlearning_std_fixedmap_size4_noslip_20251021_052005`**
- `datos_output/acciones_frozenlake_qlearning_std_fixedmap_size4_noslip_20251021_052005_episodes_10000.csv`
- `datos_output/computo_frozenlake_qlearning_std_fixedmap_size4_noslip_20251021_052005_episodes_10000.csv`
- `datos_output/victorias_frozenlake_qlearning_std_fixedmap_size4_noslip_20251021_052005_episodes_10000.csv`

**`qlearning_std_fixedmap_size4_slip_20251021_052629`**
- `datos_output/resumen_frozenlake_qlearning_std_fixedmap_size4_slip_20251021_052629_episodes_10000.csv`

**`qlearning_std_fixed_size4_noslip_20251021_063441`**
- `datos_output/acciones_frozenlake_qlearning_std_fixed_size4_noslip_20251021_063441_episodes_5000.csv`
- `datos_output/computo_frozenlake_qlearning_std_fixed_size4_noslip_20251021_063441_episodes_5000.csv`
- `datos_output/resumen_frozenlake_qlearning_std_fixed_size4_noslip_20251021_063441_episodes_5000.csv`
- `datos_output/victorias_frozenlake_qlearning_std_fixed_size4_noslip_20251021_063441_episodes_5000.csv`

**`qlearning_std_per_episode_random_size4_noslip_20251021_060949`**
- `datos_output/acciones_frozenlake_qlearning_std_per_episode_random_size4_noslip_20251021_060949_episodes_30000.csv`
- `datos_output/computo_frozenlake_qlearning_std_per_episode_random_size4_noslip_20251021_060949_episodes_30000.csv`
- `datos_output/resumen_frozenlake_qlearning_std_per_episode_random_size4_noslip_20251021_060949_episodes_30000.csv`
- `datos_output/victorias_frozenlake_qlearning_std_per_episode_random_size4_noslip_20251021_060949_episodes_30000.csv`
- `modelos/meta_frozenlake_qlearning_std_per_episode_random_size4_noslip_20251021_060949_episodes_30000.json`
- `modelos/qtable_frozenlake_qlearning_std_per_episode_random_size4_noslip_20251021_060949_episodes_30000.npy`

**`dqn_per_episode_random_size4_noslip_20251021_065345`**
- `datos_output/acciones_frozenlake_dqn_per_episode_random_size4_noslip_20251021_065345_episodes_10000.csv`
- `datos_output/computo_frozenlake_dqn_per_episode_random_size4_noslip_20251021_065345_episodes_10000.csv`
- `datos_output/resumen_frozenlake_dqn_per_episode_random_size4_noslip_20251021_065345_episodes_10000.csv`
- `datos_output/victorias_frozenlake_dqn_per_episode_random_size4_noslip_20251021_065345_episodes_10000.csv`
- `modelos/meta_frozenlake_dqn_per_episode_random_size4_noslip_20251021_065345_episodes_10000.json`
- `modelos/policy_frozenlake_dqn_per_episode_random_size4_noslip_20251021_065345_episodes_10000.pth`
- `modelos/target_frozenlake_dqn_per_episode_random_size4_noslip_20251021_065345_episodes_10000.pth`

**`dqn_per_episode_random_size4_noslip_20251021_230530`**
- `datos_output/acciones_frozenlake_dqn_per_episode_random_size4_noslip_20251021_230530_episodes_5000.csv`
- `datos_output/computo_frozenlake_dqn_per_episode_random_size4_noslip_20251021_230530_episodes_5000.csv`
- `datos_output/resumen_frozenlake_dqn_per_episode_random_size4_noslip_20251021_230530_episodes_5000.csv`
- `datos_output/victorias_frozenlake_dqn_per_episode_random_size4_noslip_20251021_230530_episodes_5000.csv`

**`dqn_per_episode_random_size4_noslip_20251022_001006`**
- `datos_output/acciones_frozenlake_dqn_per_episode_random_size4_noslip_20251022_001006_episodes_5000.csv`
- `datos_output/computo_frozenlake_dqn_per_episode_random_size4_noslip_20251022_001006_episodes_5000.csv`
- `datos_output/resumen_frozenlake_dqn_per_episode_random_size4_noslip_20251022_001006_episodes_5000.csv`
- `datos_output/victorias_frozenlake_dqn_per_episode_random_size4_noslip_20251022_001006_episodes_5000.csv`

**`dqn_gap_`**
- `datos_output/acciones_frozenlake_dqn_gap_per_episode_random_size4_noslip_20260127_014422_episodes_5000.csv`
- `datos_output/computo_frozenlake_dqn_gap_per_episode_random_size4_noslip_20260127_014422_episodes_5000.csv`
- `datos_output/resumen_frozenlake_dqn_gap_per_episode_random_size4_noslip_20260127_014422_episodes_5000.csv`
- `datos_output/victorias_frozenlake_dqn_gap_per_episode_random_size4_noslip_20260127_014422_episodes_5000.csv`
- `modelos/meta_frozenlake_dqn_gap_per_episode_random_size4_noslip_20260127_014422_episodes_5000.json`
- `modelos/policy_frozenlake_dqn_gap_per_episode_random_size4_noslip_20260127_014422_episodes_5000.pth`
- `modelos/target_frozenlake_dqn_gap_per_episode_random_size4_noslip_20260127_014422_episodes_5000.pth`

**`dqn_openai_replicate`**
- `datos_output/acciones_frozenlake_dqn_openai_replicate_gym_rewards_4_noslip_20251021_222753_episodes_1.csv`
- `datos_output/acciones_frozenlake_dqn_openai_replicate_gym_rewards_4_noslip_20251021_223348_episodes_1.csv`
- `datos_output/computo_frozenlake_dqn_openai_replicate_gym_rewards_4_noslip_20251021_222753_episodes_1.csv`
- `datos_output/computo_frozenlake_dqn_openai_replicate_gym_rewards_4_noslip_20251021_223348_episodes_1.csv`
- `datos_output/resumen_frozenlake_dqn_openai_replicate_gym_rewards_4_noslip_20251021_222753_episodes_1.csv`
- `datos_output/resumen_frozenlake_dqn_openai_replicate_gym_rewards_4_noslip_20251021_223348_episodes_1.csv`
- `datos_output/victorias_frozenlake_dqn_openai_replicate_gym_rewards_4_noslip_20251021_222753_episodes_1.csv`
- `datos_output/victorias_frozenlake_dqn_openai_replicate_gym_rewards_4_noslip_20251021_223348_episodes_1.csv`

**`dqn_groq_per_episode_random_size4_noslip_20251021_18`**
- `datos_output/acciones_frozenlake_dqn_groq_per_episode_random_size4_noslip_20251021_180459_episodes_3.csv`
- `datos_output/acciones_frozenlake_dqn_groq_per_episode_random_size4_noslip_20251021_181112_episodes_300.csv`
- `datos_output/computo_frozenlake_dqn_groq_per_episode_random_size4_noslip_20251021_180459_episodes_3.csv`
- `datos_output/computo_frozenlake_dqn_groq_per_episode_random_size4_noslip_20251021_181112_episodes_300.csv`
- `datos_output/resumen_frozenlake_dqn_groq_per_episode_random_size4_noslip_20251021_180459_episodes_3.csv`
- `datos_output/resumen_frozenlake_dqn_groq_per_episode_random_size4_noslip_20251021_181112_episodes_300.csv`
- `datos_output/victorias_frozenlake_dqn_groq_per_episode_random_size4_noslip_20251021_180459_episodes_3.csv`
- `datos_output/victorias_frozenlake_dqn_groq_per_episode_random_size4_noslip_20251021_181112_episodes_300.csv`

**`dqn_groq_replicate_gym_reward`**
- `datos_output/acciones_frozenlake_dqn_groq_replicate_gym_reward_per_episode_random_size4_noslip_20251022_003712_episodes_5000.csv`
- `datos_output/computo_frozenlake_dqn_groq_replicate_gym_reward_per_episode_random_size4_noslip_20251022_003712_episodes_5000.csv`
- `datos_output/resumen_frozenlake_dqn_groq_replicate_gym_reward_per_episode_random_size4_noslip_20251022_003712_episodes_5000.csv`
- `datos_output/victorias_frozenlake_dqn_groq_replicate_gym_reward_per_episode_random_size4_noslip_20251022_003712_episodes_5000.csv`

**`dqn_groq_own_reward_system_fast_route_custom_prompt_adv_directive_nofall`**
- `datos_output/acciones_frozenlake_dqn_groq_own_reward_system_fast_route_custom_prompt_adv_directive_nofall_per_episode_random_size4_noslip_20260125_154946_episodes_10000.csv`
- `datos_output/computo_frozenlake_dqn_groq_own_reward_system_fast_route_custom_prompt_adv_directive_nofall_per_episode_random_size4_noslip_20260125_154946_episodes_10000.csv`
- `datos_output/resumen_frozenlake_dqn_groq_own_reward_system_fast_route_custom_prompt_adv_directive_nofall_per_episode_random_size4_noslip_20260125_154946_episodes_10000.csv`
- `datos_output/victorias_frozenlake_dqn_groq_own_reward_system_fast_route_custom_prompt_adv_directive_nofall_per_episode_random_size4_noslip_20260125_154946_episodes_10000.csv`

**`dqn_groq_own_reward_system_fast_route_per`**
- `datos_output/acciones_frozenlake_dqn_groq_own_reward_system_fast_route_per_episode_random_size4_noslip_20251022_011526_episodes_5000.csv`
- `datos_output/acciones_frozenlake_dqn_groq_own_reward_system_fast_route_per_episode_random_size4_noslip_20251022_014527_episodes_10000.csv`
- `datos_output/computo_frozenlake_dqn_groq_own_reward_system_fast_route_per_episode_random_size4_noslip_20251022_011526_episodes_5000.csv`
- `datos_output/computo_frozenlake_dqn_groq_own_reward_system_fast_route_per_episode_random_size4_noslip_20251022_014527_episodes_10000.csv`
- `datos_output/resumen_frozenlake_dqn_groq_own_reward_system_fast_route_per_episode_random_size4_noslip_20251022_011526_episodes_5000.csv`
- `datos_output/resumen_frozenlake_dqn_groq_own_reward_system_fast_route_per_episode_random_size4_noslip_20251022_014527_episodes_10000.csv`
- `datos_output/victorias_frozenlake_dqn_groq_own_reward_system_fast_route_per_episode_random_size4_noslip_20251022_011526_episodes_5000.csv`
- `datos_output/victorias_frozenlake_dqn_groq_own_reward_system_fast_route_per_episode_random_size4_noslip_20251022_014527_episodes_10000.csv`

**`preferences_basic_finishornot`**
- `datos_output/acciones_frozenlake_dqn_llm_episodic_preferences_basic_finishornot_per_episode_random_size4_noslip_20260125_173740_episodes_5000.csv`
- `datos_output/computo_frozenlake_dqn_llm_episodic_preferences_basic_finishornot_per_episode_random_size4_noslip_20260125_173740_episodes_5000.csv`
- `datos_output/resumen_frozenlake_dqn_llm_episodic_preferences_basic_finishornot_per_episode_random_size4_noslip_20260125_173740_episodes_5000.csv`
- `datos_output/victorias_frozenlake_dqn_llm_episodic_preferences_basic_finishornot_per_episode_random_size4_noslip_20260125_173740_episodes_5000.csv`
- `modelos/meta_frozenlake_dqn_llm_episodic_preferences_basic_finishornot_per_episode_random_size4_noslip_20260125_173740_episodes_5000.json`
- `modelos/policy_frozenlake_dqn_llm_episodic_preferences_basic_finishornot_per_episode_random_size4_noslip_20260125_173740_episodes_5000.pth`
- `modelos/target_frozenlake_dqn_llm_episodic_preferences_basic_finishornot_per_episode_random_size4_noslip_20260125_173740_episodes_5000.pth`

**`preferences_1_shortest_route`**
- `datos_output/acciones_frozenlake_dqn_llm_episodic_preferences_1_shortest_route_per_episode_random_size4_noslip_20260125_185841_episodes_10000.csv`
- `datos_output/computo_frozenlake_dqn_llm_episodic_preferences_1_shortest_route_per_episode_random_size4_noslip_20260125_185841_episodes_10000.csv`
- `datos_output/resumen_frozenlake_dqn_llm_episodic_preferences_1_shortest_route_per_episode_random_size4_noslip_20260125_185841_episodes_10000.csv`
- `datos_output/victorias_frozenlake_dqn_llm_episodic_preferences_1_shortest_route_per_episode_random_size4_noslip_20260125_185841_episodes_10000.csv`
- `modelos/meta_frozenlake_dqn_llm_episodic_preferences_1_shortest_route_per_episode_random_size4_noslip_20260125_185841_episodes_10000.json`
- `modelos/policy_frozenlake_dqn_llm_episodic_preferences_1_shortest_route_per_episode_random_size4_noslip_20260125_185841_episodes_10000.pth`
- `modelos/target_frozenlake_dqn_llm_episodic_preferences_1_shortest_route_per_episode_random_size4_noslip_20260125_185841_episodes_10000.pth`

**`preferences_2_safest_route`**
- `datos_output/acciones_frozenlake_dqn_llm_episodic_preferences_2_safest_route_per_episode_random_size4_noslip_20260125_194738_episodes_10000.csv`
- `datos_output/computo_frozenlake_dqn_llm_episodic_preferences_2_safest_route_per_episode_random_size4_noslip_20260125_194738_episodes_10000.csv`
- `datos_output/resumen_frozenlake_dqn_llm_episodic_preferences_2_safest_route_per_episode_random_size4_noslip_20260125_194738_episodes_10000.csv`
- `datos_output/victorias_frozenlake_dqn_llm_episodic_preferences_2_safest_route_per_episode_random_size4_noslip_20260125_194738_episodes_10000.csv`
- `modelos/meta_frozenlake_dqn_llm_episodic_preferences_2_safest_route_per_episode_random_size4_noslip_20260125_194738_episodes_10000.json`
- `modelos/policy_frozenlake_dqn_llm_episodic_preferences_2_safest_route_per_episode_random_size4_noslip_20260125_194738_episodes_10000.pth`
- `modelos/target_frozenlake_dqn_llm_episodic_preferences_2_safest_route_per_episode_random_size4_noslip_20260125_194738_episodes_10000.pth`

**`llmReplicaStd_20260917_074103`**
- `datos_output/acciones_frozenlake_DeepQN_conLLM_Meta-Llama-3.1-8B-Instruct-Turbo_per_episode_random_size4_noslip_llmReplicaStd_20260917_074103_ep10000.csv`
- `datos_output/computo_frozenlake_DeepQN_conLLM_Meta-Llama-3.1-8B-Instruct-Turbo_per_episode_random_size4_noslip_llmReplicaStd_20260917_074103_ep10000.csv`
- `datos_output/victorias_frozenlake_DeepQN_conLLM_Meta-Llama-3.1-8B-Instruct-Turbo_per_episode_random_size4_noslip_llmReplicaStd_20260917_074103_ep10000.csv`
- `modelos/policy_frozenlake_DeepQN_conLLM_Meta-Llama-3.1-8B-Instruct-Turbo_per_episode_random_size4_noslip_llmReplicaStd_20260917_074103_ep10000.pth`

**`llmReplicaStd_20260917_161652`**
- `datos_output/acciones_frozenlake_DeepQN_conLLM_Meta-Llama-3.1-8B-Instruct-Turbo_per_episode_random_size4_noslip_llmReplicaStd_20260917_161652_ep10000.csv`
- `datos_output/computo_frozenlake_DeepQN_conLLM_Meta-Llama-3.1-8B-Instruct-Turbo_per_episode_random_size4_noslip_llmReplicaStd_20260917_161652_ep10000.csv`
- `datos_output/victorias_frozenlake_DeepQN_conLLM_Meta-Llama-3.1-8B-Instruct-Turbo_per_episode_random_size4_noslip_llmReplicaStd_20260917_161652_ep10000.csv`
- `modelos/meta_frozenlake_DeepQN_conLLM_Meta-Llama-3.1-8B-Instruct-Turbo_per_episode_random_size4_noslip_llmReplicaStd_20260917_161652_ep10000.json`
- `modelos/policy_frozenlake_DeepQN_conLLM_Meta-Llama-3.1-8B-Instruct-Turbo_per_episode_random_size4_noslip_llmReplicaStd_20260917_161652_ep10000.pth`
- `modelos/target_frozenlake_DeepQN_conLLM_Meta-Llama-3.1-8B-Instruct-Turbo_per_episode_random_size4_noslip_llmReplicaStd_20260917_161652_ep10000.pth`
- `resumen_modelos/resumen_frozenlake_DeepQN_conLLM_Meta-Llama-3.1-8B-Instruct-Turbo_per_episode_random_size4_noslip_llmReplicaStd_20260917_161652_ep10000.json`
- `resumen_modelos/resumen_frozenlake_DeepQN_conLLM_Meta-Llama-3.1-8B-Instruct-Turbo_per_episode_random_size4_noslip_llmReplicaStd_20260917_161652_ep10000.txt`

**`traza_frozenlake_2_episodios`**
- `datos_output/traza_frozenlake_2_episodios_Meta-Llama-3.1-8B-Instruct-Turbo_20260917_073034.txt`
- `datos_output/traza_frozenlake_2_episodios_Meta-Llama-3.1-8B-Instruct-Turbo_20260917_073135.txt`
- `datos_output/traza_frozenlake_2_episodios_Meta-Llama-3.1-8B-Instruct-Turbo_20260917_073213.txt`
