# Deprecados — juegos fuera de la tesina

Movido el 2026-11-17 a `codigo/juegos/_deprecado/2026-11-17/`. Nada se borró: todo sigue en el repo y en el historial de git.

La tesina solo reporta Tic-Tac-Toe y Frozen Lake. Estos juegos fueron pruebas exploratorias.

| Juego | Qué había | Último commit | Estado |
| --- | --- | --- | --- |
| `connect_four/` | DQN estándar contra oponente aleatorio, óptimo y self-play (3 scripts) | 2025-10-23 "trial unsuccesful" | Sin resultados concluyentes. La tesina lo menciona en §3.2.1 y promete resultados que no aparecen: corregir ese párrafo |
| `jericho/` | Zork con DQN estándar, DQN + Groq, v2 y aleatorio; incluye `zork1-3.z5` | 2024-12-13 | Exploratorio (juego de texto); no aparece en la tesina |
| `shopping/` | Tienda simulada con DQN estándar, DQN + Groq y DQN + GPT-4 | 2025-10-22 | Exploratorio; no aparece en la tesina |

## Visualizaciones

`codigo/visualizaciones/viz_generator.ipynb` y `imagenes/g1–g5` se movieron a `codigo/visualizaciones/_deprecado/2026-11-17/`. Generaban las Figs. 4.1–4.6 desde archivos que ya están deprecados. Cada juego tendrá su propio `analisis/figuras.py`; `codigo/visualizaciones/` queda para el estilo compartido de las gráficas.

## Lista completa de archivos

**`connect_four/`** (25 archivos)
- `connect_four/connectfour_deepqn_std_optimal.py`
- `connect_four/connectfour_deepqn_std_random.py`
- `connect_four/connectfour_deepqn_std_selfplay.py`
- `connect_four/datos_output/acciones_connect4_dqn_selfplay_target_20251023_010502_episodes_5000.csv`
- `connect_four/datos_output/acciones_connect4_dqn_standard_random_20251022_235403_episodes_5000.csv`
- `connect_four/datos_output/acciones_connect4_dqn_vs_heuristic_20251023_012455_episodes_5000.csv`
- `connect_four/datos_output/computo_connect4_dqn_selfplay_target_20251023_010502_episodes_5000.csv`
- `connect_four/datos_output/computo_connect4_dqn_standard_random_20251022_235403_episodes_5000.csv`
- `connect_four/datos_output/computo_connect4_dqn_vs_heuristic_20251023_012455_episodes_5000.csv`
- `connect_four/datos_output/resumen_connect4_dqn_selfplay_target_20251023_010502_episodes_5000.csv`
- `connect_four/datos_output/resumen_connect4_dqn_standard_random_20251022_235403_episodes_5000.csv`
- `connect_four/datos_output/resumen_connect4_dqn_vs_heuristic_20251023_012455_episodes_5000.csv`
- `connect_four/datos_output/victorias_connect4_dqn_selfplay_target_20251023_010502_episodes_5000.csv`
- `connect_four/datos_output/victorias_connect4_dqn_standard_random_20251022_235403_episodes_5000.csv`
- `connect_four/datos_output/victorias_connect4_dqn_vs_heuristic_20251023_012455_episodes_5000.csv`
- `connect_four/modelos/meta_connect4_dqn_selfplay_target_20251023_010502_episodes_5000.json`
- `connect_four/modelos/meta_connect4_dqn_standard_random_20251022_235403_episodes_5000.json`
- `connect_four/modelos/meta_connect4_dqn_vs_heuristic_20251023_012455_episodes_5000.json`
- `connect_four/modelos/policy_connect4_dqn_selfplay_target_20251023_010502_episodes_5000.pth`
- `connect_four/modelos/policy_connect4_dqn_standard_random_20251022_235403_episodes_5000.pth`
- `connect_four/modelos/policy_connect4_dqn_vs_heuristic_20251023_012455_episodes_5000.pth`
- `connect_four/modelos/target_connect4_dqn_selfplay_target_20251023_010502_episodes_5000.pth`
- `connect_four/modelos/target_connect4_dqn_standard_random_20251022_235403_episodes_5000.pth`
- `connect_four/modelos/target_connect4_dqn_vs_heuristic_20251023_012455_episodes_5000.pth`
- `connect_four/testing_ui/streamlit_connectfour.py`

**`jericho/`** (45 archivos)
- `jericho/datos_output/data_zork_20241212_212902.csv`
- `jericho/datos_output/data_zork_20241212_213532.csv`
- `jericho/datos_output/data_zork_20241212_215113.csv`
- `jericho/datos_output/data_zork_20241212_215205.csv`
- `jericho/datos_output/data_zork_20241212_215526.csv`
- `jericho/datos_output/data_zork_20241212_215858.csv`
- `jericho/datos_output/data_zork_20241212_220529.csv`
- `jericho/datos_output/data_zork_20241212_221209.csv`
- `jericho/jericho-game-suite/zork1.z5`
- `jericho/jericho-game-suite/zork2.z5`
- `jericho/jericho-game-suite/zork3.z5`
- `jericho/logs/log_zork_deepqn_20241203_213849_200_episodes.txt`
- `jericho/logs/log_zork_deepqn_20241203_222528_1000_episodes.txt`
- `jericho/logs/log_zork_deepqn_standard_10_episodes.txt`
- `jericho/logs/log_zork_deepqnv2_20241204_151729_epochs_50.txt`
- `jericho/logs/log_zork_deepqnv2_20241204_162644_epochs_50.txt`
- `jericho/logs/log_zork_deepqnv2_20241204_180858_epochs_50.txt`
- `jericho/logs/log_zork_deepqnv2_20241209_010956_epochs_50.txt`
- `jericho/logs/log_zork_deepqnv2_20241211_020452_epochs_50.txt`
- `jericho/logs/log_zork_deepqnv2_20241211_021323_epochs_50.txt`
- `jericho/logs/log_zork_deepqnv2_20241212_210119_epochs_50_llm.txt`
- `jericho/logs/log_zork_deepqnv2_20241212_210222_epochs_50_llm.txt`
- `jericho/logs/log_zork_deepqnv2_20241212_210739_epochs_50.txt`
- `jericho/logs/log_zork_deepqnv2_20241212_211103_epochs_50.txt`
- `jericho/logs/log_zork_deepqnv2_20241212_211302_epochs_50.txt`
- `jericho/logs/log_zork_deepqnv2_20241212_211444_epochs_50.txt`
- `jericho/logs/log_zork_deepqnv2_20241212_212133_epochs_50.txt`
- `jericho/logs/log_zork_deepqnv2_20241212_212902_epochs_50.txt`
- `jericho/logs/log_zork_deepqnv2_20241212_213532_epochs_50.txt`
- `jericho/logs/log_zork_deepqnv2_20241212_215113_epochs_50.txt`
- `jericho/logs/log_zork_deepqnv2_20241212_215205_epochs_50.txt`
- `jericho/logs/log_zork_deepqnv2_20241212_215526_epochs_50.txt`
- `jericho/logs/log_zork_deepqnv2_20241212_215858_epochs_50.txt`
- `jericho/logs/log_zork_deepqnv2_20241212_220529_epochs_50.txt`
- `jericho/logs/log_zork_deepqnv2_20241212_221209_epochs_20.txt`
- `jericho/modelos/model__zork_deepqnv2_20241209_010956_epochs_50.pth`
- `jericho/modelos/model__zork_deepqnv2_20241211_021323_epochs_50.pth`
- `jericho/modelos/model_zork_deepqn_20241203_222528_1000_episodes.pth`
- `jericho/resultados/viz_zork_avg_rewards_20241211_021323_epochs_50.png`
- `jericho/resultados/viz_zork_deepqn_20241203_222528_1000_episodes.png`
- `jericho/resultados/viz_zork_deepqnv2_20241209_010956_epochs_50.png`
- `jericho/zork_deepqn_groq.py`
- `jericho/zork_deepqn_std.py`
- `jericho/zork_deepqn_v2.py`
- `jericho/zork_randomchoice.py`

**`shopping/`** (21 archivos)
- `shopping/datos_output/acciones_shopping_deepqn_gpt4_20250921_011609_episodes_100.csv`
- `shopping/datos_output/acciones_shopping_deepqn_groq_20250920_234751_episodes_100.csv`
- `shopping/datos_output/acciones_shopping_deepqn_std_20250921_005154_episodes_100.csv`
- `shopping/datos_output/computo_shopping_deepqn_gpt4_20250921_011609_episodes_100.csv`
- `shopping/datos_output/computo_shopping_deepqn_groq_20250920_234751_episodes_100.csv`
- `shopping/datos_output/computo_shopping_deepqn_std_20250921_005154_episodes_100.csv`
- `shopping/datos_output/resumen_shopping_deepqn_gpt4_20250921_011609_episodes_100.csv`
- `shopping/datos_output/resumen_shopping_deepqn_groq_20250920_234751_episodes_100.csv`
- `shopping/datos_output/resumen_shopping_deepqn_std_20250921_005154_episodes_100.csv`
- `shopping/logs/log_shopping_deepqn_gpt4_20250921_011609_episodes_100.txt`
- `shopping/logs/log_shopping_deepqn_groq_20250920_234751_episodes_100.txt`
- `shopping/logs/log_shopping_deepqn_std_20250921_005154_episodes_100.txt`
- `shopping/modelos/policy_net_shopping_deepqn_groq_20250920_234751_episodes_100.pth`
- `shopping/modelos/policy_net_shopping_deepqn_std_20250921_005154_episodes_100.pth`
- `shopping/modelos/policy_shopping_deepqn_gpt4_20250921_011609_episodes_100.pth`
- `shopping/modelos/target_net_shopping_deepqn_groq_20250920_234751_episodes_100.pth`
- `shopping/modelos/target_net_shopping_deepqn_std_20250921_005154_episodes_100.pth`
- `shopping/modelos/target_shopping_deepqn_gpt4_20250921_011609_episodes_100.pth`
- `shopping/shop_deepqn_gpt4.py`
- `shopping/shop_deepqn_groq.py`
- `shopping/shop_deepqn_std.py`
