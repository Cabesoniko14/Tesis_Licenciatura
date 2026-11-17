# Tic-Tac-Toe

- `src/`       código único (juego, oponente aleatorio, Q-Learning, prompts, recompensa LLM, evaluación)
- `configs/`   un archivo por experimento: T0, T1, …
- `runs/<run_id>/`  todo lo que produce una corrida: config, jugadas, episodios, épocas, llamadas al LLM, Q-table
- `analisis/`  script de figuras; `analisis/figs/` las figuras de la tesina
- `_deprecado/` código y corridas anteriores (ver `_deprecado/DEPRECADOS.md`)

Experimentos (propuesta, se confirma en la sesión de auditoría de TTT):

| ID | Agente | Recompensa | Cuándo se llama al LLM |
|----|--------|-----------|------------------------|
| T0 | Q-Learning tabular | Estándar: +1 gana, +0.3 bloquea, 0 otra, −1 si el rival gana después | — |
| T1 | Q-Learning tabular | LLM réplica de T0 | Por jugada, viendo el turno completo |
| T2 | Q-Learning tabular | LLM réplica del resultado (+1 / 0 / −1) | Una vez al final de la partida |
