# Tic-Tac-Toe

- `src/`       código (juego, oponente aleatorio, agentes, prompts, recompensa LLM, entrenamiento, traza)
- `configs/`   un JSON por experimento; el código no se edita entre corridas
- `runs/<run_id>/`  todo lo que produce una corrida: config, jugadas, partidas, épocas, llamadas al LLM, modelo y figuras generales (`figs/`)
- `analisis/`  notebook de análisis con todos los experimentos; `analisis/figs/` las figuras de la tesina
- `_deprecado/` código y corridas anteriores (ver `_deprecado/DEPRECADOS.md`)

Todos: oponente aleatorio, el agente siempre mueve primero, 100 épocas × 100 partidas, semilla 42, CPU determinista.
LLM: DeepInfra, Llama 3.1 8B, temperatura 0.

| ID | Agente | Recompensa | Cuándo se llama al LLM |
|----|--------|-----------|------------------------|
| T0 | Q-Learning tabular | Estándar: +1 gana, +0.3 bloquea, 0 otra, −1 si el rival gana después | — |
| T1 | Q-Learning tabular | LLM réplica de T0 | Por jugada, viendo el turno completo |
| T2 | Q-Learning tabular | LLM: +1 gana / 0 empata / −1 pierde | Al final de la partida |
| T3 | DQN | Estándar | — |
| T4 | DQN | LLM: +1 gana / 0 empata / −1 pierde | Al final de la partida |
