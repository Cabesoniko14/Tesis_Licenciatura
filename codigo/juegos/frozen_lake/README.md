# Frozen Lake

- `src/`       código (entorno, red, prompts, recompensa LLM, entrenamiento, traza)
- `configs/`   un JSON por experimento; el código no se edita entre corridas
- `runs/<run_id>/`  todo lo que produce una corrida: config, transiciones, episodios, épocas, llamadas al LLM, modelo y figuras generales (`figs/`)
- `analisis/`  notebook de análisis con todos los experimentos; `analisis/figs/` las figuras de la tesina
- `_deprecado/` código y corridas anteriores (ver `_deprecado/DEPRECADOS.md`)

Todos: mapa 4x4 no resbaloso, máximo 50 pasos, 100 épocas × 100 episodios, semilla 42, CPU determinista.
El LLM (DeepInfra, Llama 3.1 8B, temperatura 0) se llama una vez al final de cada episodio.

| ID | Mapa | Agente | Recompensa |
|----|------|--------|-----------|
| A0 | Fijo | Q-Learning tabular | Gym |
| A1 | Fijo | DQN | Gym |
| A2 | Fijo | DQN | LLM réplica |
| B0 | Aleatorio por episodio | Q-Learning tabular (opcional) | Gym |
| B1 | Aleatorio por episodio | DQN | Gym |
| B2 | Aleatorio por episodio | DQN | LLM réplica |
| B3 | Aleatorio por episodio | DQN | LLM: preferir el borde |
| B4 | Aleatorio por episodio | DQN | LLM: ruta más segura |
