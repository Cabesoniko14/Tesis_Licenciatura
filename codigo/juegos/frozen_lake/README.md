# Frozen Lake

- `src/`       código único (entorno, red, prompts, recompensa LLM, entrenamiento, evaluación)
- `configs/`   un archivo por experimento: A0 … A4, B0 … B2
- `runs/<run_id>/`  todo lo que produce una corrida: config, transiciones, episodios, épocas, llamadas al LLM, modelo
- `analisis/`  script de figuras; `analisis/figs/` las figuras de la tesina
- `_deprecado/` código y corridas anteriores (ver `_deprecado/DEPRECADOS.md`)

| ID | Mapa | Agente | Recompensa |
|----|------|--------|-----------|
| A0 | Fijo | Q-Learning tabular | Gym |
| A1 | Fijo | DQN | Gym |
| A2 | Fijo | DQN | LLM réplica |
| A3 | Fijo | DQN | LLM: preferir el borde |
| A4 | Fijo | DQN | LLM: ruta más segura |
| B0 | Aleatorio 4x4 | Q-Learning tabular (opcional) | Gym |
| B1 | Aleatorio 4x4 | DQN | Gym |
| B2 | Aleatorio 4x4 | DQN | LLM réplica |
