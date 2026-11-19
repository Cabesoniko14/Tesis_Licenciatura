# Frozen Lake

- `src/`       código necesario para hacer las ejecuciones usando módulos
- `configs/`   un JSON por experimento conteniendo el detalle de cada ejecución
- `runs/<run_id>/`  outputs por ejecución de cada juego, conteniendo: config, transiciones, episodios, épocas, llamadas al LLM, modelo y figuras generales (`figs/`)
- `analisis/`  notebook de análisis con todos los experimentos y las figuras generales de cada ejecución en `analisis/figs/` 
- `_deprecado/` código deprecado por juegos no usados o experimentos anteriores (ver `_deprecado/DEPRECADOS.md`)

Todos: mapa 4x4 no resbaloso, máximo 50 pasos, 100 épocas × 100 episodios, semilla fija.
LLM: (DeepInfra, Llama 3.1 8B, temperatura 0) se llama una vez al final de cada episodio (nunca por cada paso).

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
