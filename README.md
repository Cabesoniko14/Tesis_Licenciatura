# Tésis de Licenciatura en Ciencia de Datos

<img src="imgs/itam.png" width="500" height="200">


## Tema

Modelos Extensos de Lenguaje como alternativa a la función de recompensas en Q-Learning aplicado a juegos.

## Resumen

Esta tesis busca una alternativa a la definición manual numérica de funciones de recompensas en Q-Learning. Particularmente, se busca explorar con este nuevo método si el ajuste de recompensas mediante lenguaje natural/prompts puede alcanzar un nivel similar de efectividad a una función estática. Los beneficios previstos serían mayor facilidad de ajustes a entrenamientos, facilidad de informar a la función de recompensas, y adaptabilidad a escenarios donde la función de recompensas no es clara.

## Escenarios de prueba

- **TicTacToe**: prueba base de los algoritmos de aprenizaje por refuerzo estándar contra el propuesto. Además, incluye la variante de más de un agente participando.
- **FrozenLake**: el ambiente de Gymnasium para un solo jugador, donde el agente debe llegar a una meta sin caer en obstáculos permite enfrentar distintas variantes a la función de recompensa y un escenario más complejo para pruebas.

## Limitaciones

- Recursos disponibles para gastar en cómputo y APIs de LLMs.
- Acceso a escenarios de prueba con una mayor cantidad de acciones u episodios por cómputo.
- Limitantes en evaluaciones por acción por costos de APIs

