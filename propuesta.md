# Propuesta de Proyecto: Entrenamiento de un Agente RL Basado en Visión para ViZDoom - *Defend the Center*

## 🎯 Objetivo General

Desarrollar un agente inteligente que aprenda a jugar el escenario **Defend the Center** de ViZDoom exclusivamente a partir de la información visual (frames del entorno), empleando aprendizaje por refuerzo profundo (Deep Reinforcement Learning, DRL). El enfoque se centra en simular un agente que toma decisiones únicamente en base a lo que "ve", como lo haría un humano, sin acceso a variables internas del entorno.

## 🧠 Enfoque Técnico

Se usará una red neuronal similar a DQN (*Deep Q-Network*) con posibles mejoras como atención espacial. La entrada de la red será una imagen preprocesada del entorno y la salida será un conjunto discreto de acciones posibles.

## 🛠 Herramientas y Tecnologías

- **ViZDoom**: Entorno de simulación basado en Doom para tareas de RL.
- **PyTorch**: Framework de aprendizaje profundo usado para definir y entrenar las redes neuronales.
- **OpenCV y NumPy**: Procesamiento de imágenes y manejo eficiente de datos.
- **Gym Interface**: Adaptación del entorno ViZDoom a la interfaz de `gym` para facilitar el diseño de agentes.
- **Clase `GrayscaleProcessor` (procesamiento visual)**: Módulo clave que encapsula el preprocesamiento visual.

---

## 👁️ Procesamiento Visual

Una parte esencial del proyecto es la **transformación de las imágenes crudas del entorno** en representaciones útiles para el agente. Aquí es donde entra la clase `GrayscaleProcessor`, la cual no solo convierte las imágenes a escala de grises, sino que también realiza:

- **Redimensionamiento de imagen**: Adaptación a la forma esperada por la red (ej. 100x160 píxeles).
- **Normalización**: Escalado de intensidades a valores entre 0 y 1.
- **Stacking de frames**: Apilamiento temporal de múltiples frames para capturar el movimiento (similar al marco de Atari DQN).
- **Ajustes opcionales** como detección de contornos, mapas de calor, o simplificación del canal de color.

Este enfoque modular permite fácilmente incorporar otras técnicas de visión, como:

- Mapas de atención (ej. saliency maps).
- Resaltado de enemigos con segmentación semántica o etiquetas del entorno (si se usan).
- Canales auxiliares generados por modelos como YOLO (para detección).

---

## 🤖 Arquitectura del Agente

- **Red DQN o DQN con Atención** (`DQNWithAttention`)
- **Entrada**: Imagen procesada (grayscale, shape `[C, H, W]`)
- **Salida**: Q-valores para cada acción posible.
- **Entrenamiento**:
  - Política ε-greedy.
  - Replay memory.
  - Target network.
  - Posible uso de priorización en el muestreo de experiencias.
