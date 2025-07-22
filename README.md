# ViZDoom RL Project 🧠🎮

Este proyecto entrena agentes de aprendizaje por refuerzo profundo (DQN y variantes) en entornos del juego [ViZDoom](https://vizdoom.cs.put.edu.pl/) utilizando distintos tipos de entradas visuales y variables internas del juego. El objetivo es evaluar cómo distintas representaciones de la observación (procesamiento de imágenes y entradas adicionales) afectan el rendimiento del agente.

---

## 🚀 Objetivo del Proyecto

Entrenar agentes inteligentes que puedan aprender a jugar escenarios de ViZDoom mediante técnicas de Aprendizaje por Refuerzo Profundo. Se evalúan tres configuraciones diferentes para las observaciones que recibe el agente:

1. **Canny + Escala de Grises**
   (`config_defend_the_center.yaml`)
2. **Canny + Escala de Grises + Mapa de Profundidad**
   (`config_defend_the_center_profundidad.yaml`)
3. **(2) + Variables del juego** (como vida y munición) y arquitectura recurrente
   (`config_defend_the_center_recurrente.yaml`)

---

## 🧩 Estructura del Repositorio

```
vizdoom_rl_project/
├── config/                              # Configuraciones YAML de entrenamiento
│   ├── config_defend_the_center.yaml
│   ├── config_defend_the_center_profundidad.yaml
│   └── config_defend_the_center_recurrente.yaml
├── env/                                 # Definición de entorno personalizado (wrapper Gym)
├── models/                              # Modelos de red neuronal
├── checkpoints/                         # Pesos guardados de modelos entrenados
├── utils/                               # Herramientas auxiliares: memoria de experiencia, visualizaciones
├── main.py                              # Script principal de entrenamiento
├── eval.py                              # Evaluación de agentes entrenados
├── requirements.txt                     # Dependencias del proyecto
├── propuesta.md                         # Documento con idea inicial del proyecto
└── README.md                            # Este archivo
```

---

## 🛠️ Requisitos

Este proyecto ha sido probado con:

* **Python 3.9**
* **ViZDoom**
* **PyTorch**
* **OpenCV**
* **NumPy**
* **Matplotlib**
* **tqdm**

Instala todas las dependencias con:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Instalación

1. Clona el repositorio:

```bash
git clone https://github.com/smindiol/vizdoom_rl_project.git
cd vizdoom_rl_project
```

2. Crea y activa un entorno virtual (opcional pero recomendado):

```bash
python -m venv pviz
# En Linux/macOS:
source pviz/bin/activate
# En Windows:
pviz\Scripts\activate.bat
```

3. Instala las dependencias:

```bash
pip install -r requirements.txt
```

---

## 🧪 Cómo Probar el Proyecto

Una vez instalado, puedes ejecutar una evaluación rápida con un modelo ya entrenado (por defecto se usa el de profundidad):

```bash
python eval.py --config config/config_defend_the_center_profundidad.yaml --model checkpoints/modelo_entrenado.pth
```

---

## 🏋️‍♂️ Cómo Entrenar un Agente

Puedes entrenar un agente con:

```bash
python main.py --config config/config_defend_the_center.yaml
```

Otras configuraciones disponibles:

* `config/config_defend_the_center.yaml`: utiliza imágenes en escala de grises + bordes Canny.
* `config/config_defend_the_center_profundidad.yaml`: añade mapa de profundidad.
* `config/config_defend_the_center_recurrente.yaml`: añade variables internas del juego (vida, munición) y usa red recurrente.

Ejemplo:

```bash
python main.py --config config/config_defend_the_center_profundidad.yaml
python main.py --config config/config_defend_the_center_recurrente.yaml
```

Los modelos se guardarán automáticamente en la carpeta `checkpoints/`.

---

## 🧠 Detalles Técnicos

* El entorno está basado en Gym, usando una clase personalizada que envuelve ViZDoom.
* Las imágenes de entrada son procesadas con OpenCV para extraer bordes (Canny) y/o mapas de profundidad.
* Se usa un frame-skip para acelerar el entrenamiento.
* Algunas arquitecturas usan mecanismos de atención espacial y redes recurrentes para mejorar la toma de decisiones.
* Las variables del juego (vida, munición) se integran como entradas adicionales a la red.

---

## 📊 Evaluación

El rendimiento del agente se evalúa promediando la recompensa total por episodio en distintos escenarios, comparando los resultados obtenidos con distintas entradas (observaciones simples vs enriquecidas).

---


## 👨‍💻 Autor

Desarrollado por [Sebastian Mindiola](https://github.com/smindiol)
Repositorio: [https://github.com/smindiol/vizdoom\_rl\_project](https://github.com/smindiol/vizdoom_rl_project)
