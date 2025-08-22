#    3. Relación entre velocidad y distancia de frenado

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Predecir la distancia necesaria para que un vehículo frene completamente según su
#  velocidad inicial.

# Usando de base los datos de MRU (Movimiento Rectilíneo Uniforme)
# Como datos de entrenamiento usaremos la formula m/s^2 con las velocidades en intervalos de 0,40 y 100
# y una desaceleracion de 5
np.random.seed(0)
velocidades = np.linspace(0, 40, 100)
deceleracion = 5.0

# generar distancias que usaremos en el entrenamiento con aletoriedad
distancias = (velocidades ** 2) / (2 * deceleracion) + np.random.normal(0, 1, velocidades.shape)

# Creacion del modelo con una capa que tiene 10 neuronas y una capa de salida con una neurona
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=10, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mse')
history = model.fit(velocidades, distancias, epochs=200, verbose=0)

# Para probar el modelo usaremos las velocidades 10, 20, 30, 40 y 50
vel_test = np.array([10, 20, 30, 40, 50])
dist_pred = model.predict(vel_test)

# Mostrar predicciones
for v, d in zip(vel_test, dist_pred):
    print(f"Velocidad: {v} m/s -> Distancia de frenado: {d[0]:.2f} m")

# Graficar resultados
plt.scatter(velocidades, distancias, label='Datos basado en formula')
plt.plot(vel_test, dist_pred, color='red', label='Predicción del modelo')
plt.xlabel('Velocidad inicial (m/s)')
plt.ylabel('Distancia de frenado (m)')
plt.title('Predicción de distancia de frenado')
plt.legend()
plt.grid(True)
plt.show()

#  Diferencia entre loss y accuracy.
# Loss hace referencia a la forma en la que se le puede indicar al modelo si esta fallando en
# su entrenamiento usando la comparacion de los datos.

# Accuracy es la medida de referencia que indica si el modelo esta dando resultados correctos
# mostrando el rendimiento general.