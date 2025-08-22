#   1. Conversión de kilómetros a millas

import tensorflow as tf
import numpy as np

# Datos de entrenamiento (km -> millas)
# Se eligieron saltos de 5 unidades comenzando desde 1.
# Así se cubre un rango medio, evitando usar datos muy pequeños.
# Por último, al usar estos datos obtenemos unidades precisas como 10, 20 y 30.
km = np.array([1, 5, 10, 15, 20, 25, 30], dtype=float)
millas = np.array([0.621, 3.107, 6.214, 9.321, 12.427, 15.534, 18.641], dtype=float)

# Creamos las capas cada una con 3 neuronas
oculta1 = tf.keras.layers.Dense(units=3, input_shape =[1])
oculta2 = tf.keras.layers.Dense(units=3)
# Capa de salida con una neurona para el resultado final
output = tf.keras.layers.Dense(units=1)

# Agregamos las capad a el modelo secuancial
modelo = tf.keras.Sequential([oculta1, oculta2, output])

modelo.compile(
   optimizer=tf.keras.optimizers.Adam(0.1),
   loss='mean_squared_error'
)

# Entrenar el modelo
modelo.fit(km, millas, epochs=500, verbose=0)

# Probar el modelo con entrada de 50 Km
km_nuevo = 50
prediccion = modelo.predict(np.array([km_nuevo]))
print(f"{km_nuevo} Km son aprox {prediccion[0][0]:.3f} millas")