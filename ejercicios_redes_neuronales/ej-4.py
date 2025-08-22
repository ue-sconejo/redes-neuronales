#   4. Clasificación básica: ¿La fruta es pesada o ligera?
import tensorflow as tf
import numpy as np

#  Clasificar frutas como “ligeras” (0) o “pesadas” (1) según su peso.
#  Instrucción: Investigar pesos reales de diferentes frutas y definir un umbral para la clasificación,
#  justificando la elección.

#  Usarenmos como base la tabla que aparece en https://www.calculatorsconversion.com/es/calculo-del-peso-de-frutas-y-verduras-por-unidad-2/
pesos_frutas = [ 182, 150, 100, 120, 135, 150, 190, 160, 130, 76, 120, 140, 180, 5, 200, 250, 2, 150, 250, 80]
pesos_modelo = np.array(pesos_frutas, dtype=float)

# Hacemos que si la fruta pese mas de 150g se considere pesada
etiquetas = np.array( [1 if peso >= 150 else 0 for peso in pesos_frutas], dtype=float)

# Creacion del modelo
layer = tf.keras.layers.Dense(units=1, input_shape=[1], activation='sigmoid')
modelo = tf.keras.Sequential([layer])

# En lugar de mean_squared_error usamos binary_crossentropy ya que utilizamos como parametros 0 y 1
# para catalogar las frutas, binary_crossentropy es mas optimo para etiquetas binarias como en este caso
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Entrenamiento del modelo
modelo.fit(pesos_modelo, etiquetas, epochs=500, verbose=0)

# Probar el modelo con nuevos pesos
nuevos_pesos = np.array([20, 30, 155, 190])
predicciones = modelo.predict(nuevos_pesos)

# Realizar las comparaciones
for peso, pred in zip(nuevos_pesos, predicciones):
    clase = 1 if pred >= 0.5 else 0
    print(f"Fruta con Peso de: {peso}g / Se cataloga como: {'Pesada' if clase == 1 else 'Ligera'} ")

# binary_crossentropy es una función de pérdida usada en clasificación binaria. 
# Su objetivo es medir qué tan bien el modelo predice una clase binaria (0 o 1).