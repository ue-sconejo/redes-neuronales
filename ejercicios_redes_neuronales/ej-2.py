#    2. Predicción de altura de una planta según días de crecimiento

import tensorflow as tf
import numpy as np

# Modelar el crecimiento de una planta usando datos reales o documentados de altura (en cm)
# vs. días desde la germinación.

# Se utilizara como planta un girason ya que al estar erguido por defecto seria mas facil de medir
# y puede llegar a medir hasta 260 cm.
# Para entrenar el modelo se utilizara los datos de H. S. Reed y R. H. Holland 1919 sunflower growth
# Es una tabla con 13 dias recopilando el crecimiento de un girasol
# Url con los datos https://d10lpgp6xz60nq.cloudfront.net/physics_images/COL_BOD_SAT_PT_06_E04_012_Q01.png

dias = np.array([0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84], dtype=float)
altura_cm = np.array([0.0, 17.93, 36.36, 67.76, 98.10, 131.0, 169.5, 205.5, 228.3, 247.1, 250.5, 253.8, 254.5], dtype=float)

# Creamos las capas cada una con 3 neuronas
layer_1 = tf.keras.layers.Dense(units=3, input_shape =[1])

# Capa de salida con una neurona para el resultado final
output = tf.keras.layers.Dense(units=1)

# Agregamos las capad a el modelo secuancial
modelo = tf.keras.Sequential([layer_1, output])

modelo.compile(
   optimizer=tf.keras.optimizers.Adam(0.1),
   loss='mean_squared_error'
)

# Entrenar el modelo
modelo.fit(dias, altura_cm, epochs=500, verbose=0)

# Prueba del modelo
dato_entrada = 17
prediccion = modelo.predict(np.array([dato_entrada]))

print(f"En {dato_entrada} dias el girason crecera {prediccion[0][0]:.3f} cm")

# Pregunta Qué significa overfitting y cómo evitarlo.
# Overfittin sucede cuando un modelo esta sobreentrenado y reconoce muy bien los patrones
# tiene como ventaja que puede ser muy preciso en los resultados pero si se necestia reentrenar este se resistira.

# Como evitarlo: existen tecnicas para evitar el overfitting como ejemplo esta el dropout
# se desactivan aleatoriamente neuronas durante el entrenamiento para evitar que el modelo se acostumbre demasiado.
# EJ: tf.keras.layers.Dropout(0.3)