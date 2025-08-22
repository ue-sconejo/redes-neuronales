#    5. Predicción de gastos mensuales según ingresos
import tensorflow as tf
import numpy as np

# Estimar los gastos mensuales de una persona en base a su ingreso mensual.

# Como datos de entrenamiento usaremos Regional Cost of Living Analysis
# sacado de kaggle https://www.kaggle.com/datasets/heidarmirhajisadati/regional-cost-of-living-analysis
# Que es un data set con los ingresos y costo de vida de diferentes paises 
# se seleccionaron los datos de 2023 como referencia

ingreso = np.array([1657.72, 3379.18, 3011.79, 2363.67, 3505.72, 4872.81, 1545.17, 2542.3, 7458.14], dtype=float)
costo_vida = np.array([5347.7, 1266.22, 1226.23, 2448.39, 5238.55, 5260.25, 3401.28, 597.83, 2742.98], dtype=float)

# Creamos el modelo y agregamos 2 capas
layer = tf.keras.layers.Dense(units=3, input_shape =[1])
output = tf.keras.layers.Dense(units=1)

modelo = tf.keras.Sequential([layer, output])

# Usamos el optimizer SGD (Stochastic Gradient Descent) con un ajuste de 0.01
# SGD es optimo para modelos simples
modelo.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    loss='mean_squared_error'
)

# Entrenar el modelo
modelo.fit(ingreso, costo_vida, epochs=500, verbose=0)

# Prueba del modelo con 1500 USD
dato_entrada = 1500
prediccion = modelo.predict(np.array([dato_entrada]))

print(f"Para unos ingresos de {dato_entrada} $USD, el posible gasto sera de {prediccion[0][0]:.3f} $USD")

#  Qué es Adam y qué otros optimizadores existen en TensorFlow.
# Adam (Adaptive Moment Estimation) es un optimizador que 
# adapta la tasa de aprendizaje para cada parámetro usando estimaciones de primer y segundo momento
# esto ayuda a que Adam sea eficiente y confiable en varias tareas, especialmente con datos ruidosos

# Aparte de Adam existen SGD (Stochastic Gradient Descent), RMSProp, Adagrad, Adadelta, Nadam, Ftrl