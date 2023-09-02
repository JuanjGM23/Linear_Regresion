
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import matplotlib
matplotlib.use('TkAgg')

def f(x):
    y = (0.0918 * x) + 1.2859 + 0.1 * np.random.randn(x.shape[0])
    return y
# Crear datos de entrenamiento
x = np.arange(0, 20, 0.25)
print(x.shape)
y = f(x)
print(y.shape)
# Diagrama de dispersión
plt.scatter(x, y, color='red')
# Instanciamos la regresion lineal
linear_regression = LinearRegression()
# Entrenamos el modelo de regresion lineal
linear_regression.fit(x.reshape(-1, 1), y)  # (#muestras, #caracteristicas)
print("w =", linear_regression.coef_, ", b =", linear_regression.intercept_)
# Nueva muestra. Imprime el valor correspondiente de y
new_sample = np.array([5])
print(f(new_sample))
# Predecir la nueva muestra. Imprimir la predicción
prediction = linear_regression.predict(new_sample.reshape(1, -1))
print(prediction)
# Predecir todos los valores de x
predictions = linear_regression.predict(x.reshape(-1, 1))
_, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x, y, color='blue')
ax.plot(x, predictions, color='red')
plt.show()
