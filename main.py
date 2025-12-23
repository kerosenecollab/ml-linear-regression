import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Данные
x = np.array([[0], [1], [2], [3], [4], [5]])
y = np.array([5, 8, 11, 14, 17, 20])

# Модель
model = LinearRegression()
model.fit(x, y)

# Предсказание для всех точек (исходные + новые)
x_new = np.array([[6], [7], [8], [9]])
x_all = np.vstack((x, x_new))
y_all_pred = model.predict(x_all)

# График
plt.scatter(x, y, color='blue', label='Исходные точки')   # исходные точки
plt.scatter(x_new, model.predict(x_new), color='red', label='Новые точки')  # новые точки
plt.plot(x_all, y_all_pred, color='green', label='Линия регрессии')  # одна линия через все

plt.legend()
plt.show()
