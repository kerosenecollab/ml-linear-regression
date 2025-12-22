import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 8, 27, 64, 125])

poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)
y_pred = model.predict(x_poly)

plt.scatter(x, y)
plt.plot(x, y_pred)
plt.show()


