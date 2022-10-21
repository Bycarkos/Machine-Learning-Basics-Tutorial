import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ice = pd.read_csv('SeaIce.txt', delim_whitespace = True)
ice2 = ice[ice.data_type != '-9999']
print(ice2.head())
print('shape:', ice2.shape)

from sklearn import preprocessing

# from sklearn.linear_model import LinearRegression as Regressor
from sklearn.linear_model import SGDRegressor as Regressor
# from regressor import Regressor

est = Regressor()

x = ice2[['year']].values
y = ice2[['extent']].values.ravel()

x = preprocessing.scale(x)
y = preprocessing.scale(y)


est.fit(x, y)
print("Coefficients:", est.coef_)
print("Intercept:", est.intercept_)
print("Score:", est.score(x, y))

from sklearn import metrics

# Analysis for all months together.
x = ice2[['year']].values
y = ice2[['extent']].values.ravel()

x = preprocessing.scale(x)
y = preprocessing.scale(y)

model = Regressor()
model.fit(x, y)
y_hat = model.predict(x)
plt.plot(x, y,'o', alpha = 0.5)
plt.plot(x, y_hat, 'r', alpha = 0.5)
plt.xlabel('year')
plt.ylabel('extent (All months)')
print("MSE:", metrics.mean_squared_error(y_hat, y))
print("R^2:", metrics.r2_score(y_hat, y))
print("var:", y.var())
plt.show()

# Analysis for a particular month.
jan = ice2[ice2.mo == 1]
x = jan[['year']].values
y = jan[['extent']].values.ravel()

x = preprocessing.scale(x)
y = preprocessing.scale(y)

model = Regressor()
model.fit(x, y)

y_hat = model.predict(x)

plt.figure()
plt.plot(x, y,'-o', alpha = 0.5)
plt.plot(x, y_hat, 'r', alpha = 0.5)
plt.xlabel('year')
plt.ylabel('extent (January)')
plt.show()
print("MSE:", metrics.mean_squared_error(y_hat, y))
print("R^2:", metrics.r2_score(y_hat, y))
print("Coefficients:", est.coef_)
print("Intercept:", est.intercept_)
print("Score:", est.score(x, y))
