import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('/home/lenovo/Desktop/temperatures.csv')
X=dataset[["YEAR"]]
y=dataset[["ANNUAL"]]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.show()
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.show()
