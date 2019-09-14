import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing, cross_validation, svm
# %matplotlib inline      for jupyter only
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl

df = pd.read_csv('KU2DE.csv', parse_dates=True, index_col=0)
last3days = df.tail(3)  # last 3 days for check if prediction is good
#training = df.drop(df.tail(3).index)
training = df.copy()
print('Initial data length: {}, Training data length: {}'.format(
    len(df), len(df)-3))

forecast_col = 'Adj Close'

training['mavg100'] = training['Adj Close'].rolling(window=100).mean()
training['rets'] = training['Adj Close'] / training['Adj Close'].shift(1) - 1

training['HL_PCT'] = (training['High'] - training['Low']
                      ) / training['Close'] * 100.00
training['PCT_change'] = (
    training['Close'] - training['Open']) / training['Open'] * 100.00

# data prep
training.fillna(value=-99999, inplace=True)

X = np.array(training.drop(['Adj Close'], 1))
Y = np.array(training['Adj Close'])
X1 = preprocessing.scale(X)

Y = Y[:-3]
X = X1[:-3]

X_prediction = X1[-3:]

# Linear regression
linregr = LinearRegression(n_jobs=-1)
linregr.fit(X, Y)
prediction = linregr.predict(X_prediction)
print('Linear regression {}'.format(prediction))
last3days['LinRegr'] = prediction

# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X, Y)
prediction2 = clfpoly2.predict(X_prediction)
print('Quadratic regression 2 {}'.format(prediction2))
last3days['Qreg2'] = prediction2

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X, Y)
prediction3 = clfpoly3.predict(X_prediction)
print('Quadratic regression 3 {}'.format(prediction3))
last3days['Qreg3'] = prediction3

# see the dataframe
print(last3days.head())

# plotting
mpl.rc('figure', figsize=(8, 7))
mpl.__version__
style.use('ggplot')
last3days['Adj Close'].plot(label='Kuka real stock price')
last3days['LinRegr'].plot(label='Linear Regression')
last3days['Qreg2'].plot(label='Quadratic Regression 2')
last3days['Qreg3'].plot(label='Quadratic Regression 3')
plt.legend()
plt.show()
