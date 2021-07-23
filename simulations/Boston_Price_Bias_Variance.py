# https://machinelearningmastery.com/calculate-the-bias-variance-trade-off/

import numpy as np
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import datasets
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

from mlxtend.evaluate import bias_variance_decomp

def render_bias_variance(test_sizes, errors, biases, variances) :
    plt.plot(test_sizes, errors, color='blue', marker='o',
        markersize=5, label='MSE')
    plt.plot(test_sizes, biases, color='green', marker='+',
        markersize=5, label='Bias')
    plt.plot(test_sizes, variances, color='red', marker='x',
        markersize=5, label='Variance')
    plt.title('Bias-Variance')
    plt.xlabel('Training Data Size')
    plt.ylabel('MSE total')
    plt.grid()
    plt.legend(loc='upper left')
    plt.show()
    plt.figure()

def calculate_mse_bias_variance(X, y, test_size) :
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=test_size, random_state=1)
    mse, bias, var = bias_variance_decomp(pipeline,
        X_train, y_train, X_test, y_test, loss='mse',
        num_rounds=200, random_seed=1)
    errors.append(mse)
    biases.append(bias)
    variances.append(var)
    print('Estimator :', estimator_names[i])
    print('Test Size :', test_size)
    print('MSE: %.3f' % mse)
    print('Bias: %.3f' % bias)
    print('Variance: %.3f' % var)
    print('--------------------------------')


dataset = datasets.load_boston()
X = dataset.data
y = dataset.target

pipelines = []
estimator_names = []

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

pipelines.append(LinearRegression())
pipelines.append(make_pipeline(PolynomialFeatures(degree=2), LinearRegression()))
pipelines.append(SVR())
pipelines.append(DecisionTreeRegressor())
pipelines.append(AdaBoostRegressor(loss='square'))
pipelines.append(KNeighborsRegressor())
#pipelines.append(MLPRegressor(solver='lbfgs', max_iter=1000))

estimator_names.append('Linear Regression')
estimator_names.append('Polynomial Regression')
estimator_names.append('Support Vector Machine')
estimator_names.append('Decision Tree')
estimator_names.append('Ada Boost')
estimator_names.append('Nearest Neighbors')
#estimator_names.append('MLP Regressor')

test_sizes = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

for i, pipeline in enumerate(pipelines) :
    errors = []
    biases = []
    variances = []
    for test_size in test_sizes : 
        calculate_mse_bias_variance(X, y, test_size)

    render_bias_variance(test_sizes, errors, biases, variances)