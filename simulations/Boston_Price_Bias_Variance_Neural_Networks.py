# https://machinelearningmastery.com/calculate-the-bias-variance-trade-off/

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn import datasets
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
        test_size=0.2, random_state=1)
    mse, bias, var = bias_variance_decomp(
        MLPRegressor(hidden_layer_sizes=test_size, max_iter=1000),
        X_train, y_train, X_test, y_test, loss='mse',
        num_rounds=200, random_seed=1)
    errors.append(mse)
    biases.append(bias)
    variances.append(var)
    print('Estimator : Neural Networks')
    print('Test Size :', test_size)
    print('MSE: %.3f' % mse)
    print('Bias: %.3f' % bias)
    print('Variance: %.3f' % var)
    print('--------------------------------')


dataset = datasets.load_boston()

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
print(df.head(3))
print(df.describe())

X = dataset.data
y = dataset.target

degrees = np.arange(1, 50, 5)
errors = []
biases = []
variances = []
for degree in degrees : 
    calculate_mse_bias_variance(X, y, degree)
render_bias_variance(degrees, errors, biases, variances)