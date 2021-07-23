# https://vitalflux.com/learning-curves-explained-python-sklearn-example/

import numpy as np
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import learning_curve
from sklearn import datasets
import matplotlib.pyplot as plt

from mlxtend.evaluate import bias_variance_decomp

def Learning_curve_per_estimator(estimator_pipeline, estimator_name) :
    train_sizes, train_scores, test_scores = learning_curve(estimator=estimator_pipeline,
    X=X_train, y=y_train, cv=10, train_sizes = np.linspace(0.1, 1.0, 10), n_jobs=1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, color='blue', marker='o',
    markersize=5, label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', marker='+',
    markersize=5, linestyle='--', label='Validation Accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std,
    alpha=0.15, color='green')
    plt.title('Learning Curve ' + estimator_name)
    plt.xlabel('Training Data Size')
    plt.ylabel('Model Accuracy')
    plt.grid()
    plt.legend(loc='lower right')
    plt.show()
    plt.figure()

dataset = datasets.load_boston()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, stratify=y, random_state=1) #test_size=0.3

pipelines = []
estimator_names = []

pipelines.append(LogisticRegression(penalty='l2',solver='lbfgs', random_state=1, max_iter=10000, fit_intercept=True))
pipelines.append(SGDRegressor())
pipelines.append(SVR())
pipelines.append(DecisionTreeRegressor())
pipelines.append(AdaBoostRegressor(loss='square'))
pipelines.append(KNeighborsRegressor())
pipelines.append(MLPRegressor(solver='lbfgs', max_iter=10000))

estimator_names.append('Linear Regression')
estimator_names.append('Stochastic Gradient Descent')
estimator_names.append('Support Vector Machine')
estimator_names.append('Decision Tree')
estimator_names.append('Ada Boost')
estimator_names.append('Nearest Neighbors')
estimator_names.append('MLP Regressor')

for i, pipeline in enumerate(pipelines) :
    estimator_name = estimator_names[i]
    Learning_curve_per_estimator(estimator_pipeline=make_pipeline(StandardScaler(), pipeline), estimator_name=estimator_name)
