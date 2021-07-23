# https://vitalflux.com/learning-curves-explained-python-sklearn-example/
# https://www.geeksforgeeks.org/validation-curve/

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve
from sklearn import datasets
import matplotlib.pyplot as plt

dataset = datasets.load_digits()
X, y = dataset.data, dataset.target

param_range = np.arange(1, 20, 1)

train_score, test_score = validation_curve(KNeighborsClassifier(), X, y,
    param_name="n_neighbors",
    param_range=param_range,
    cv=10, scoring="r2")

mean_train_score = np.mean(train_score, axis = 1)
std_train_score = np.std(train_score, axis = 1)

mean_test_score = np.mean(test_score, axis = 1)
std_test_score = np.std(test_score, axis = 1)

plt.plot(param_range, mean_train_score,
     label = "Training Score", color = 'b')
plt.plot(param_range, mean_test_score,
   label = "Cross Validation Score", color = 'g')

plt.title("Validation Curve with KNN Classifier")
plt.xlabel("Number of Neighbours")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.legend(loc = 'best')
plt.show()