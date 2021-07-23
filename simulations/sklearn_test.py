from towardsDataScienceExample import NUM_MODELS
import numpy as np
import matplotlib.pyplot as plt
from random import randint, seed, sample
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

NUM_OBS = 150

#x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
x = np.array(sample(range(1,50), 6)).reshape(-1,1)
y = np.array([5, 20, 14, 32, 22, 38])

print(x)
print(x.shape)
print(y)
print(y.shape)

model = LinearRegression().fit(x, y)

r_sq = model.score(x,y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')