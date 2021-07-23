# https://www.kaggle.com/adityakadiwal/water-potability


#Load libraries
import pandas 
from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

def curve_learning_function(estimator, data, features, target, train_sizes, cv, sub_title, y_lim) : 
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=estimator,
        X = data[features],
        y = data[target], train_sizes=train_sizes, cv = cv,
        scoring= 'neg_mean_squared_error'
        )

    print ('Train sizes:' ,train_sizes)
    print ('Train scores:' ,train_scores)
    print ('Validation scores:' ,validation_scores)

    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis = 1) 
    print('Mean training scores\n\n', pandas.Series(train_scores_mean, index = train_sizes))
    print('\n', '-' * 20) # separator
    print('\nMean validation scores\n\n', pandas.Series(validation_scores_mean, index = train_sizes))

    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
    plt.ylabel('MSE', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    plt.title('Learning curves for a linear regression model ' + sub_title, fontsize = 16, y = 1.03)
    plt.legend()
    plt.ylim(0,y_lim)
    plt.figure()



dataset = read_csv('water_potability.csv')
print(dataset.info())
dataset = dataset.dropna()
print(dataset.describe())

train_sizes = [1, 10, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1608]

featuresFull = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
target = 'Potability'

curve_learning_function(LinearRegression(), dataset, featuresFull, target, train_sizes, 15, 'full-featured', 0.5)

curve_learning_function(LinearSVR(loss='squared_epsilon_insensitive', random_state=1, max_iter=10000), dataset, featuresFull, target, train_sizes, 15, 'full-featured', 1.2)

features = ['ph', 'Hardness', 'Solids']

curve_learning_function(LinearRegression(), dataset, features, target, train_sizes, 15, 'less-featured', 0.5)