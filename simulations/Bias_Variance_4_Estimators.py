# https://github.com/sayanam/MachineLearning/blob/master/ExperimentationWithBiasAndVariance/BiasAndVariance_V2.ipynb

import sys
import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

population_data= pd.read_csv("CASP.csv")

test_data = population_data.sample(300, replace=False, random_state=100)
train_data = population_data.drop(test_data.index, axis=0).reindex()
test_data = test_data.reindex()

X_train = population_data.drop(['F9'], axis=1)
y_train = population_data['F9']
X_test = test_data.drop(['F9'], axis=1)
y_test = test_data['F9']

#Linear Regression
population_linear_model = LinearRegression().fit(X_train, y_train)
y_hat_pop = population_linear_model.predict(X_test)

#Decision Tree
population_decision_tree = DecisionTreeRegressor(criterion='mse', min_samples_leaf=3).fit(X_train, y_train)
y_hat_pop_tree = population_decision_tree.predict(X_test)

#Bagging Decision Tree
population_bagging_model = BaggingRegressor().fit(X_train, y_train)
y_hat_pop_bagging = population_bagging_model.predict(X_test)

#Random Forest
population_randomForest_model = RandomForestRegressor().fit(X_train, y_train)
y_hat_pop_randomForest = population_randomForest_model.predict(X_test)

def calculate_varaince_of_model(samplePredictions, y_test):
    predictions_mean_model = samplePredictions.mean(axis =1)
    colNames = samplePredictions.columns
    variance = np.zeros(len(colNames))
    i = 0
    for colName in colNames:
        variance[i] = np.mean(np.square(samplePredictions[colName] - predictions_mean_model))
        rmse = mean_squared_error(y_test, samplePredictions[colName])
        i += 1
    return round(np.mean(variance),3), round(np.mean(rmse),3)

def calculate_bias_of_model(samplePredictions, y_hat_pop):
    return np.square((np.abs(samplePredictions.mean(axis=1) -y_hat_pop).mean()))

def samplePredForLinearRegression(X_train, y_train):
    sample_Linear_Model = LinearRegression().fit(X_train, y_train)
    return sample_Linear_Model.predict(X_test)

def samplePredForDecisionTree(X_train, y_train):
    sample_Tree_Model = DecisionTreeRegressor(criterion='mse', min_samples_leaf=3).fit(X_train, y_train)
    return sample_Tree_Model.predict(X_test)
    
def samplePredForDTBaggin(X_train, y_train):
    sample_bagging_Model = BaggingRegressor().fit(X_train, y_train)
    return sample_bagging_Model.predict(X_test)

def samplePredForRandomForest(X_train, y_train):
    sample_RandomForest_Model = RandomForestRegressor().fit(X_train, y_train)
    return sample_RandomForest_Model.predict(X_test)

def samplePredForRidgeRegression(X_train, y_train, alpha):
    sample_ridge_model = Ridge(alpha=alpha, normalize=True)
    sample_ridge_model.fit(X_train,y_train)
    return sample_ridge_model.predict(X_test)

def get_bias_variance(sampleCount, noOfModels):
    bias_variance_result = pd.DataFrame(columns=['sample_count','no_of_models','algorithm','bias','variance', 'mse'])
    print('Builds Linear Regression, Decision Tree, Bagging, Random Forest algorithms')
    print('Total No of models built is',str(noOfModels* 4))
    
    samplePredictionsLinearModel = pd.DataFrame()
    samplePredictionsTree = pd.DataFrame()
    samplePredictionsBagging = pd.DataFrame()
    samplePredictionsRandomForest = pd.DataFrame()
    
    with tqdm.tqdm(total=noOfModels, file=sys.stdout) as pbar:
        for i in range(0, noOfModels):
            pbar.set_description('Building Model : %d' % (1 + i))
            sample = train_data.sample(sampleCount,replace=False)
            X_train = sample.drop(['F9'], axis=1)
            y_train = sample['F9']

            samplePredictionsLinearModel['sample'+str(i+1)] = samplePredForLinearRegression(X_train, y_train)
            samplePredictionsTree['sample'+str(i+1)] = samplePredForDecisionTree(X_train, y_train)
            samplePredictionsBagging['sample'+str(i+1)] = samplePredForDTBaggin(X_train, y_train)
            samplePredictionsRandomForest['sample'+str(i+1)] = samplePredForRandomForest(X_train, y_train)
            pbar.update(1)
    
    var_mse_linear_model = calculate_varaince_of_model(samplePredictionsLinearModel, y_test)
    var_mse_tree_model = calculate_varaince_of_model(samplePredictionsTree, y_test)
    var_mse_bagging_model = calculate_varaince_of_model(samplePredictionsBagging, y_test)
    var_mse_random_forest_model = calculate_varaince_of_model(samplePredictionsRandomForest, y_test)
    
    bias_linear_model = calculate_bias_of_model(samplePredictionsLinearModel, y_hat_pop)
    bias_tree_model = calculate_bias_of_model(samplePredictionsTree, y_hat_pop_tree)
    bias_bagging_model = calculate_bias_of_model(samplePredictionsBagging, y_hat_pop_bagging)
    bias_random_forest_model = calculate_bias_of_model(samplePredictionsRandomForest, y_hat_pop_randomForest)
    
    s_linear = pd.Series(data={'sample_count':sampleCount, 'no_of_models':noOfModels, 'algorithm':'LR',
                    'bias':bias_linear_model, 'variance' : var_mse_linear_model[0], 'mse' : var_mse_linear_model[1]}, name = 0)
    
    s_tree = pd.Series(data={'sample_count':sampleCount, 'no_of_models':noOfModels, 'algorithm':'DecisionTree',
                    'bias':bias_tree_model, 'variance' : var_mse_tree_model[0], 'mse' : var_mse_tree_model[1]}, name = 0)
    
    s_bagging = pd.Series(data={'sample_count':sampleCount, 'no_of_models':noOfModels, 'algorithm':'Bagging',
                    'bias':bias_bagging_model, 'variance' : var_mse_bagging_model[0], 'mse' : var_mse_bagging_model[1]}, name = 0)
    
    s_rf = pd.Series(data={'sample_count':sampleCount, 'no_of_models':noOfModels, 'algorithm':'RandomForest',
                    'bias':bias_random_forest_model, 'variance' : var_mse_random_forest_model[0], 'mse' : var_mse_random_forest_model[1]}, name = 0)
    
    bias_variance_result = bias_variance_result.append(s_linear)
    bias_variance_result = bias_variance_result.append(s_tree)
    bias_variance_result = bias_variance_result.append(s_bagging)
    bias_variance_result = bias_variance_result.append(s_rf)
    
    bias_variance_result.reset_index(inplace=True)
    bias_variance_result.drop(['index'], axis=1, inplace=True)
    
    return bias_variance_result

def get_bias_variance_ridge_regression(sampleCount, noOfModels):
    bias_variance_result = pd.DataFrame(columns=['alpha','bias','variance', 'mse'])
    print('Ridge Regression')
    samplePredictionsRidge = pd.DataFrame()
    alphas = [0.01, 0.05,0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.40]
    
    for alpha in alphas:
        for i in range(0, noOfModels):
            sample = train_data.sample(sampleCount,replace=False)
            X_train = sample.drop(['F9'], axis=1)
            y_train = sample['F9']

            samplePredictionsRidge['sample'+str(i+1)] = samplePredForRidgeRegression(X_train, y_train, alpha)
        
        var_mse_ridge_model = calculate_varaince_of_model(samplePredictionsRidge, y_test)
        bias_ridge_model = calculate_bias_of_model(samplePredictionsRidge, y_hat_pop)
    
        s_linear = pd.Series(data={'alpha' : alpha, 'bias':bias_ridge_model, 
                               'variance' : var_mse_ridge_model[0], 'mse' : var_mse_ridge_model[1]}, name =0)
        
        bias_variance_result = bias_variance_result.append(s_linear)
    
    fig, ax = plt.subplots(1,2, figsize = (14,6))
    
    ax[0].plot(bias_variance_result.alpha, bias_variance_result.bias)
    ax[0].plot(bias_variance_result.alpha, bias_variance_result.variance)
    ax[0].legend(['bias','variance'])
    ax[0].set_xlabel('alpha')
    
    ax[1].plot(bias_variance_result.alpha, bias_variance_result.mse)
    ax[1].legend(['mse'])
    ax[1].set_xlabel('alpha')
    plt.show()
    
    bias_variance_result.reset_index(inplace=True)
    bias_variance_result.drop(['index'], axis=1, inplace=True)
    
    return bias_variance_result

noOfSamples = [100,500,1000,2000,4000,8000,10000]
#noOfSamples = [100,300,500,700,900,1000,1200]
noOfModels = 50
lenNoOfSamples = len(noOfSamples)
bias_variance_result = pd.DataFrame()
print('Building Model for samples ', noOfSamples)
for i in range(0, lenNoOfSamples):
    noOfSample = noOfSamples[i]
    print('Building models with sample size : ', noOfSample)
    bias_variance_result = bias_variance_result.append(get_bias_variance(noOfSample,noOfModels))
bias_variance_result.reset_index(inplace=True)
bias_variance_result.drop(['index'], axis=1, inplace=True)

# Plotting the obtained results
fig = plt.figure(figsize=(14,8))
layout = (2, 2)

ax0 = plt.subplot2grid(layout, (0, 0))
ax0.title.set_text('Bias')

temp = bias_variance_result[bias_variance_result['algorithm'] == 'LR']
ax0.plot(temp.sample_count, temp.bias)

temp = bias_variance_result[bias_variance_result['algorithm'] == 'DecisionTree']
ax0.plot(temp.sample_count, temp.bias)

temp = bias_variance_result[bias_variance_result['algorithm'] == 'Bagging']
ax0.plot(temp.sample_count, temp.bias)

temp = bias_variance_result[bias_variance_result['algorithm'] == 'RandomForest']
ax0.plot(temp.sample_count, temp.bias)

ax0.legend(['LR','DecisionTree','Bagging','RandomForest'], loc=1)
ax0.set_xlabel('Sample Size')

ax1 = plt.subplot2grid(layout, (0, 1))
ax1.title.set_text('Variance')

temp = bias_variance_result[bias_variance_result['algorithm'] == 'LR']
ax1.plot(temp.sample_count, temp.variance)

temp = bias_variance_result[bias_variance_result['algorithm'] == 'DecisionTree']
ax1.plot(temp.sample_count, temp.variance)

temp = bias_variance_result[bias_variance_result['algorithm'] == 'Bagging']
ax1.plot(temp.sample_count, temp.variance)

temp = bias_variance_result[bias_variance_result['algorithm'] == 'RandomForest']
ax1.plot(temp.sample_count, temp.variance)

ax1.legend(['LR','DecisionTree','Bagging','RandomForest'], loc=1)
ax1.set_xlabel('Sample Size')


ax2 = plt.subplot2grid(layout, (1, 0), colspan=2)
ax2.title.set_text('MSE')

temp = bias_variance_result[bias_variance_result['algorithm'] == 'LR']
ax2.plot(temp.sample_count, temp.mse)

temp = bias_variance_result[bias_variance_result['algorithm'] == 'DecisionTree']
ax2.plot(temp.sample_count, temp.mse)

temp = bias_variance_result[bias_variance_result['algorithm'] == 'Bagging']
ax2.plot(temp.sample_count, temp.mse)

temp = bias_variance_result[bias_variance_result['algorithm'] == 'RandomForest']
ax2.plot(temp.sample_count, temp.mse)

ax2.legend(['LR','DecisionTree','Bagging','RandomForest'], loc=1)
ax2.set_xlabel('Sample Size')


plt.tight_layout()

df = bias_variance_result[bias_variance_result['sample_count'] == 8000][['algorithm','bias','variance','mse']]

t = pd.DataFrame(columns=['algorithm', 'measure', 'value'])
for measure in ['bias','variance','mse']:
    values = df[measure].values
    algos = df['algorithm'].values
    for i in range (0,len(values)):
        t = t.append(pd.Series(data={'algorithm':algos[i],'measure':measure, 'value' : values[i]},name =0))

plt.figure(figsize=(10,5))
sns.barplot(x='algorithm', y='value', hue='measure', data=t)

t = get_bias_variance_ridge_regression(1000, 30)

t = get_bias_variance_ridge_regression(300, 15)
