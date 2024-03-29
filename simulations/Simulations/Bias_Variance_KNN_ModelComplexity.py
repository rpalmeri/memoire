import sys
import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor

#Load the data and prepare train & test data
population_data= pd.read_csv("CASP.csv")

test_data = population_data.sample(1000, replace=False, random_state=100)
train_data = population_data.drop(test_data.index, axis=0).reindex()
test_data = test_data.reindex()

X_train = population_data.drop(['F9'], axis=1)
y_train = population_data['F9']
X_test = test_data.drop(['F9'], axis=1)
y_test = test_data['F9']

def calculate_variance_of_model(samplePredictions, y_test):
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

def samplePredForEstimator(n_neigh, X_train, y_train):
    sample_Model = KNeighborsRegressor(n_neighbors=n_neigh).fit(X_train, y_train)
    return sample_Model.predict(X_test)

def get_bias_variance(sampleCount, noOfModels, estimator_name, n_neigh):
    bias_variance_result = pd.DataFrame(columns=['neigh','no_of_models','algorithm','bias','variance', 'mse'])
    print('Builds ', estimator_name)
    print('Total No of models built is',str(noOfModels))
    
    samplePredictionsModel = pd.DataFrame()
    
    with tqdm.tqdm(total=noOfModels, file=sys.stdout) as pbar:
        for i in range(0, noOfModels):
            pbar.set_description('Building Model : %d' % (1 + i))
            sample = train_data.sample(sampleCount,replace=False)
            X_train = sample.drop(['F9'], axis=1)
            y_train = sample['F9']

            samplePredictionsModel['sample'+str(i+1)] = samplePredForEstimator(n_neigh, X_train, y_train)
            pbar.update(1)
    
    var_mse_linear_model = calculate_variance_of_model(samplePredictionsModel, y_test)
    
    bias_linear_model = calculate_bias_of_model(samplePredictionsModel, y_test)
    
    s_linear = pd.Series(data={'neigh':n_neigh, 'no_of_models':noOfModels, 'algorithm':'KNN',
                    'bias':bias_linear_model, 'variance' : var_mse_linear_model[0], 'mse' : var_mse_linear_model[1]}, name = 0)
    
    bias_variance_result = bias_variance_result.append(s_linear)
    
    bias_variance_result.reset_index(inplace=True)
    bias_variance_result.drop(['index'], axis=1, inplace=True)
    
    return bias_variance_result

def bias_variance_mse_graphics(noOfSamples, noOfModels, estimator_name, noOfNeighbors) :
    lenNoOfNeighbors = len(noOfNeighbors)
    bias_variance_result = pd.DataFrame()
    print('Building Model for samples ', noOfSamples)
    for i in range(0, lenNoOfNeighbors):
        noOfNeighbor = noOfNeighbors[i]
        print('Building models with k neighbors : ', noOfNeighbor)
        bias_variance_result = bias_variance_result.append(get_bias_variance(40000,noOfModels, estimator_name, noOfNeighbor))
    bias_variance_result.reset_index(inplace=True)
    bias_variance_result.drop(['index'], axis=1, inplace=True)
    print(bias_variance_result)

    # Plotting the obtained results
    fig, ax = plt.subplots(1,2, figsize = (14,6))
    ax[0].plot(bias_variance_result.neigh, bias_variance_result.bias)
    ax[0].plot(bias_variance_result.neigh, bias_variance_result.variance)
    ax[0].legend(['bias','variance'])
    ax[0].set_xlabel('k')
        
    ax[1].plot(bias_variance_result.neigh, bias_variance_result.mse)
    ax[1].legend(['mse'])
    ax[1].set_xlabel('k')
    plt.show()

noOfNeighbors = [1,5,10,15,20,25,50,75,100,150,200,250,500,750,1000,2000]
bias_variance_mse_graphics(40000, 1000, 'KNN', noOfNeighbors)