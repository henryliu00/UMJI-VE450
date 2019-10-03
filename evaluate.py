from train import make_model_path
import tensorflow as tf
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn
import pandas as pd

def predict(config, model, batch_data_test, load_data=0):
    # load and predict
    y_pred_list = []
    y_real_list = []
    with tf.Session() as sess:
        if load_data:
            model.restore(make_model_path(config),sess)
        print("start prediction")
        for ds in batch_data_test:
            loss, pred = model.predict(ds, sess)
            y_pred_list.append(pred)
            y_real_list.append(ds[-1])
    
    # reshape
    y_pred = np.asarray(y_pred_list).reshape(-1)
    y_test = np.asarray(y_real_list).reshape(-1)
    return y_pred, y_test

def evaluate(config, model, batch_data_test, load_data=0):
    ''' evaluate data with  plot'''
    y_pred, y_test = predict(config, model, batch_data_test, load_data)

    # create the list of difference between prediction and test data
    diff= abs(y_pred - y_test)
    ratio= abs(y_pred/y_test)

    # plot the difference and the threshold (for the test data)
    # An estimation of anomly population of the dataset
    outliers_fraction = 0.01
    # select the most distant prediction/reality data points as anomalies
    
    diff = pd.Series(diff)
    number_of_outliers = int(outliers_fraction*len(diff))
    threshold = diff.nlargest(number_of_outliers).min()
    fig, ax = plt.subplots(2,1,figsize=(8,10))
    axs = ax[0]

    axs.plot(diff,color='blue', label='diff')
    axs.set_title('the difference between the predicted values and actual values with the threshold line')

    axs.hlines(threshold, 0, 1000, color='red', label='threshold')
    axs.set_xlabel('test data index')
    axs.set_ylabel('difference value after scaling')
    axs.legend(loc='upper left')
    # plot the predicted values and actual values with anomely detection (for the test data)
    axs = ax[1]
    a = pd.Series(y_pred)[diff > threshold]
    axs.plot(y_pred,color='red', label='predicted values')
    axs.plot(y_test,color='blue', label='actual values')
    axs.scatter(list(a.index),a.values, color='red', label='anomalies value')
    axs.set_title('the predicted values and actual values (for the test data)')

    axs.set_xlabel('test data index')
    axs.set_ylabel('number of taxi passengers after scaling')
    axs.legend(loc='upper left')
    plt.show()








