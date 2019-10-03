import pandas as pd
import numpy as np
import sklearn.preprocessing 

def unroll(data,sequence_length=24):
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    return np.asarray(result)

def historical(q_train, idx, unroll_length=48, history_length=7):
    assert(history_length*unroll_length < idx), "Invalid index, not enough historical data"
    result = []
    for i in range(history_length,0,-1):
        result.append(q_train[idx-unroll_length*i])
    return np.asarray(result)

def preprocessing(config, test_split_size=0.1, valid_split_size=0.1):
    ''' preprocess data according to configuration 
        (!) when wrapping into batches, for convenience, it would erase scrap data
    '''

    # load data
    try:
        dataset_path = "data/nyc_taxi.csv"
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print("nyc_taxi.csv doesn't exist")
        print("you can run $ANALYTICS_ZOO_HOME/bin/data/NAB/nyc_taxi/get_nyc_taxi.sh to download nyc_taxi.csv")

    # featurization
    df['datetime'] = pd.to_datetime(df['timestamp'])
    # add hour
    df['hours'] = df['datetime'].dt.hour
    df['awake'] = (((df['hours'] >= 6) & (df['hours'] <= 23)) | (df['hours'] == 0)).astype(int)
    # add categorical feature indicating awake/sleep (6:00-00:00)
    df['categories'] = df['awake']
    data_n = df[['value', 'hours', 'awake']]

    # scaling
    standard_scaler = sklearn.preprocessing.StandardScaler()
    np_scaled = standard_scaler.fit_transform(data_n)
    data_n = pd.DataFrame(np_scaled)

    #important parameters and train/test size
    prediction_time = 2 # predict one hour later
    unroll_length = config.T
    history_length = config.n
    # testdatacut = testdatasize + unroll_length  + 1

    #generate input data
    base_x = np.array(data_n)
    # generate q
    q = unroll(base_x,unroll_length)
    #generate Xi
    result = []
    start_cut = (history_length+1) * unroll_length + 1
    for i in range(start_cut,q.shape[0]):
        result.append(historical(q, i, unroll_length, history_length))
    X = np.asarray(result)
    q = q[start_cut:]
    # generate y
    y = q[:,-1,0].reshape(-1,1)
    X = X[:-prediction_time]
    q = q[:-prediction_time]
    y = y[prediction_time:] # predict one hour later 
    
    # data split 
    batch_size = config.batch_size
    total_length = q.shape[0] // batch_size
    valid_split = int( total_length * (1- valid_split_size - test_split_size) )
    test_split = int( total_length * (1 - test_split_size) )

    batch_data_train = [ (X[100*i:100*(i+1)], q[100*i:100*(i+1)], y[100*i:100*(i+1)]) for i in range(0, valid_split) ]
    batch_data_valid = [ (X[100*i:100*(i+1)], q[100*i:100*(i+1)], y[100*i:100*(i+1)]) for i in range(valid_split, test_split) ]
    batch_data_test = [ (X[100*i:100*(i+1)], q[100*i:100*(i+1)], y[100*i:100*(i+1)]) for i in range(test_split, total_length) ]

    return batch_data_train, batch_data_valid, batch_data_test

    

