import pandas as pd
import numpy as np

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
    ''' preprocess data according to configuration '''

    # load data
    try:
        dataset_path = "nyc_taxi.csv"
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

    #select and standardize data
    data_n = df[['value', 'hours', 'awake']]
    standard_scaler = preprocessing.StandardScaler()
    np_scaled = standard_scaler.fit_transform(data_n)
    data_n = pd.DataFrame(np_scaled)

    #important parameters and train/test size
    prediction_time = 1 
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
    for i in range(start_cut,q_train.shape[0]):
        result.append(historical(q, i, unroll_length, history_length))
    X = np.asarray(result)
    q = q[start_cut:]
    # generate y_train
    y_train = q[:,-1,0].reshape(-1,1)

