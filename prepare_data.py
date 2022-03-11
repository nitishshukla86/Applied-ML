import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def get_data(name='AAPL',start='2020-01-01', end='2021-06-12'):
    aapl= yf.Ticker("aapl")
    df = yf.download('AAPL', 
                        start=start, 
                        end=end, 
                        progress=False,
    )
    df['Date'] = pd.to_datetime(df.index,format='%Y-%m-%d')
    df.index = df['Date']
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close','Open'])
    for i in range(0,len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]
        new_data['Open'][i] = data['Open'][i]
    
    new_data.index = new_data.Date
    new_data.drop('Date', axis=1, inplace=True)
    dataset = new_data.values
    n=int(0.9*len(dataset))
    train = dataset[0:n,:]
    valid = dataset[n:,:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train_open, y_train_open,x_train_close, y_train_close = [], [],[],[]
    for i in range(60,len(train)):
        x_train_open.append(scaled_data[i-60:i,1])
        y_train_open.append(scaled_data[i,1])
        x_train_close.append(scaled_data[i-60:i,0])
        y_train_close.append(scaled_data[i,0])
    x_train_open, y_train_open,x_train_close, y_train_close = np.array(x_train_open),\
        np.array(y_train_open),np.array(x_train_close),np.array(y_train_close)

    x_train_open = np.reshape(x_train_open, (x_train_open.shape[0],x_train_open.shape[1],1))
    x_train_close = np.reshape(x_train_close, (x_train_close.shape[0],x_train_close.shape[1],1))
    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs  = scaler.transform(inputs)
    X_test_open,X_test_close = [],[]
    for i in range(60,inputs.shape[0]):
        X_test_close.append(inputs[i-60:i,0])
        X_test_open.append(inputs[i-60:i,1])
    X_test_open,X_test_close = np.array(X_test_open),np.array(X_test_close)

    X_test_open = np.reshape(X_test_open, (X_test_open.shape[0],X_test_open.shape[1],1))
    X_test_close = np.reshape(X_test_close, (X_test_close.shape[0],X_test_close.shape[1],1))  
    
    return n,new_data,scaler,x_train_open,x_train_close,y_train_open,y_train_close,X_test_open,X_test_close
        

def get_data_SVM(name='AAPL',start='2020-01-01', end='2021-06-12'):
    aapl= yf.Ticker("aapl")
    df = yf.download('AAPL', 
                        start=start, 
                        end=end, 
                        progress=False,
    )
    aapl= yf.Ticker("aapl")
    df = yf.download('AAPL', 
                            start='2020-01-01', 
                            end='2021-06-12', 
                            progress=False,
        )
    df['Date'] = pd.to_datetime(df.index,format='%Y-%m-%d')
    df.index = df['Date']
    df = df.drop(['Date'], axis='columns')
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low

    # Store all predictor variables in a variable X
    X = df[['Open-Close', 'High-Low']]
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    split_percentage = 0.8
    split = int(split_percentage*len(df))

    # Train data set
    X_train = X[:split]
    y_train = y[:split]

    # Test data set
    X_test = X[split:]
    y_test = y[split:]
    return df,X_train,X_test,y_train,y_test


    
    
    