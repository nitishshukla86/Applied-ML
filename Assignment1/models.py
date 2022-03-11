from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.svm import SVC


def lstm(input_shape):
    model = Sequential()
    model.add(LSTM(50,  return_sequences=True,input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def support_vector_clf(X_train,y_train):
    return SVC().fit(X_train, y_train)


