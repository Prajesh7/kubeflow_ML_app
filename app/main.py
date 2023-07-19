from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/")
def root():
    return {"hello": "world"}
    
@app.get("/sail")
def get_dail():
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import optuna

    data = pd.read_csv("app/SAIL.NS.csv")
    data['year'] = pd.to_datetime(data['Date']).dt.year
    data['month'] = pd.to_datetime(data['Date']).dt.month
    data['day'] = pd.to_datetime(data['Date']).dt.day
    data = data.drop(['Date'], axis = 1)

    for c in data.columns:
        data[c].fillna(data[c].mean(), inplace=True)

    from sklearn.preprocessing import MinMaxScaler
    mms = MinMaxScaler(feature_range=(0,1))
    scaled_data = mms.fit_transform(data['Close'].values.reshape(-1,1))

    prediction_days = 60
    x_train = []
    y_train = []

    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i - prediction_days: i, 0])
        y_train.append(scaled_data[i, 0])
        
    xdata, ydata = np.array(x_train), np.array(y_train)

    train_size = 0.66
    val_size = 0.2
    test_size = 0.2

    xtrain, ytrain = xdata[0:int(xdata.shape[0]*train_size)], ydata[0:int(ydata.shape[0]*train_size)]
    xval, yval = xdata[xtrain.shape[0]+1:int(xtrain.shape[0] + (xdata.shape[0]-xtrain.shape[0])*0.5)], ydata[ytrain.shape[0]+1:int(ytrain.shape[0] + (ydata.shape[0]-ytrain.shape[0])*0.5)]
    xtest, ytest = xdata[int(xtrain.shape[0]+xval.shape[0]+1):], ydata[int(ytrain.shape[0]+yval.shape[0]+1):]


    from sklearn.ensemble import RandomForestRegressor
    rfr_final = RandomForestRegressor(n_estimators=10, max_depth=10)
    rfr_final.fit(xtrain, ytrain)
    rfr_optuna = RandomForestRegressor(n_estimators=11, max_depth=57)
    rfr_optuna.fit(xtrain, ytrain)

    # ytrain_pred = rfr_final.predict(xtrain)
    # plt.figure(figsize=(25, 10))
    # # plt.subplot(2,2,1)
    # plt.plot(mms.inverse_transform(ytrain[ytrain.shape[0]-100:].reshape(-1, 1)))
    # plt.plot(mms.inverse_transform(ytrain_pred[ytrain.shape[0]-100:].reshape(-1, 1)), c = 'red')
    # # plt.show()
    # # plt.subplot(2,2,1)

    # ytrain_pred = rfr_optuna.predict(xtrain)
    # plt.figure(figsize=(25, 10))
    # # plt.subplot(2,2,2)
    # plt.plot(mms.inverse_transform(ytrain[ytrain.shape[0]-100:].reshape(-1, 1)))
    # plt.plot(mms.inverse_transform(ytrain_pred[ytrain.shape[0]-100:].reshape(-1, 1)), c = 'red')
    # # plt.show()

    # ytest_pred = rfr_final.predict(xtest)
    # plt.figure(figsize=(25, 10))
    # # plt.subplot(2,2,3)
    # plt.plot(mms.inverse_transform(ytest[ytest.shape[0]-100:].reshape(-1, 1)))
    # plt.plot(mms.inverse_transform(ytest_pred[ytest.shape[0]-100:].reshape(-1, 1)), c = 'red')
    # # plt.show()

    ytest_pred = rfr_optuna.predict(xtest)
    plt.figure(figsize=(25, 10))
    # plt.subplot(2,2,4)
    plt.plot(mms.inverse_transform(ytest[ytest.shape[0]-100:].reshape(-1, 1)))
    plt.plot(mms.inverse_transform(ytest_pred[ytest.shape[0]-100:].reshape(-1, 1)), c = 'red')
    # plt.show()
    
    plt.savefig('sail.png')
    file = open('sail.png', mode='rb')
    
    return StreamingResponse(file, media_type="image/png")