from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import base64
from io import BytesIO

from mlFlowProject.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

@app.route('/', methods=['GET'])
def homepage():
    return render_template("index.html")

@app.route('/train', methods = ['GET'])
def train():
    os.system("python main.py")
    return "Training Successful!"

@app.route('/predict', methods=['POST'])  # route to make predictions and show the result
def predict():
    if request.method == 'POST':
        try:
            previous_open = float(request.form['previous_open'])

            obj = PredictionPipeline()
            next_close_prediction = obj.predict(np.array(previous_open).reshape(1, -1))
            test_data = pd.read_csv(Path("artifacts/data_transformation/test.csv"))
            plt.figure(figsize = (20, 10))
            plt.plot(test_data['Close'][len(test_data)-100:])
            plt.xlabel('Time')
            plt.ylabel('Close Price')
            plt.title('Close Price over Time')
            plt.savefig('plot.png')
            plt.close()
            # return render_template('results.html', prediction=next_close_prediction)

            with open('plot.png', 'rb') as f:
                encoded_plot = base64.b64encode(f.read()).decode('utf-8')

            return render_template('results.html', prediction=next_close_prediction, plot=encoded_plot)

        except Exception as e:
            print('The Exception message is:', e)
            return 'Something went wrong'

    else:
        return render_template('index.html')
    
if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 80, debug = True)