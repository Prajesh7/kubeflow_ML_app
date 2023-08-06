from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd

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

            return render_template('results.html', prediction=next_close_prediction)

        except Exception as e:
            print('The Exception message is:', e)
            return 'Something went wrong'

    else:
        return render_template('index.html')
    
if __name__ == '__main__':
    app.run(host = "0.0.0.0", port = 80, debug = True)