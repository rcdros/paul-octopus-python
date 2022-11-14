import importlib

from flask import Flask, send_from_directory

from predictor.AbstractPredictor import AbstractPredictor
from utils.azure_storage import download_file_from_azure
from utils.csv import *

app = Flask(__name__)


@app.route("/")
def index():
    return "Paul the Octopus is alive!!!"


@app.route("/predict/<predictor_name>")
def predict(predictor_name):

    # Download all the files that you need for your predictors
    download_file_from_azure(container_name='files', blob_name='ranking.csv')
    download_file_from_azure(container_name='files', blob_name='historical-results.csv')
    download_file_from_azure(container_name='files', blob_name='matches-schedule.csv')


    # Instantiate the Predictor class based on the predictor_name
    PredictorClass = getattr(importlib.import_module(f'predictor.{predictor_name}'), predictor_name)
    predictor: AbstractPredictor = PredictorClass()

    # Make predictions and write them into the predictions.csv file
    predictions = predictor.predict()
    write_csv(predictions, ['home', 'home_score', 'away_score', 'away'], 'predictions.csv')

    # Return the predictions.csv file for downloading
    return send_from_directory('.', 'predictions.csv')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)


