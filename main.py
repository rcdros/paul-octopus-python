import importlib

from predictor.AbstractPredictor import AbstractPredictor

from flask import Flask, send_from_directory

from utils.azure_storage import download_file_from_azure
from utils.csv import *

app = Flask(__name__)


@app.route("/")
def index():
    return "Paul the Octopus is alive!!!"


@app.route("/predict/<predictor_name>")
def predict(predictor_name):

    # Download all the files that you need for your predictors
    download_file_from_azure(container_name='files', blob_name='sample_predictions_submission.csv')
    download_file_from_azure(container_name='files', blob_name='historical_win-loose-draw_ratios.csv')

    matches = read_csv('sample_predictions_submission.csv')

    PredictorClass = getattr(importlib.import_module(f'predictor.{predictor_name}'), predictor_name)
    predictor: AbstractPredictor = PredictorClass()

    predictions = predictor.predict(matches)
    print(predictions)
    write_csv(predictions, ['home', 'home_score', 'away_score', 'away'], 'predictions.csv')

    return send_from_directory('.', 'predictions.csv')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)


