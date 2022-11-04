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

    download_file_from_azure(container_name='files', blob_name='sample_predictions_submission.csv')

    matches = read_csv('sample_predictions_submission.csv')

    PredictorClass = getattr(importlib.import_module(f'predictor.{predictor_name}'), predictor_name)
    predictor: AbstractPredictor = PredictorClass()

    predictions = predictor.predict(matches)
    print(predictions)
    write_csv(predictions, ['home', 'home_score', 'away_score', 'away'], 'predictions.csv')

    return send_from_directory('.', 'predictions.csv')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)

    # Argparse to select predictor, input and output (default values when not passed)

    # Abstract class to load data, sample implementation for csv load data
    # Abstract class to prediction, sample implementations for dummy predictors
    # Abstract class to output data, sample implementation for csv file (other examples: Save to GCS, Save to BQ...)

