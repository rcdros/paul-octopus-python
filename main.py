import csv

from predictor.OneZeroPredictor import OneZeroPredictor


csv_columns = ['home', 'home_score', 'away_score', 'away']


def read_csv():
    with open('sample_predictions_submission.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        matches_data = [row for row in reader]

    return matches_data


def write_csv(data):
    with open('predictions.csv', 'w', newline='') as fp:
        writer = csv.DictWriter(fp, delimiter=',', fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(data)


if __name__ == '__main__':

    # Argparse to select predictor, input and output (default values when not passed)

    # Abstract class to load data, sample implementation for csv load data
    # Abstract class to prediction, sample implementations for dummy predictors
    # Abstract class to output data, sample implementation for csv file (other examples: Save to GCS, Save to BQ...)

    matches = read_csv()
    predictor = OneZeroPredictor()
    predictions = predictor.predict(matches)
    print(predictions)
    write_csv(predictions)
