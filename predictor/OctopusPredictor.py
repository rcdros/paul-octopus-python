from predictor.AbstractPredictor import AbstractPredictor
from utils.csv import read_csv


class OctopusPredictor(AbstractPredictor):
    def predict_match(self, home, away):
        historical_data = read_csv('historical_win-loose-draw_ratios.csv')

        home_points = 0
        away_points = 0
        for data in historical_data:
            if data['country1'] == home:
                home_points += float(data['wins'])
            if data['country2'] == away:
                away_points += float(data['looses'])

        return {'home': home, 'home_score': int(home_points), 'away_score': int(away_points), 'away': away}
