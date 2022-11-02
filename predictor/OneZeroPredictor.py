from predictor.AbstractPredictor import AbstractPredictor


class OneZeroPredictor(AbstractPredictor):
    def predict_match(self, home, away):
        return {'home': home, 'home_score': 1, 'away_score': 0, 'away': away}
