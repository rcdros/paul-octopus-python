from abc import ABC, abstractmethod


class AbstractPredictor(ABC):

    @abstractmethod
    def predict_match(self, home, away):
        print(f'predicting {home} x {away}')
        pass

    def predict(self, matches):
        predictions = []
        for match in matches:
            predictions.append(self.predict_match(match['home'], match['away']))

        return predictions
