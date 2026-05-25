from sklearn.metrics import accuracy_score

from stage_4_code.evaluate import evaluate


class ClassificationEvaluator(evaluate):

    def __init__(self):

        super().__init__(
            "AccuracyEvaluator",
            "Evaluate classification accuracy"
        )

    def evaluate(self, y_true, y_pred):

        return accuracy_score(y_true, y_pred)