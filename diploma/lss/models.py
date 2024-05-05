from pydantic import BaseModel
from numpy.typing import NDArray
import numpy as np


class MetaModel(BaseModel):
    risk_free_rate: float

    def predict(self, X, y):
        """
        :param X: Денежный поток сейчас (в деньгах)
        :param y: Матрица денежных поток на следующие шаги
        :return: (continuation, model) Вектор продолжения (ожидаемых выплат от продолжения) и модель
        """
        pass


class NaiveModel(MetaModel):
    def predict(self, X: NDArray, y: NDArray):
        X = X
        risk_free_rate_vec = np.array(
            [
                (1 / (1 + self.risk_free_rate / 100)) ** (i + 1)
                for i in range(y.shape[1])
            ]
        ).reshape(y.shape[1], 1)

        y = y.dot(risk_free_rate_vec).squeeze()

        regression_model = np.poly1d(np.polyfit(X, y, 2))

        return regression_model(X), regression_model
