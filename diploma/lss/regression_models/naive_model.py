from lss.regression_models.meta_model import MetaModel
from numpy.typing import NDArray
import numpy as np


class NaiveModel(MetaModel):
    def predict(self, X: NDArray, y: NDArray):
        X = X

        risk_free_rate_vec = np.array(
            [
                np.exp(-self.risk_free_rate / 100 * (i + 1) / 365)
                for i in range(y.shape[1])
            ]
        ).reshape(y.shape[1], 1)

        y = y.dot(risk_free_rate_vec).squeeze()

        if y.shape == ():
            y = np.array([y])

        regression_model = np.poly1d(np.polyfit(X, y, 2))

        return regression_model(X), regression_model
