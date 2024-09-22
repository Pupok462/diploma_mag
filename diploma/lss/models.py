from pydantic import BaseModel
from numpy.typing import NDArray
import numpy as np
from sklearn.svm import SVR


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
        """
        :param X: Цены акции когда опцион в деньгах в момент веремени t
        :param y: Дисконтированные будущие денежные потоки к моменту времени t с учетом матрицы остановок
        :return: Вектор continuation, regression formula
        """
        # TODO: ТУТ почему-то дисконтирвоание не правильное
        X = X

        risk_free_rate_vec = np.array(
            [
                (1 / np.exp(-self.risk_free_rate/100 * (i + 1)/365))
                for i in range(y.shape[1])
            ]
        ).reshape(y.shape[1], 1)

        y = y.dot(risk_free_rate_vec).squeeze()

        if y.shape == ():
            y = np.array([y])

        regression_model = np.poly1d(np.polyfit(X, y, 2))

        return regression_model(X), regression_model

#
# class CleverModel(MetaModel):
#
#     def predict(self, X: NDArray, y: NDArray):
#         X = X
#         risk_free_rate_vec = np.array(
#             [
#                 (1 / np.exp(- self.risk_free_rate/100 * (i + 1)/365))
#                 for i in range(y.shape[1])
#             ]
#         ).reshape(y.shape[1], 1)
#
#         y = y.dot(risk_free_rate_vec).squeeze()
#
#         if y.shape == ():
#             y = np.array([y])
#
#         regression_model = SVR(kernel="rbf")
#
#         regression_model.fit(X, y)
#
#         return regression_model.predict(X), regression_model