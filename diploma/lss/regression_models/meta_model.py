from pydantic import BaseModel

class MetaModel(BaseModel):
    risk_free_rate: float

    def predict(self, X, y):
        """
        :param X: Цены акции когда опцион в деньгах в момент времени t
        :param y: Дисконтированные будущие денежные потоки к моменту времени t с учетом матрицы остановок
        :return: Вектор continuation, regression formula
        """
        pass