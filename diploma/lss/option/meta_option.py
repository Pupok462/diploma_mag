from pydantic import BaseModel
from numpy.typing import NDArray


class MetaOption(BaseModel):
    def path_cash_flow(self, monte_carlo_path, strike) -> NDArray:
        """

        :param monte_carlo_path: сгенерированный путь Монте-Карло траектории
        :param strike: Strike price
        :return: Возвращает денежный поток в шаг i в зависимости от типа опциона
        """
        pass

    def path_mask(self, monte_carlo_path, strike) -> NDArray:
        """

        :param monte_carlo_path:
        :param strike: Strike price
        :return:
        """
        pass
