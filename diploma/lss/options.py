from numpy.typing import NDArray
import numpy as np
from pydantic import BaseModel


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


class CallOption(MetaOption):
    def path_cash_flow(self, monte_carlo_path, strike) -> NDArray:
        return np.maximum(monte_carlo_path - strike, 0)

    def path_mask(self, monte_carlo_path, strike) -> NDArray:
        return monte_carlo_path > strike


class PutOption(MetaOption):
    def path_cash_flow(self, monte_carlo_path, strike) -> NDArray:
        return np.where(
            monte_carlo_path == 0, 0, np.maximum(strike - monte_carlo_path, 0)
        )

    def path_mask(self, monte_carlo_path, strike) -> NDArray:
        return monte_carlo_path < strike
