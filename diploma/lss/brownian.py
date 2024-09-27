import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class BrownianMotionSimulation(BaseModel):
    model_config = ConfigDict(frozen=True)

    S_0: float
    risk_free_rate: float
    volatility: float

    @staticmethod
    def brownian_motion(num_steps) -> NDArray:
        """
            W_i = W_{i-1} + N(0,1)
        :return W: Вектор броуновского движения
        """
        W = np.zeros(num_steps)  # W[0] всегда == 0

        for i in range(1, num_steps):
            W[i] = W[i - 1] + np.random.normal(
                0, 1
            )  # \xi_i \sim np.random.normal(0, 1)

        return W

    def _simulate_path(self, W_t) -> NDArray:
        """
            S_t = S_0 e^{ (r - \frac{\sigma^2}{2}) t  + \sigma \sqrt{t} W_t}
        :param W_t: Вектор броуновского движения.
        :return: Монте-Карло выборку для использования в дальнейшем
        """
        dt = 1 / 365
        t = np.arange(1, len(W_t) + 1) / 365

        volatility = self.volatility / 100
        risk_free_rate = self.risk_free_rate / 100

        path = self.S_0 * np.exp(
            (risk_free_rate - (volatility**2) / 2) * t
            + volatility * np.sqrt(dt) * W_t
        )
        return path

    def simulate(self, num_paths, len_paths) -> NDArray:
        """
            Генерация траекторий методом Монте-Карло. + Antithetic Variates

        :param num_paths: Количество траекторий (должно быть четным числом).
        :param len_paths: Длина каждой траектории.
        :return: Массив с траекториями Монте-Карло и их Antithetic Variates.
        """

        simulations = np.zeros((num_paths, len_paths))

        for i in range(0, num_paths // 2 * 2, 2):
            brownian_i = self.brownian_motion(len_paths)

            simulations[i] = self._simulate_path(brownian_i)
            simulations[i + 1] = self._simulate_path(-brownian_i)

        if num_paths % 2 != 0:
            brownian = self.brownian_motion(len_paths)
            simulations[-1] = self._simulate_path(brownian)

        return simulations
