from lss.option.meta_option import MetaOption
import numpy as np
from numpy.typing import NDArray

class PutOption(MetaOption):
    def path_cash_flow(self, monte_carlo_path, strike) -> NDArray:
        """
            (K - S_i)^+
        :param monte_carlo_path: 1d array
        :param strike: float
        :return:
        """
        assert np.sum(monte_carlo_path > 0) == monte_carlo_path.shape[0], "Существует значение в Монте-Карло пути <= 0"

        return np.maximum(strike - monte_carlo_path, 0)

    def path_mask(self, monte_carlo_path, strike) -> NDArray:
        return monte_carlo_path < strike