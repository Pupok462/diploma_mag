from lss.option.meta_option import MetaOption
import numpy as np
from numpy.typing import NDArray

class CallOption(MetaOption):
    def path_cash_flow(self, monte_carlo_path, strike) -> NDArray:
        return np.maximum(monte_carlo_path - strike, 0)

    def path_mask(self, monte_carlo_path, strike) -> NDArray:
        return monte_carlo_path > strike
