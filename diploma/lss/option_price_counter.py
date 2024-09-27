from pydantic import BaseModel, ConfigDict, Field
import numpy as np
from lss.option.call_option import CallOption
from lss.option.put_option import PutOption
from lss.option.meta_option import MetaOption
from numpy.typing import NDArray


class OptionPriceCounter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    option: MetaOption = Field(default=PutOption())

    risk_free_rate: float
    cash_flow_matrix: NDArray
    monte_carlo_paths: NDArray
    strike: float

    def _american_option_price(self) -> float:
        option_price = 0
        risk_free_rate = self.risk_free_rate / 100

        for col in range(self.cash_flow_matrix.shape[1] - 1):
            option_price += np.sum(self.cash_flow_matrix[:, col]) * np.exp(
                -risk_free_rate * col / 365
            )

        return option_price / self.cash_flow_matrix.shape[0]

    def _european_option_price(self) -> float:
        risk_free_rate = self.risk_free_rate / 100

        return (
            np.sum(
                self.option.path_cash_flow(self.monte_carlo_paths[:, -1], self.strike)
            )
            * np.exp(-risk_free_rate * (self.monte_carlo_paths.shape[1] - 1) / 365)
            / self.monte_carlo_paths.shape[0]
        )

    def count_option_prices(self):
        return self._american_option_price(), self._european_option_price()
