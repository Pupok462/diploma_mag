from dataclasses import dataclass
from hedge.option_schema import OptionSchema
import numpy as np
from scipy.stats import norm


@dataclass
class ImpliedVolatility:
    option: OptionSchema

    def black_scholes(self, sigma):
        r"""
            # Формула Блека-Шоулза

            .. math::
                EC = S_now * N(d1) - N(d2) * K * e^{-rT}

                EP = K * e^{-rT} * N(-d2) - S_now * N(-d1)

                     ln(S/K) + (r + \sigma^2)*T
                d1 = --------------------------
                          \sigma * \sqrt{T}

                d2 = d1 - \sigma * \sqrt{T}

                N(x) = \frac{1}{\sqrt{2*pi}} * \int^{x}_{-\infty} e^{-z^2/2} dz

        :param sigma:
        :return: EC/EP цену европейского колл/пут опциона при заданной волатильности
        """
        N = norm.cdf
        T = self.option.expiration_time / self.option.expiration_time_denominator

        d1 = (
            np.log(self.option.spot_price / self.option.strike)
            + (self.option.risk_free_rate + sigma**2 / 2) * T
        ) / (sigma * np.sqrt(T))

        d2 = d1 - sigma * np.sqrt(T)

        if self.option.optionType == "CALL":
            return self.option.spot_price * N(d1) - N(d2) * self.option.strike * np.exp(
                -self.option.risk_free_rate * T
            )
        elif self.option.optionType == "PUT":
            return self.option.strike * np.exp(-self.option.risk_free_rate * T) * N(
                -d2
            ) - self.option.spot_price * N(-d1)

    def count(self, tol=1e-4):
        left = 0
        right = 1

        diff_left = self.black_scholes(left) - self.option.option_price
        diff_right = self.black_scholes(right) - self.option.option_price

        while diff_left < 0 and diff_right < 0:
            right *= 2
            diff_right = self.black_scholes(right) - self.option.option_price

        middle = (left + right) / 2

        diff_middle = self.black_scholes(middle) - self.option.option_price

        while right - left > tol:
            if diff_middle < 0:
                left = middle
                middle = (left + right) / 2

                diff_middle = self.black_scholes(middle) - self.option.option_price

            elif diff_middle > 0:
                right = middle
                middle = (left + right) / 2

                diff_middle = self.black_scholes(middle) - self.option.option_price
        sigma = middle
        return sigma
