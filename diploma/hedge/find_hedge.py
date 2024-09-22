from hedge.option_schema import OptionSchema
from hedge.implied_volatility import ImpliedVolatility
from dataclasses import dataclass
from lss.brownian import BrownianMotionSimulation
from lss.options import PutOption
from lss.models import NaiveModel
from lss.algorithm import LongStaffSchwartz
import numpy as np
from scipy.stats import norm


@dataclass
class FindHedge:
    option: OptionSchema
    implied_volatility: ImpliedVolatility

    @staticmethod
    def _bs_put(option: OptionSchema, sigma: float, S: float):
        """
            EP = K * e^{-rT} * N(-d2) - S_now * N(-d1)
        :return:
        """
        N = norm.cdf
        T = option.expiration_time / option.expiration_time_denominator

        d1 = (np.log(S / option.strike) + (option.risk_free_rate + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        return option.strike * np.exp(-option.risk_free_rate * T) * N(-d2) - S * N(-d1)

    def simulate_paths(self, volatility: float, num_simulations):
        brownian = BrownianMotionSimulation(
            S_0=self.option.spot_price,
            risk_free_rate=self.option.risk_free_rate * 100,
            volatility=volatility * 100,
        )
        monte_carlo_simulations = brownian.simulate(
            num_paths=num_simulations, len_paths=int(self.option.expiration_time + 1)
        )
        return monte_carlo_simulations

    def _find_american_option_price(self, volatility: float, num_simulations: int, simulations=None):

        if simulations is None:
            simulations = self.simulate_paths(volatility, num_simulations)

        naive_model = NaiveModel(risk_free_rate=self.option.risk_free_rate*100)

        option_model = PutOption()

        lss_model = LongStaffSchwartz(
            risk_free_rate=self.option.risk_free_rate*100,
            option=option_model,
            strike=self.option.strike,
            monte_carlo_paths=simulations,
            model=naive_model,
        )
        _, lss_option_price = lss_model.evaluate(verbose=False)

        if len(list(lss_model.models_list)) == 0 or sum(list(lss_model.models_list)) == len(
                list(lss_model.models_list)):
            model = []
        else:
            model = list(lss_model.models_list)[0]
        return lss_option_price, model, simulations

    # @staticmethod
    # def _find_d_price(spot_price, model, eps=1e-8):
    #     """
    #                 regression(x+eps) - regression(x-eps)
    #         delta = -------------------------------------
    #                              2*eps
    #
    #     :param price: Спотовая цена сейчас (Шаг 0)
    #     :param model: Модель регрессии с шага 1 на шаг 2
    #     :return: delta
    #     """
    #     delta = (model(spot_price + eps) - model(spot_price - eps)) / (2 * eps)
    #
    #     return delta

    def _find_hedge(self, delta_prev, hedge_prev):
        """
            hedge_now = delta_prev * S_now + (hedge_prev - delta_prev * S_prev) * e^{r/365}

        :param delta_prev: Дельта на прошлом шаге
        :param hedge_prev: Хедж на прошлом шаге
        :return:
        """
        T = 1 / self.option.expiration_time_denominator
        exp_rT = np.exp(-self.option.risk_free_rate * T)
        S_now = self.option.spot_price
        S_prev = self.option.spot_price_prev

        hedge_now = delta_prev * S_now + (hedge_prev - delta_prev * S_prev) * exp_rT

        return hedge_now

    def run(self, step_num: int, hedge_prev: float = None, delta_prev: float = None, num_simulations: int = 10, eps=1e-4):
        """
            Функция поиска хеджа

            1) Оценивает $\sigma_{implied}$
            2) Находит цену американского опциона
            3) Находит производную по цене и хедж

           |-----------------------------------------------------------------------------------------------------------|
           |1 STEP                                                                                                     |
           |    STEP 1.0  option_price_0                                                                               |
           |    STEP 1.1  Оформляем портфель AP_price_0/hedge_0 = delta_0 * S_0 + (AP_price_0 - delta_0 * S_0)         |
           |    STEP 1.2  delta_0                                                                                      |
           |    STEP 1.3* Фиктивный шаг AP_price_0 vs hedge_0 тк hedge_0 = AP_price_0                                  |
           |    STEP 1.4* Так же фиктивный шаг hedge_0 vs (K - S_0) т.к. заведомо нельзя сравнивать опцион стоит больше|
           |-----------------------------------------------------------------------------------------------------------|
           |2 STEP                                                                                                     |
           |    STEP 2.0  option_price_1                                                                               |
           |    STEP 2.1  Балансируем портфель hedge_1 = delta_0 * S_1 + (hedge_0 - delta_0 * S_0) * e^{r/365}         |
           |    STEP 2.2  delta_1                                                                                      |
           |    STEP 2.3  Проверяем AP_price_1 vs hedge_1                                                              |
           |    STEP 2.4  Проверяем можем ли покрыть опцион if (K - S_1) > 0  -> hedge_1 vs (K - S_1)                  |
           |-----------------------------------------------------------------------------------------------------------|
           |3 STEP                                                                                                     |
           |    STEP 3.0  option_price_2                                                                               |
           |    STEP 3.1  Балансируем портфель hedge_2 = delta_1 * S_2 + (hedge_1 - delta_1 * S_1) * e^{r/365}         |
           |    STEP 3.2  delta_2                                                                                      |
           |    STEP 3.3  Проверяем AP_price_2 vs hedge_2                                                              |
           |    STEP 3.4  Проверяем можем ли покрыть опцион if (K - S_2) > 0  -> hedge_2 vs (K - S_2)                  |
           |-----------------------------------------------------------------------------------------------------------|
           | etc ...                                                                                                   |

        :return:
        hedge_now - Хедж сейчас
        delta_now - Дельта сейчас
        american_option_price - Цена опциона
        implied_volatility - Имплайд волатильность

        """
        if self.option.expiration_time == 0:
            hedge_now = self._find_hedge(delta_prev, hedge_prev)
            return hedge_now, None, None, None

        # STEP *.0
        implied_volatility = self.implied_volatility.count()
        american_option_price, models, lss_paths = self._find_american_option_price(
            implied_volatility, num_simulations
        )

        # STEP *.1
        if step_num == 0:
            hedge_now = american_option_price
        else:
            hedge_now = self._find_hedge(delta_prev, hedge_prev)

        # Delta count

        sim_b = lss_paths * (self.option.spot_price + eps) / self.option.spot_price
        sim_l = lss_paths * (self.option.spot_price - eps) / self.option.spot_price
        AP_price_b, _, _ = self._find_american_option_price(
            implied_volatility, num_simulations, sim_b
        )
        AP_price_l, _, _ = self._find_american_option_price(
            implied_volatility, num_simulations, sim_l
        )
        # find delta
        delta_now = (AP_price_b - AP_price_l) / (2 * eps)


        # if self.option.expiration_time <= 2:
        #     p_eps_p = self._bs_put(self.option, implied_volatility, self.option.spot_price + eps)
        #     p_eps_m = self._bs_put(self.option, implied_volatility, self.option.spot_price - eps)
        #
        #     delta_now = (p_eps_p - p_eps_m)/2*eps
        # else:
        #     delta_now = self._find_d_price(self.option.spot_price, models, eps)

        return hedge_now, delta_now, american_option_price, implied_volatility
