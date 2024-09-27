import numpy as np
from typing import Union, List, NoReturn
from pydantic import BaseModel, ConfigDict, Field
from numpy.typing import NDArray
from lss.option.call_option import CallOption
from lss.option.put_option import PutOption
from lss.option.meta_option import MetaOption
from lss.option_price_counter import OptionPriceCounter


class OutOfSample(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    risk_free_rate: float
    strike: float
    option: MetaOption = Field(default=PutOption())
    models_list: List = Field(default=[])
    monte_carlo_paths: NDArray
    cash_flow_matrix: NDArray = Field(default=np.array([]))

    american_option_price: float = Field(None)
    european_option_price: float = Field(None)

    @staticmethod
    def optimal_early_exercise_decision(
        exercise: np.array, continuation: np.array
    ) -> Union[np.array, np.array]:
        """
        :param exercise:
        :param continuation:
        :return: Возвращает элементов который потребуется для матрицы денежных потоков
        """
        return (
            np.where(exercise > continuation)[0],
            np.where(continuation > exercise)[0],
        )


    def evaluate(self, verbose: bool = False) -> NoReturn:
        # Инициализируем пути Монте-Карло и матрицу остановок
        out_of_sample_paths = self.monte_carlo_paths.copy()

        escape_ind = np.array([], dtype=int)

        stopping_matrix = np.zeros(
            (self.monte_carlo_paths.shape[0], self.monte_carlo_paths.shape[1] - 1)
        )

        for i in range(1, out_of_sample_paths.shape[1] - 1):
            # Берем модель с обучения
            model_i = self.models_list[i - 1]

            oos_path = out_of_sample_paths[:, i]
            oos_path[escape_ind] = 0

            # Если путь в деньгах
            in_money_mask = np.where(oos_path == 0, False, oos_path < self.strike)

            in_money_indices = np.where(in_money_mask)[0]

            exerices = self.option.path_cash_flow(
                oos_path[in_money_indices], self.strike
            )

            continuation = model_i(oos_path[in_money_indices])
            index_ex, index_cont = self.optimal_early_exercise_decision(
                exerices, continuation
            )

            index_ex, index_cont = (
                in_money_indices[index_ex],
                in_money_indices[index_cont],
            )

            escape_ind = np.concatenate((escape_ind, index_ex))

            stopping_matrix[index_ex, i - 1] = 1

        # Последний шаг OOS алгоритма
        last_oos_path = out_of_sample_paths[:, self.monte_carlo_paths.shape[1] - 1]
        last_oos_path[escape_ind] = 0

        in_money_mask = np.where(last_oos_path == 0, False, last_oos_path < self.strike)
        in_money_indeces = np.where(in_money_mask)[0]
        stopping_matrix[in_money_indeces, -1] = 1

        self.cash_flow_matrix = (
            np.maximum(self.strike - out_of_sample_paths[:, 1:], 0) * stopping_matrix
        )

        # -------------------------------------------------------------------------------
        # ------------- Считаем цены американского и европейских опционов ---------------
        self.american_option_price, self.european_option_price = OptionPriceCounter(
            risk_free_rate=self.risk_free_rate,
            cash_flow_matrix=self.cash_flow_matrix.copy(),
            monte_carlo_paths=self.monte_carlo_paths.copy(),
            strike=self.strike
        ).count_option_prices()
        # -------------------------------------------------------------------------------