import numpy as np
from typing import Union, List, NoReturn
from pydantic import BaseModel, ConfigDict, Field
from numpy.typing import NDArray
from lss.regression_models.meta_model import MetaModel
from lss.option.call_option import CallOption
from lss.option.put_option import PutOption
from lss.option.meta_option import MetaOption
from lss.option_price_counter import OptionPriceCounter


class LongStaffSchwartz(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    risk_free_rate: float = Field(gt=0)
    strike: float = Field(gt=0)
    option: MetaOption = Field(PutOption())
    model: MetaModel
    monte_carlo_paths: NDArray

    cash_flow_matrix: NDArray = Field(default=np.array([]))
    models_list: List = Field(default=[])

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

    @property
    def decision_matrix(self):
        """
        :return: Матрица 1 и 0 обозначающая остановки и продолжения
        """
        assert (
            self.cash_flow_matrix.size != 0
        ), "Матрица денежных потоков пустая, запустите сначала метод evolute()"
        return np.where(self.cash_flow_matrix > 0, 1, 0)

    @staticmethod
    def update_cash_flow_matrix(
        cash_flow_matrix: NDArray,
        exercise,
        exercise_ind,
        continuation_ind,
        in_money_indices,
        step_num,
    ) -> NDArray:
        """

        :param continuation_ind:
        :param cash_flow_matrix:
        :param exercise:
        :param exercise_ind:
        :param in_money_indices:
        :param step_num:
        :return: Обновляем матрицу денежных потоков исходя из матрицы продолжения или исполнения (обновляются N столбец и N-1)
        """
        # Обновляем значения под Exercise
        cash_flow_matrix[in_money_indices[exercise_ind], step_num - 1] = exercise[exercise_ind]
        cash_flow_matrix[in_money_indices[exercise_ind], step_num:] = 0

        # Обновляем значения под continuation
        cash_flow_matrix[in_money_indices[continuation_ind], step_num - 1] = 0

        return cash_flow_matrix

    def evaluate(self, verbose: bool = False) -> NoReturn:
        """
            Запуск алгоритма LongStaffSchwartz

        :param verbose:
        """

        # Инициализация матрицы денежных потоков (полностью нули)
        # Столбцов на 1 меньше чем в МК симуляциях, так как мы не учитываем S_0 цену
        cash_flow_matrix = np.zeros(
            (self.monte_carlo_paths.shape[0], self.monte_carlo_paths.shape[1] - 1)
        )

        if verbose:
            print("----- INITIAL CASH FLOW MATRIX -----")
            print(cash_flow_matrix)
            print("------------------------------------")

        # Запишем денежные потоки на последнем шаге в последний столбец матрицы cash_flow_matrix
        cash_flow_matrix[:, -1] = self.option.path_cash_flow(
            self.monte_carlo_paths[:, -1].copy(), self.strike
        )

        if verbose:
            print(f"----- CASH FLOW MATRIX AT TIME {cash_flow_matrix.shape[1]}-----")
            print(cash_flow_matrix)
            print("------------------------------------")

        # ---------------------------------------------------------------------------------
        # --------------------------------- MAIN BODY -------------------------------------
        # ---------------------------------------------------------------------------------

        # Итерируемся в обратном порядке для подсчета цены продолжения и исполнения
        for i in range(cash_flow_matrix.shape[1] - 1, 0, -1):

            # Берем часть матрицы денежных потоков на i и больше
            cash_flow_i = cash_flow_matrix[:, i:]
            # И берем цены базового актива на шаге i
            monte_carlo_i = self.monte_carlo_paths[:, i]

            # Записываем индексы путей которые в деньгах в зависимости от типа опциона
            in_money_mask = self.option.path_mask(
                monte_carlo_i, self.strike
            )
            in_money_indices = np.where(in_money_mask)[0]

            # Вычисляем вектор продолжения регрессируя и вектор исполнения (денежный поток в момент времени i)
            continuation, model_i = self.model.predict(
                X=monte_carlo_i[in_money_indices],
                y=cash_flow_i[in_money_indices],
            )

            self.models_list = [model_i] + self.models_list

            # Считаем цену исполнения сейчас (просто вектор путей в деньгах)
            exercise = self.option.path_cash_flow(
                monte_carlo_i[in_money_indices], self.strike
            )

            # Возвращаем индексы для исполнения и продолжения из правила оптимального решения
            ind_ex, ind_cont = self.optimal_early_exercise_decision(
                exercise, continuation
            )

            # Обновляем i и i - 1 вектора денежных потоков и записываем в матрицу денежных потоков
            cash_flow_matrix = self.update_cash_flow_matrix(
                cash_flow_matrix=cash_flow_matrix,
                exercise=exercise,
                exercise_ind=ind_ex,
                continuation_ind=ind_cont,
                in_money_indices=in_money_indices,
                step_num=i
            )

            if verbose:
                print(f"----- CASH FLOW MATRIX AT TIME {i} -----")
                print(cash_flow_matrix)
                print("-----------------------------------------")

        # -------------------------------------------------------------------------------
        # --------------------------------- END BODY ------------------------------------
        # -------------------------------------------------------------------------------

        self.cash_flow_matrix = cash_flow_matrix

        if verbose:
            print("----- DECISION MATRIX -----")
            print(self.decision_matrix)
            print("---------------------------")

        # -------------------------------------------------------------------------------
        # ------------- Считаем цены американского и европейских опционов ---------------
        self.american_option_price, self.european_option_price = OptionPriceCounter(
            risk_free_rate=self.risk_free_rate,
            cash_flow_matrix=self.cash_flow_matrix.copy(),
            monte_carlo_paths=self.monte_carlo_paths.copy(),
            strike=self.strike,
        ).count_option_prices()
        # -------------------------------------------------------------------------------

        # return self.convert_cash_flow_into_decision(cash_flow_matrix)

    def confident_interval(self):
        """
        :return: Возвращает 95% доверительного интервала
        """
        if self.cash_flow_matrix.size == 0:
            raise Exception(
                "Забыли сначала запустить алгоритм чтобы матрица cash flow появилась"
            )

        confident_interval = []
        # TODO: WHY ISNT DISCOUNTING???
        for path in self.cash_flow_matrix:
            confident_interval.append(
                np.amax(path) * np.exp(-self.risk_free_rate / 100 * np.argmax(path) + 1)
            )

        return (
            np.std(np.array(confident_interval))
            / np.sqrt(self.cash_flow_matrix.shape[0])
        ) * 1.645
