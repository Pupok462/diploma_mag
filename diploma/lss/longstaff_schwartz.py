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

    @property
    def decision_matrix(self):
        """
        :return: Матрица 1 и 0 обозначающая остановки и продолжения
        """
        assert self.cash_flow_matrix.size != 0, "Матрица денежных потоков пустая, запустите сначала метод evolute()"
        return np.where(self.cash_flow_matrix > 0, 1, 0)

    @staticmethod
    def update_cash_flow_matrix(
        cash_flow_matrix: NDArray,
        exercise,
        exercise_ind,
        regression_indices,
        step_num,
    ) -> NDArray:
        """

        :param cash_flow_matrix:
        :param exercise:
        :param exercise_ind:
        :param regression_indices:
        :param step_num:
        :return: Обновляем матрицу денежных потоков исходя из матрицы продолжения или исполнения (обновляются N столбец и N-1)
        """
        # Обновляем матрицу денежных потоков на шаге N - 1
        cash_flow_matrix[:, step_num - 1][regression_indices[exercise_ind]] = exercise[
            exercise_ind
        ]

        # Обновляем матрицу денежных потоков на шаге N
        cash_flow_matrix[regression_indices[exercise_ind], step_num] = 0

        return cash_flow_matrix


    def evaluate(self, verbose: bool = False) -> NoReturn:
        """
            Запуск алгоритма LongStaffSchwartz

        :param verbose:
        """

        # Инициализация матрицы денежных потоков (полностью нули)
        # Столбцов на 1 меньше чем в МК симуляциях, так как мы не учитываем S_0 цену
        cash_flow_matrix = np.zeros((self.monte_carlo_paths.shape[0], self.monte_carlo_paths.shape[1] - 1)) #TODO: APPROVED

        if verbose:
            print("----- INITIAL CASH FLOW MATRIX -----")
            print(cash_flow_matrix)
            print("------------------------------------")

        # Запишем денежные потоки на последнем шаге в последний столбец матрицы cash_flow_matrix
        cash_flow_matrix[:, -1] = self.option.path_cash_flow(
            self.monte_carlo_paths[:, -1].copy(), self.strike #TODO: APPROVED
        )

        if verbose:
            print(f"----- CASH FLOW MATRIX AT TIME {cash_flow_matrix.shape[1]}-----")
            print(cash_flow_matrix)
            print("------------------------------------")

        # ---------------------------------------------------------------------------------
        # --------------------------------- MAIN BODY -------------------------------------
        # ---------------------------------------------------------------------------------

        # Итерируемся в обратном порядке для подсчета цены продолжения и исполнения
        for i in range(cash_flow_matrix.shape[1] - 1, 0, -1): #TODO: APPROVED

            # Берем часть матрицы денежных потоков на i и больше
            cash_flow_i = cash_flow_matrix[:, i:]
            # И берем цены базового актива на шаге i
            monte_carlo_i = self.monte_carlo_paths[:, i] #TODO: APPROVED

            # Записываем индексы путей которые в деньгах в зависимости от типа опциона
            in_money_mask = self.option.path_mask(monte_carlo_i, self.strike) #TODO: APPROVED
            in_money_indices = np.where(in_money_mask)[0] #TODO: APPROVED

            # Вычисляем вектор продолжения регрессируя и вектор исполнения (денежный поток в момент времени i)
            continuation, model_i = self.model.predict(
                X=monte_carlo_i[in_money_indices], y=cash_flow_i[in_money_indices] #TODO: APPROVED
            )


            self.models_list = [model_i] + self.models_list

            # Считаем цену исполнения сейчас (просто вектор путей в деньгах)
            exercise = self.option.path_cash_flow(
                monte_carlo_i[in_money_indices], self.strike
            )

            # Возвращаем индексы для исполнения и продолжения из правила оптимального решения
            ind_ex = np.where(exercise > continuation)[0]
            # ind_ex, ind_cont = self.optimal_early_exercise_decision(
            #     exercise, continuation
            # )

            # Обновляем i и i - 1 вектора денежных потоков и записываем в матрицу денежных потоков
            cash_flow_matrix = self.update_cash_flow_matrix(
                cash_flow_matrix, exercise, ind_ex, in_money_indices, i
            )

            if verbose:
                print(f"----- CASH FLOW MATRIX AT TIME {i} -----")
                print(cash_flow_matrix)
                print("-----------------------------------------")

        # -------------------------------------------------------------------------------
        # --------------------------------- END BODY ------------------------------------
        # -------------------------------------------------------------------------------

        # Зануляем те элементы которые не раз встречаются не в нулевом формате
        for raw in range(len(cash_flow_matrix)):
            fnz_i = np.argmax(cash_flow_matrix[raw] != 0)
            cash_flow_matrix[raw, fnz_i + 1 :] = 0

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
            strike=self.strike
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
                np.amax(path) * np.exp(- self.risk_free_rate/100 * np.argmax(path) + 1)
            )

        # TODO: WTF IS COEF 1.645???
        return (
            np.std(np.array(confident_interval))
            / np.sqrt(self.cash_flow_matrix.shape[0])
        ) * 1.645

