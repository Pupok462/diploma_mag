import numpy as np
from typing import Union, Literal, List
from pydantic import BaseModel, ConfigDict, Field
from numpy.typing import NDArray
from lss.models import MetaModel, NaiveModel
from lss.options import MetaOption, PutOption, CallOption


class LongStaffSchwartz(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    risk_free_rate: float = Field(default=5.8239, gt=0)
    strike: float = Field(default=1.1, gt=0)
    option: MetaOption = Field(default=PutOption())
    model: MetaModel = Field(default=NaiveModel(risk_free_rate=5.8239))
    monte_carlo_paths: NDArray = Field(
        default=np.array(
            [
                [1, 1.09, 1.08, 1.34],
                [1, 1.16, 1.26, 1.54],
                [1, 1.22, 1.07, 1.03],
                [1, 0.93, 0.97, 0.92],
                [1, 1.11, 1.56, 1.52],
                [1, 0.76, 0.77, 0.90],
                [1, 0.92, 0.84, 1.01],
                [1, 0.88, 1.22, 1.34],
            ]
        )
    )

    cash_flow_matrix: NDArray = Field(default=np.array([]))
    models_list: List = Field(default=[])

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

    @staticmethod
    def update_cash_flow_matrix(
        cash_flow_matrix: np.array,
        exercise,
        exercise_ind,
        regression_indices,
        step_num,
    ) -> np.array:
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

    @staticmethod
    def convert_cash_flow_into_decision(cash_flow_matrix: np.array) -> np.array:
        """
        :param cash_flow_matrix: матрица денежных потоков
        :return: Матрица 1 и 0 обозначающая остановки и продолжения
        """
        return np.where(cash_flow_matrix > 0, 1, 0)

    def evaluate(self, verbose: bool = False):
        """

        :param verbose:
        :return: Decision matrix and american price
        """

        # Создаем матрицу денежных потоков (количество траекторий, длина траекторий)
        self.cash_flow_matrix = []
        self.models_list = []

        cash_flow_matrix = np.zeros(
            (self.monte_carlo_paths.shape[0], self.monte_carlo_paths.shape[1] - 1)
        )

        if verbose:
            print("----- INITIAL CASH FLOW MATRIX -----")
            print(cash_flow_matrix)
            print("------------------------------------")

        n = cash_flow_matrix.shape[1] - 1

        # Запишем денежные потоки на последнем шаге в последний столбец матрицы cf_matrix
        cash_flow_matrix[:, -1] = self.option.path_cash_flow(
            self.monte_carlo_paths[:, -1], self.strike
        )

        if verbose:
            print(f"----- CASH FLOW MATRIX AT TIME {n + 1}-----")
            print(cash_flow_matrix)
            print("------------------------------------")

        # ---------------------------------------------------------------------------------
        # --------------------------------- MAIN BODY -------------------------------------
        # ---------------------------------------------------------------------------------

        # Итерируемся в обратном порядке для подсчета цены продолжения и исполнения
        for i in range(n, 0, -1):
            # Берем вектор денежных потоков на i шаге и цены опциона на шаге i
            cash_flow_i = cash_flow_matrix[:, i:]
            monte_carlo_i = self.monte_carlo_paths[:, i]

            # Записываем индексы путей которые в деньгах в зависимости от типа опциона
            in_money_mask = self.option.path_mask(monte_carlo_i, self.strike)
            in_money_indices = np.where(in_money_mask)[0]

            # Вычисляем вектор продолжения регрессируя и вектор исполнения (денежный поток в момент времени i)
            continuation, model_i = self.model.predict(
                X=monte_carlo_i[in_money_indices], y=cash_flow_i[in_money_indices]
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
            print(self.convert_cash_flow_into_decision(cash_flow_matrix))
            print("---------------------------")

        return (
            self.convert_cash_flow_into_decision(cash_flow_matrix),
            self._american_option_price(),
        )

    def get_decisions(self):
        return self.convert_cash_flow_into_decision(self.cash_flow_matrix)

    def _american_option_price(self):
        option_price = 0

        risk_free_rate = self.risk_free_rate/100

        for col in range(self.cash_flow_matrix.shape[1]):
            option_price += np.sum(self.cash_flow_matrix[:, col]) * np.exp(-risk_free_rate * col/365)

        return option_price / self.cash_flow_matrix.shape[0]

    def _european_option_price(self):
        """
        :param self.cash_flow_matrix:
        :return:
        """
        risk_free_rate = self.risk_free_rate / 100

        return (
                np.sum(
                    self.option.path_cash_flow(self.monte_carlo_paths[:, -1], self.strike)
                )
                * np.exp(- risk_free_rate * (self.monte_carlo_paths.shape[1]) / 365)
                / self.monte_carlo_paths.shape[0]
        )

    def option_price(
        self, type_of_option: Literal["european", "american", "all"] = "all"
    ):
        if self.cash_flow_matrix.size == 0:
            raise Exception(
                "Забыли сначала запустить алгоритм чтобы матрица cash flow появилась"
            )

        if type_of_option == "american":
            return f"Evolute the option price: {self._american_option_price()}"

        elif type_of_option == "european":
            return f"European option price:  {self._european_option_price()}"

        elif type_of_option == "all":
            return (
                f"Evolute the option price:  {self._american_option_price()} \n"
                f"European option price:  {self._european_option_price()}"
            )

    def confident_interval(self):
        """
        :return: Возвращает 95% доверительного интервала
        """
        if self.cash_flow_matrix.size == 0:
            raise Exception(
                "Забыли сначала запустить алгоритм чтобы матрица cash flow появилась"
            )

        confident_interval = []
        for path in self.cash_flow_matrix:
            confident_interval.append(
                np.amax(path) * np.exp(- self.risk_free_rate/100 * np.argmax(path) + 1)
            )

        return (
            np.std(np.array(confident_interval))
            / np.sqrt(self.cash_flow_matrix.shape[0])
        ) * 1.645


class OutOfSample(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    risk_free_rate: float = Field(default=5.8239, gt=0)
    strike: float = Field(default=1.1, gt=0)
    option: MetaOption = Field(default=PutOption())
    models_list: List = Field(default=[])
    monte_carlo_paths: NDArray = Field(
        default=np.array(
            [
                [1, 1.09, 1.08, 1.34],
                [1, 1.16, 1.26, 1.54],
                [1, 1.22, 1.07, 1.03],
                [1, 0.93, 0.97, 0.92],
                [1, 1.11, 1.56, 1.52],
                [1, 0.76, 0.77, 0.90],
                [1, 0.92, 0.84, 1.01],
                [1, 0.88, 1.22, 1.34],
            ]
        )
    )

    cash_flow_matrix: NDArray = Field(default=np.array([]))

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

    def _american_option_price(self):
        option_price = 0

        risk_free_rate = self.risk_free_rate/100

        for col in range(self.cash_flow_matrix.shape[1]):
            option_price += np.sum(self.cash_flow_matrix[:, col]) * np.exp(- risk_free_rate * col/365)

        return option_price / self.cash_flow_matrix.shape[0]

    def _european_option_price(self):
        """
        :param self.cash_flow_matrix:
        :return:
        """
        risk_free_rate = self.risk_free_rate / 100

        return (
                np.sum(
                    self.option.path_cash_flow(self.monte_carlo_paths[:, -1], self.strike)
                )
                * np.exp(- risk_free_rate * (self.monte_carlo_paths.shape[1]) / 365)
                / self.monte_carlo_paths.shape[0]
        )

    def evaluate(self, verbose: bool = False):
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

        return self._american_option_price(), self._european_option_price()
