import matplotlib.pyplot as plt


def show_simulations(simulations):
    """

    Отображение графика всех Монте-Карло траекторий.
    :param simulations: Массив Монте-Карло траекторий.
    """

    for i in range(simulations.shape[0]):
        plt.plot(simulations[i])

    plt.title("Все Монте-Карло пути")
    plt.xlabel("Шаг")
    plt.ylabel("Значение")
    plt.show()
