import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import random
import pandas as pd
from hedge.option_schema import OptionSchema
import numpy as np


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


def show_eval_of_option_price(data: pd.DataFrame, option_id: str = None):
    if option_id is None:
        option_ids = data["id"].unique()
        option_id = option_ids[random.randint(0, len(option_ids))]

    current_option = (
        data[data["id"] == option_id].sort_values(by=["dttm"]).to_dict("records")
    )

    sample_data = [
        {
            "dttm": datetime.strptime(c_o["dttm"].split(" ")[0], "%Y-%m-%d").date(),
            "option": OptionSchema.model_validate(c_o),
        }
        for c_o in current_option
    ]

    x = [el["dttm"] for el in sample_data]
    y = [el["option"].option_price for el in sample_data]

    upper = []
    lower = []
    for opt in [el["option"] for el in sample_data]:
        disc_strike = opt.strike * np.exp(
            -opt.expiration_time / opt.expiration_time_denominator * opt.risk_free_rate
        )
        if opt.optionType == "CALL":
            lower.append(max(opt.spot_price - disc_strike, 0))
            upper.append(opt.spot_price)
        else:
            lower.append(max(disc_strike - opt.spot_price, 0))
            upper.append(disc_strike)

    spot_price = [el["option"].spot_price for el in sample_data]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=lower,
            fill="tonexty",
            fillcolor="rgba(0, 100, 80, 0.2)",
            name="Lower Fair Price",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=upper,
            fill="tonexty",
            fillcolor="rgba(0, 100, 80, 0.2)",
            name="Upper Fair Price",
        )
    )
    fig.add_trace(go.Scatter(x=x, y=spot_price, name="Spot Prices"))
    fig.add_trace(go.Scatter(x=x, y=y, name="Option Prices"))
    fig.update_layout(title=str(option_id))
    fig.show()
