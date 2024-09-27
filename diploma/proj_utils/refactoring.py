import pandas as pd
from hedge.option_schema import OptionSchema
import numpy as np


def check_for_duplicate_dates(data: pd.DataFrame):
    data = data.sort_values(by=["id", "dttm"])

    date_prev = None
    option_id_prev = None

    bad_dates = 0

    for idx, row in data.iterrows():
        if date_prev is None:
            date_prev = row.dttm
            option_id_prev = row.id
            continue

        if option_id_prev != row.id:
            option_id_prev = row.id
            date_prev = row.dttm
            continue

        if date_prev == row.dttm and option_id_prev == row.id:
            bad_dates += 1
            date_prev = row.dttm

    return bad_dates, round(bad_dates / data.shape[0] * 100, 2)


def find_arbitrage(row):
    """
        max(S_0 − K, 0) < EC < S_0
        max(K − S_0, 0) < EP < K

    :param row:
    :return:
    """
    dict_row = row.to_dict()
    option = OptionSchema.model_validate(dict_row)

    strike = option.strike * np.exp(
        -option.expiration_time
        / option.expiration_time_denominator
        * option.risk_free_rate
    )

    if option.optionType == "CALL":
        if max(option.spot_price - strike, 0) < option.option_price < option.spot_price:
            return 0
        else:
            return 1
    else:
        if max(strike - option.spot_price, 0) < option.option_price < strike:
            return 0
        else:
            return 1
