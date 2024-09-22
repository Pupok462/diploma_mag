from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any


class OptionSchema(BaseModel):
    full_name: str = Field(
        description="Название опциона на бирже, Например: BTC-8DEC23-24000-C",
        alias="id",
    )
    spot_price: float = Field(description="Цена базового актива cейчас", alias="S")
    spot_price_prev: float = Field(description="Цена базового актива на прошлом шаге", alias="S_prev")
    strike: float = Field(description="Страйк цена", alias="K", default=None)
    option_price: float = Field(description="Цена опциона Close", alias="price")
    expiration_time: float = Field(
        description="Сколько временных шагов (дней/часов/минут) живет опционов, Например: 30/365",
        alias="T",
        default=None,
    )
    expiration_time_denominator: float = Field(
        "Знаменатель для того чтобы перевести дни/часы/минуты относительно года",
        alias="T_denominator",
    )
    risk_free_rate: float = Field(
        description="Безрисковая процентная ставка", alias="r"
    )
    optionType: str = Field(description="Тип опциона", default=None)
    base: str = Field(description="Название криптовалюты", default=None)

    @classmethod
    def model_validate(
        cls: type[BaseModel],
        obj: Any,
        *,
        strict: bool | None = None,
        from_attributes: bool | None = None,
        context: dict[str, Any] | None = None,
    ) -> BaseModel:
        obj["base"], exp_date, K, op_type = obj["id"].split("-")

        obj["K"] = float(K)

        today_date = datetime.strptime(obj["dttm"].split(" ")[0], "%Y-%m-%d")
        exp_date = datetime.strptime(exp_date, "%d%b%y")
        obj["T"] = (exp_date - today_date).days
        obj["T_denominator"] = 365

        if op_type == "C":
            obj["optionType"] = "CALL"
        elif op_type == "P":
            obj["optionType"] = "PUT"

        return super().model_validate(
            obj=obj, strict=strict, from_attributes=from_attributes, context=context
        )
