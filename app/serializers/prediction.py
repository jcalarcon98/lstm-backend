from typing import List

from pydantic import BaseModel


class Prediction(BaseModel):
    days: int
    countries: List[str]
