from pydantic import BaseModel


class CycloneInput(BaseModel):

    latitude: float
    longitude: float

    wind_speed: float
    pressure: float