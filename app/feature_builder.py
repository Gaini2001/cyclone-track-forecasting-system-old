from datetime import datetime
import numpy as np


def build_features(data):

    now = datetime.utcnow()

    month = now.month
    day = now.day
    hour = now.hour

    day_of_year = now.timetuple().tm_yday

    features = {

        "LAT": data["latitude"],
        "LON": data["longitude"],

        "WMO_WIND": data["wind_speed"],
        "WMO_PRES": data["pressure"],

        "MONTH": month,
        "DAY": day,
        "HOUR": hour,

        "DAY_OF_YEAR": day_of_year,

        "MONTH_SIN":
            np.sin(2*np.pi*month/12),

        "MONTH_COS":
            np.cos(2*np.pi*month/12),

        "HOUR_SIN":
            np.sin(2*np.pi*hour/24),

        "HOUR_COS":
            np.cos(2*np.pi*hour/24),

        # temporary placeholders
        "LAT_LAG_1": data["latitude"],
        "LAT_LAG_2": data["latitude"],
        "LAT_LAG_3": data["latitude"],

        "LON_LAG_1": data["longitude"],
        "LON_LAG_2": data["longitude"],
        "LON_LAG_3": data["longitude"],

        "WIND_LAG_1": data["wind_speed"],
        "WIND_LAG_2": data["wind_speed"],
        "WIND_LAG_3": data["wind_speed"],

        "PRESSURE_LAG_1": data["pressure"],
        "PRESSURE_LAG_2": data["pressure"],
        "PRESSURE_LAG_3": data["pressure"],

        "DELTA_LAT": 0,
        "DELTA_LON": 0,

        "DELTA_WIND": 0,
        "DELTA_PRESSURE": 0,

        "MOVEMENT_DISTANCE": 0,
    }

    return features