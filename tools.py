import os

import requests


def get_weather(city: str):
    """
    Get current weather data for a city from OpenWeatherMap.
    """

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENWEATHER_API_KEY is not configured."
        )

    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={
            "q": city,
            "appid": api_key,
            "units": "metric",
        },
        timeout=15,
     )

    response.raise_for_status()

    data = response.json()

    temperature_celsius = data["main"]["temp"]
    temperature_fahrenheit = (
        temperature_celsius * 9 / 5
    ) + 32

    return {
        "city": data.get("name", city),
        "country": data.get("sys", {}).get("country"),
        "condition": data["weather"][0]["description"],
        "temperature_celsius": round(
            temperature_celsius,
            1,
        ),
        "temperature_fahrenheit": round(
            temperature_fahrenheit,
            1,
        ),
        "humidity_percent": data["main"].get(
            "humidity"
        ),
        "wind_speed_meters_per_second": data.get(
            "wind",
            {},
        ).get("speed"),
    }


def get_location():
    """
    Get the user's approximate city and country
    from their IP address.
    """

    response = requests.get(
        "https://ipapi.co/json/",
        headers={
            "User-Agent": "weather-agent/2.1"
        },
        timeout=10,
     )

    response.raise_for_status()

    data = response.json()

    city = data.get("city")
    country = data.get("country_name")

    if not city:
        raise ValueError(
            "Could not determine your current location."
        )

    if country:
        return f"{city}, {country}"

    return city
