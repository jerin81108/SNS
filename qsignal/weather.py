import requests
import streamlit as st

@st.cache_data(ttl=1800)  # cache for 30 minutes to avoid hitting rate limits
def get_live_weather(lat=11.1085, lon=77.3411):
    """
    Fetches live weather data from Open-Meteo for the given coordinates.
    Defaults to Tiruppur, Tamil Nadu.
    """
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        current = data.get("current_weather", {})
        
        # Open-Meteo weather codes (WMO)
        wmo_codes = {
            0: ("☀️", "Clear sky"),
            1: ("🌤️", "Mainly clear"),
            2: ("⛅", "Partly cloudy"),
            3: ("☁️", "Overcast"),
            45: ("🌫️", "Fog"),
            48: ("🌫️", "Depositing rime fog"),
            51: ("🌧️", "Light drizzle"),
            53: ("🌧️", "Moderate drizzle"),
            55: ("🌧️", "Dense drizzle"),
            61: ("🌧️", "Slight rain"),
            63: ("🌧️", "Moderate rain"),
            65: ("🌧️", "Heavy rain"),
            71: ("❄️", "Slight snow"),
            73: ("❄️", "Moderate snow"),
            75: ("❄️", "Heavy snow"),
            95: ("⛈️", "Thunderstorm"),
        }
        
        code = current.get("weathercode", 0)
        # Default fallback for unknown codes
        icon, desc = wmo_codes.get(code, ("☁️", "Cloudy"))
        
        temp = current.get("temperature", "--")
        wind = current.get("windspeed", "--")
        
        return {
            "icon": icon,
            "description": desc,
            "temperature": temp,
            "windspeed": wind,
            "is_error": False
        }
    except Exception as e:
        return {
            "icon": "❓",
            "description": "Weather unavailable",
            "temperature": "--",
            "windspeed": "--",
            "is_error": True,
            "error_msg": str(e)
        }
