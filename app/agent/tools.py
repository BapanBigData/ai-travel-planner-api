import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()


@tool
def get_places(city: str, query: str = "attractions") -> list:
    """
    Fetches top places (e.g., attractions, restaurants) in a city using the Foursquare Places API
    and returns them as a structured JSON list.

    Parameters:
        city (str): Name of the city (e.g., 'Kolkata')
        query (str): Type of places to search for (e.g., 'attractions', 'museums')

    Returns:
        list: A list of dictionaries, each containing:
            - name: Place name
            - categories: List of category names
            - address: Formatted address
            - phone: Telephone number if available
            - website: Website URL if available
"""
    api_key = os.getenv("FOURSQUARE_API_KEY")
    if not api_key:
        return [{"error": "Missing FOURSQUARE_API_KEY"}]

    url = "https://places-api.foursquare.com/places/search"
    headers = {
        "accept": "application/json",
        "X-Places-Api-Version": "2025-06-17",
        "authorization": api_key
    }
    params = {
        "near": city,
        "query": query,
        "limit": 10
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        return [{"error": f"Foursquare API error: {response.text}"}]

    results = response.json().get("results", [])
    if not results:
        return [{"message": f"No results found for '{query}' in {city}."}]

    extracted = []
    for place in results:
        name = place.get("name", "Unknown")
        categories = [cat.get("name") for cat in place.get("categories", [])]
        address = place['location']['formatted_address']
        phone = place.get("tel", None)
        website = place.get("website", None)

        extracted.append({
            "name": name,
            "categories": categories,
            "address": address,
            "phone": phone,
            "website": website
        })

    return extracted


@tool
def get_hotels_by_city(city: str) -> list:
    """
    Get top hotels with prices per night in INR for the given city using Hotels4 (RapidAPI).
    
    Parameters:
        city (str): City name (e.g., "Kolkata")

    Returns:
        list: List of hotels with name, price, and address
    """

    headers = {
        "x-rapidapi-key": os.getenv("RAPIDAPI_KEY_HOTELS"),
        "x-rapidapi-host": "hotels4.p.rapidapi.com"
    }

    # Step 1: Get gaiaId (destination identifier)
    location_url = "https://hotels4.p.rapidapi.com/locations/v3/search"
    location_params = {
        "q": city,
        "locale": "en_US",
        "langid": "1033",
        "siteid": "3000000"
    }

    location_resp = requests.get(location_url, headers=headers, params=location_params)
    if location_resp.status_code != 200:
        return [{"error": f"Location fetch failed: {location_resp.text}"}]

    try:
        gaia_id = location_resp.json()["sr"][0]["gaiaId"]
    except Exception:
        return [{"error": f"Could not find destination ID for city: {city}"}]

    # Step 2: Get hotel list for destination
    hotel_url = "https://hotels4.p.rapidapi.com/properties/v2/list"
    today = datetime.today()

    payload = {
        "currency": "INR",
        "locale": "en_US",
        "siteId": 3000000,
        "destination": {"regionId": gaia_id},
        "checkInDate": {
            "day": today.day,
            "month": today.month,
            "year": today.year
        },
        "checkOutDate": {
            "day": (today + timedelta(days=1)).day,
            "month": (today + timedelta(days=1)).month,
            "year": (today + timedelta(days=1)).year
        },
        "rooms": [{"adults": 1}],
        "resultsStartingIndex": 0,
        "resultsSize": 20,
        # "sort": "PRICE_LOW_TO_HIGH",
        "filters": {}
    }

    hotel_resp = requests.post(hotel_url, json=payload, headers=headers)
    if hotel_resp.status_code != 200:
        return [{"error": f"Hotel list fetch failed: {hotel_resp.text}"}]

    hotels_raw = hotel_resp.json().get("data", {}).get("propertySearch", {}).get("properties", [])
    if not hotels_raw:
        return [{"message": f"No hotel results found in {city}"}]

    results = []
    for hotel in hotels_raw:
        # print(hotel)
        # print('*'*50)
        # print()
        name = hotel.get("name")
        address = hotel.get("address", {}).get("addressLine", "No address provided")
        price = hotel.get("price", {}).get("lead", {}).get("formatted", "N/A")
        results.append({
            "name": name,
            "address": address,
            "price_per_night": price
        })

    return results


@tool
def convert_currency(amount: float, to_currency: str, base: str = "USD") -> float:
    """
    Convert a monetary amount from one currency to another using real-time exchange rates.

    This function fetches the latest exchange rate between the specified base currency
    and the target currency using the ExchangeRate-API and calculates the converted value.

    Args:
        amount (float): The amount of money to convert.
        to_currency (str): The target currency code (e.g., "EUR", "INR").
        base (str, optional): The source currency code. Defaults to "USD".

    Returns:
        float: The converted amount in the target currency, rounded to two decimal places.
                If the exchange rate is unavailable, returns a dictionary with an error message.

    Example:
        >>> convert_currency(100, "INR")
        8356.25

    Notes:
        - This tool uses the open endpoint from https://open.er-api.com.
        - Ensure that `to_currency` and `base` are valid ISO currency codes.
    """
    url = f"https://open.er-api.com/v6/latest/{base}"
    resp = requests.get(url)
    data = resp.json()
    rate = data["rates"].get(to_currency)
    if not rate:
        return {"error": f"Rate unavailable for {to_currency}"}
    return round(amount * rate, 2)



@tool
def get_weather(city: str) -> dict:
    """Get detailed current weather data for a city as a dictionary."""
    
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    
    if response.status_code != 200:
        return {"error": f"Failed to get weather: {response.text}"}
    
    data = response.json()
    
    return {
        "city": data.get("name"),
        "country": data.get("sys", {}).get("country"),
        "description": data.get("weather", [{}])[0].get("description"),
        "temperature_celsius": data.get("main", {}).get("temp"),
        "feels_like_celsius": data.get("main", {}).get("feels_like"),
        "temp_min": data.get("main", {}).get("temp_min"),
        "temp_max": data.get("main", {}).get("temp_max"),
        "humidity": data.get("main", {}).get("humidity"),
        "pressure": data.get("main", {}).get("pressure"),
        "wind_speed_mps": data.get("wind", {}).get("speed"),
        "wind_deg": data.get("wind", {}).get("deg"),
        "visibility_m": data.get("visibility"),
        "cloud_coverage_percent": data.get("clouds", {}).get("all"),
        "sunrise_utc": data.get("sys", {}).get("sunrise"),
        "sunset_utc": data.get("sys", {}).get("sunset"),
        "icon": data.get("weather", [{}])[0].get("icon")
    }



@tool
def get_flight_fares(from_code: str, to_code: str, date: str, adult: int = 1, type_: str = "economy") -> list:
    """
    Fetches flight fare data using the Flight Fare Search API on RapidAPI.

    Args:
        from_code (str): IATA code of departure airport (e.g., 'BLR')
        to_code (str): IATA code of arrival airport (e.g., 'CCU')
        date (str): Travel date in YYYY-MM-DD
        adult (int): Number of adult passengers (default: 1)
        type_ (str): Cabin class (default: 'economy')

    Returns:
        list: List of flights with key details: timing, pricing, stops, countries, and cabin info.
    """
    url = "https://flight-fare-search.p.rapidapi.com/v2/flights/"

    querystring = {
        "from": from_code,
        "to": to_code,
        "date": date,
        "adult": str(adult),
        "type": type_,
        "currency": "USD"
    }

    headers = {
        "x-rapidapi-key": os.getenv("RAPIDAPI_KEY_FLIGHTS"),
        "x-rapidapi-host": "flight-fare-search.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    # print("🔍 Raw API response:", response.status_code, response.text)

    try:
        raw = response.json()
        flights = raw.get("results", [])
        if not isinstance(flights, list) or not flights:
            return [{"message": "No flights found."}]

        results = []
        for f in flights:
            stop_info = []
            stop_summary = f.get("stopSummary", {})

            # Extract intermediate stops if present
            if isinstance(stop_summary, dict):
                for key, val in stop_summary.items():
                    if key != "connectingTime" and isinstance(val, dict):
                        stop_info.append({
                            "intermediate_airport": val.get("airport", "Unknown"),
                            "stop_duration_minutes": val.get("stopDuration")
                        })

            results.append({
                "flight_code": f.get("flight_code"),
                "airline": f.get("flight_name"),
                "cabin_type": f.get("cabinType", "Unknown"),
                "stops": f.get("stops", "Unknown"),
                "departure_city": f.get("departureAirport", {}).get("city"),
                "departure_country": f.get("departureAirport", {}).get("country", {}).get("label"),
                "departure_time": f.get("departureAirport", {}).get("time"),
                "arrival_city": f.get("arrivalAirport", {}).get("city"),
                "arrival_country": f.get("arrivalAirport", {}).get("country", {}).get("label"),
                "arrival_time": f.get("arrivalAirport", {}).get("time"),
                "duration": f.get("duration", {}).get("text"),
                "price": f.get("totals", {}).get("total"),
                "currency": f.get("totals", {}).get("currency"),
                "intermediate_stops": stop_info if stop_info else None
            })

        return results

    except Exception as e:
        return [{"error": str(e)}]
