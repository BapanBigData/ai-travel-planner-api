import os
import math
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


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance between two points on Earth."""
    R = 6371  # Earth radius in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@tool
def get_hotels_by_area_and_radius(
    bbox: str,
    arrival_date: str,
    departure_date: str,
    star_rating: str = "3,4,5",
    room_qty: int = 1,
    guest_qty: int = 1,
    children_qty: int = 0,
    children_age: str = "",
    currency: str = "USD",
    order_by: str = "popularity",
    categories_filter: str = "class::1,class::2,class::3",
    language: str = "en-us",
    travel_purpose: str = "leisure",
    offset: int = 0
) -> list:
    """
    Fetch hotel listings within a bounding box and return only key fields, sorted by distance to bbox center.

    Parameters:
        - bbox (str): Bounding box in format "min_lat,max_lat,min_lng,max_lng"
        - star_rating (str): Comma-separated star classes to filter, e.g., "3,4,5"
        - arrival_date (str): Check-in date (YYYY-MM-DD)
        - departure_date (str): Check-out date (YYYY-MM-DD)
        - room_qty (int): Number of rooms
        - guest_qty (int): Number of adults
        - children_qty (int): Number of children
        - children_age (str): Comma-separated list of children ages
        - currency (str): Price currency (e.g., USD, INR)
        - order_by (str): API sort preference (not used post-filter)
        - categories_filter (str): Used internally, overridden by star_rating
        - language (str): Response language
        - travel_purpose (str): "leisure" or "business"
        - offset (int): Pagination offset

    Returns:
        - list of hotel dicts sorted by ascending distance from bbox center.
    """

    # Convert star_rating to API-compatible format
    categories_filter = ",".join([f"class::{s.strip()}" for s in star_rating.split(",")])

    # Compute bbox center
    try:
        min_lat, max_lat, min_lng, max_lng = map(float, bbox.split(","))
        center_lat = (min_lat + max_lat) / 2
        center_lng = (min_lng + max_lng) / 2
    except Exception as e:
        return [{"error": f"Invalid bbox format: {e}"}]

    url = "https://apidojo-booking-v1.p.rapidapi.com/properties/list-by-map"
    
    querystring = {
        "room_qty": str(room_qty),
        "guest_qty": str(guest_qty),
        "bbox": bbox,
        "search_id": "none",
        "children_age": children_age,
        "price_filter_currencycode": currency,
        "categories_filter": categories_filter,
        "languagecode": language,
        "travel_purpose": travel_purpose,
        "children_qty": str(children_qty),
        "order_by": order_by,
        "offset": str(offset),
        "arrival_date": arrival_date,
        "departure_date": departure_date
    }
    headers = {
        "x-rapidapi-key": os.getenv("RAPIDAPI_KEY_HOTELS"),
        "x-rapidapi-host": "apidojo-booking-v1.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)
    if response.status_code != 200:
        return [{"error": response.text}]

    results = response.json().get("result", [])
    hotels = []

    for item in results:
        if not item.get("class"):
            continue

        lat = item.get("latitude")
        lng = item.get("longitude")
        if lat is None or lng is None:
            continue

        distance_km = haversine_distance(center_lat, center_lng, lat, lng)

        hotels.append({
            "name": item.get("hotel_name"),
            "star_rating": item.get("class"),
            "review_score": item.get("review_score"),
            "review_word": item.get("review_score_word"),
            "review_count": item.get("review_nr"),
            "address": item.get("address"),
            "city": item.get("city"),
            "district": item.get("district"),
            "latitude": lat,
            "longitude": lng,
            "price_per_night": item.get("min_total_price") or (
                item.get("price_breakdown", {}).get("all_inclusive_price")
            ),
            "currency": item.get("currencycode", currency),
            "image": item.get("main_photo_url"),
            "booking_url": item.get("url"),
            "is_free_cancellable": item.get("is_free_cancellable"),
            "is_mobile_deal": item.get("is_mobile_deal"),
            "checkin_from": item.get("checkin", {}).get("from"),
            "checkout_until": item.get("checkout", {}).get("until"),
            "distance_km": round(distance_km, 2)
        })

    # Sort by distance from bbox center
    hotels.sort(key=lambda h: h["distance_km"])
    return hotels


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
    
    url = "https://flight-fare-search.p.rapidapi.com/v2/flights"

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
