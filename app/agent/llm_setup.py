from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_tavily import TavilySearch
from langchain_community.tools import DuckDuckGoSearchRun
from app.agent.tools import get_places, get_hotels_by_area_and_radius, get_flight_fares, convert_currency, get_weather
from langgraph.graph import MessagesState
import os
from dotenv import load_dotenv

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.0,
    api_key=openai_key
)

tavily_tool = TavilySearch(api_key=tavily_key)

tools = [
    get_places, get_hotels_by_area_and_radius, get_flight_fares,
    convert_currency, get_weather,
    DuckDuckGoSearchRun(), tavily_tool
]

llm_with_tools = llm.bind_tools(tools)


TOOL_EDUCATION_SYSTEM_MESSAGE = SystemMessage(content="""
You are an AI-powered travel planning assistant.

You have access to real-time tools. Always prefer calling these tools over using your internal knowledge, especially for time-sensitive or location-specific queries.

🧭 get_places(city, query): 
Use to find top attractions, restaurants, or activities in any city.

🏨 get_hotels_by_area_and_radius(bbox, arrival_date, departure_date, star_rating): 
- Use this tool to fetch real-time hotel listings near a given location.
- If the user provides a **place name** (e.g., "Koramangala, Bangalore"), estimate its center coordinates and derive a bbox with ±0.045 degrees (~5 km) range.
- The `bbox` must be in the format: `"min_lat,max_lat,min_lng,max_lng"`.
- Acceptable star_rating values: "1", "2", ..., or comma-separated values like "3,4,5".

⚠️ When responding with hotel results:
- ALWAYS return the following fields per hotel:
  <code>name, star_rating, review_score, review_word, review_count, address, city, district, latitude, longitude, price_per_night, image, booking_url, is_free_cancellable, is_mobile_deal, checkin_from, checkout_until, distance_km</code>

- These must be shown in **HTML table format**, with each field in a separate column.
- DO NOT skip `latitude`, `longitude`, or `distance_km` — they are mandatory.

✈️ get_flight_fares(from_code, to_code, date): 
Use to find real-time flight fares between two cities.

💱 convert_currency(amount, to_currency, base): 
Use for all currency conversions.

🌦 get_weather(city): 
Use to get current weather info.

🔍 DuckDuckGoSearchRun or TavilySearch: 
Use these to get recent events, alerts, or headlines.

📌 Always respond in clean **HTML**, using:
- <h2> for section headers
- <ul><li> for bullet points
- <table> with <tr><td> for structured results
- Never use markdown or <html><body> tags
""")


def call_llm_with_tool_bind(state: MessagesState) -> dict:
    original_msgs = state["messages"]
    messages = [TOOL_EDUCATION_SYSTEM_MESSAGE] + original_msgs
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
