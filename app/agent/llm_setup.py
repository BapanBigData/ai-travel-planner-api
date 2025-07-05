from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_tavily import TavilySearch
from langchain_community.tools import DuckDuckGoSearchRun
from app.agent.tools import get_places, get_hotels_by_city, get_flight_fares, convert_currency, get_weather
from langgraph.graph import MessagesState
import os
from dotenv import load_dotenv

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.3,
    api_key=openai_key
)

tavily_tool = TavilySearch(api_key=tavily_key)

tools = [
    get_places, get_hotels_by_city, get_flight_fares,
    convert_currency, get_weather,
    DuckDuckGoSearchRun(), tavily_tool
]

llm_with_tools = llm.bind_tools(tools)

TOOL_EDUCATION_SYSTEM_MESSAGE = SystemMessage(content="""
You are an AI-powered travel planning assistant.

You have access to a set of real-time tools. Always prefer calling these tools over using your internal knowledge, especially for up-to-date or location-specific information.

🧭 get_places(city, query): Use to find top attractions, restaurants, or activities in any city.
🏨 get_hotels_by_city(city): Use to fetch real-time hotel listings and price per night in INR.
✈️ get_flight_fares(from_code, to_code, date): Use to find real-time flight fares between two cities.
💱 convert_currency(amount, to_currency, base): Use this for ALL currency conversions.
🌦 get_weather(city): Use to get the current weather and temperature for any city.
🔍 DuckDuckGoSearchRun and TavilySearch: Use these tools to get the latest info or safety alerts.

📌 VERY IMPORTANT: Return all your answers in clean **HTML format** suitable for rendering in a browser.

Use structured tags like:
- <h2> for section titles
- <ul><li> for lists
- <p> for text
- <b> for highlights
- Use tables <table><tr><td> where helpful

Example:

<h2>Flight Details</h2>
<ul>
  <li><b>Airline:</b> Thai Airways</li>
  <li><b>Departure:</b> July 5, 2025</li>
  <li><b>Price:</b> ₹16,000</li>
</ul>

Ensure proper indentation and valid HTML. Do NOT return markdown. Do NOT include <html> or <body> tags.
""")

def call_llm_with_tool_bind(state: MessagesState) -> dict:
    original_msgs = state["messages"]
    messages = [TOOL_EDUCATION_SYSTEM_MESSAGE] + original_msgs
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
