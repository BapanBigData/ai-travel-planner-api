from typing import Literal
from langgraph.graph import END
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from app.agent.tools import *
from app.agent.llm_setup import llm
from app.agent.router import Router, State
    
    
def supervisor(state: State) -> Command[Literal["hotel_search_expert", "weather_expert", "place_search_expert", "flight_fares_search_expert", "geolocation_expert", "__end__"]]:
    """
    The supervisor routes tasks to the appropriate expert based on the user message context.
    It uses an LLM with structured output to decide which agent to call next.
    """
    
    system_prompt = f"""
    You are a Supervisor Agent coordinating a travel assistant system with the following expert agents:
    {', '.join([
        'hotel_search_expert',
        'weather_expert',
        'place_search_expert',
        'flight_fares_search_expert',
        'geolocation_expert'
    ])}

    Each agent performs a specific task:
    - hotel_search_expert: Finds real-time hotels using a bounding box, travel dates, and optional filters (e.g., star rating)
    - weather_expert: Returns current weather for a given city
    - place_search_expert: Suggests attractions, restaurants, and cultural spots near a location
    - flight_fares_search_expert: Finds flights and fares between two IATA codes on a specific date
    - geolocation_expert: Converts place names into latitude, longitude, and bounding boxes (~5km radius)

    Your role:
    - Read the latest user query and conversation context
    - Decide which agent to invoke next
    - Handle dependencies (e.g., get bbox before hotel search)
    - Reply FINISH only after the user's request is fully satisfied

    Always include `latitude` and `longitude` for any place-related query (e.g., city, hotel, restaurant, attraction) in the final output — either in the main content or structured HTML.

    Response format:
    Return ONLY one of the following at a time:
    - The name of the agent to call next (e.g., hotel_search_expert)
    - A short explanation of why that agent is needed
    - Input hints or parameters for the agent
    - The word FINISH if all steps are complete

    Decision rules:
    - If the query involves hotels, stays, accommodation, or star ratings (e.g., 3-star, 4-star):
    - First call geolocation_expert to get a bounding box
    - Then call hotel_search_expert to find listings using the bbox and available dates
    - Do NOT stop after geolocation — hotel search must follow

    - For flight searches between cities with a date, directly call flight_fares_search_expert (do not call geolocation)

    - Always decompose multi-part goals (e.g., hotels + weather + flights) into sequential steps

    - Do NOT hallucinate inputs or output — only use what is extracted or known
    - Never assign tasks outside an expert’s scope

    Once the user's goal is complete and no other agent needs to be invoked, return FINISH.

    User Request:
    """
    
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]

    llm_with_structure_output = llm.with_structured_output(Router)
    response = llm_with_structure_output.invoke(messages)

    goto = response["next"]
    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": goto})


def hotel_search_agent(state: State) -> Command[Literal["supervisor"]]:
    """
    The hotel expert uses the hotel-related tools to find hotels based on location, star rating, and dates.
    """
    agent = create_react_agent(
        llm,
        tools=[get_hotels_by_area_and_radius, geocode_locationiq],
        prompt="""
            You are a hotel search expert. Geocode the location first if bbox is missing. 
            Then use the hotel search tool to find top results. For each hotel, return the following fields in HTML format:

            - name
            - star_rating
            - review_score
            - review_word
            - review_count
            - address
            - city
            - district
            - latitude
            - longitude
            - price_per_night
            - currency
            - image
            - booking_url
            - is_free_cancellable
            - is_mobile_deal
            - checkin_from
            - checkout_until
            - arrival_date
            - departure_date
            - distance_km
            
            `latitude` and `longitude` must always be included.
            Do not return bbox, display_name, or type in the final output. Just output a clean HTML <ul> list showing all above fields.
        """


    )

    result = agent.invoke(state)

    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="hotel_search_expert")
            ]
        },
        goto="supervisor",
    )


def weather_agent(state: State) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(
        llm,
        tools=[get_weather],
        prompt="""
            You are a weather expert. Use the weather tool to fetch real-time weather for the user's city.

            Return the result in clean HTML format using <ul>/<li> or <div>. Include:

            - City and Country
            - Description (e.g., clear sky, light rain)
            - Temperature (actual, feels like, min, max)
            - Humidity, Pressure
            - Wind speed and direction
            - Visibility (in meters), Cloud Coverage (%)
            - Sunrise and Sunset in UTC
            - Latitude and Longitude (always include)
            - Optional: weather icon <img> if available
            
            `latitude` and `longitude` must always be included.
            Do not use JSON or plain text. Only output valid, well-structured HTML.
        """


    )

    result = agent.invoke(state)

    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content, name="weather_expert")]},
        goto="supervisor"
    )


def place_search_agent(state: State) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(
        llm,
        tools=[get_places],
        prompt="""
            You are an expert in finding attractions, restaurants, and cultural places.

            Use the get_places tool to return the top 5 places.

            For each result, include:
            - Name
            - Category (e.g., museum, restaurant)
            - Address
            - Latitude and Longitude (always include)
            - Phone (if available)
            - Website (as a clickable link)
            
            `latitude` and `longitude` must always be included.
            Return the output in clean HTML format using <ul>/<li> or <div>. No JSON or plain text. Keep it structured and readable.
        """


    )

    result = agent.invoke(state)

    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content, name="place_search_expert")]},
        goto="supervisor"
    )


def flight_fares_search_agent(state: State) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(
        llm,
        tools=[get_flight_fares],
        prompt="""
            You are a flight search expert. Use the flight API to return flight fares and details.

            For each result, include:
            - From and To airport codes
            - Departure and Arrival time
            - Airline and number of Stops
            - Fare with currency

            Respond in HTML format using <ul>/<li> or <div>. Avoid JSON or plain text.
            """


    )

    result = agent.invoke(state)

    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content, name="flight_fares_search_expert")]},
        goto="supervisor"
    )

def geolocation_agent(state: State) -> Command[Literal["supervisor"]]:
    agent = create_react_agent(
        llm,
        tools=[geocode_locationiq],
        prompt="""
            You are a geolocation expert. Convert place names into coordinates and bounding boxes.

            Always include:
            - Display Name
            - Latitude and Longitude
            - Bounding Box (min_lat, max_lat, min_lon, max_lon)
            - Type (e.g., city, hotel, attraction)
            
            `latitude` and `longitude` must always be included.
            Respond using HTML format with <ul>/<li> or <div>. Do not use JSON or plain text.
        """


    )

    result = agent.invoke(state)

    return Command(
        update={"messages": [HumanMessage(content=result["messages"][-1].content, name="geolocation_expert")]},
        goto="supervisor"
    )
