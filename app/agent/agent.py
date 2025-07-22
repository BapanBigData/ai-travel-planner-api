from typing import Literal
from langgraph.graph import END
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from app.agent.tools import *
from app.agent.llm_setup import llm
from app.agent.router import Router, State

async def supervisor(state: State) -> Command[Literal["hotel_search_expert", "weather_expert", "place_search_expert", "flight_fares_search_expert", "__end__"]]:
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
        'flight_fares_search_expert'
    ])}

    Each agent performs a specific task:
    - hotel_search_expert: Finds real-time hotels using a given location, travel dates, and optional filters (e.g., star rating)
    - weather_expert: Returns current weather for a given city
    - place_search_expert: Suggests attractions, restaurants, and cultural spots near a location
    - flight_fares_search_expert: Finds flights and fares between two IATA codes on a specific date

    Your responsibilities:
    1. Carefully analyze the user's message to understand the full intent.
    2. Decide which expert agent is required based on the query content.
    3. Call only the specific agents that are actually relevant to the query.
    4. Do not hallucinate — never invoke an agent that isn't relevant or make assumptions about the user's intent.
    5. Return FINISH only when all relevant tasks are complete.

    Important Guidance:
    - Always decompose multi-part queries (e.g., hotel + weather + flights) into sequential steps handled by different agents.
    - Place-related queries (e.g., cities, attractions, hotels) must include `latitude` and `longitude` in the final output — these are calculated internally by the tools, not by a separate agent.
    - Do NOT fabricate input data. Only act on what is explicitly provided or can be reasonably inferred.
    - Use only the available agents listed above. Never reference or call agents that do not exist.

    Response format:
    Return ONLY one of the following:
    - The name of the agent to call next (e.g., hotel_search_expert)
    - A short explanation of why that agent is needed
    - Input hints or parameters for the agent
    - The word FINISH if the user's request has been fully satisfied
    
    
    User Query:
    """

    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    
    llm_with_structure_output = llm.with_structured_output(Router)
    response = await llm_with_structure_output.ainvoke(messages)
    
    goto = response["next"]
    
    if goto == "FINISH":
        goto = END
        
    return Command(goto=goto, update={"next": goto})


async def hotel_search_agent(state: State) -> Command[Literal["supervisor"]]:
    """
    The hotel expert uses the hotel-related tools to find hotels based on location, star rating, and dates.
    """
    agent = create_react_agent(
        llm, 
        tools=[get_hotels_by_area_and_radius], 
        prompt="""
            You are a hotel search expert. 
            For each hotel, return the following fields in HTML format:

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
            
            If the user does not mention check-in and check-out dates, assume:
            - arrival_date = today's date
            - departure_date = tomorrow's date
            
            `latitude` and `longitude` must always be included.
            Do not return bbox, display_name, or type in the final output. Just output a clean HTML <ul> list showing all above fields.
        """
    )
    
    result = await agent.ainvoke(state)
    
    return Command(update={"messages": [HumanMessage(content=result["messages"][-1].content, name="hotel_search_expert")]}, goto="supervisor")


async def weather_agent(state: State) -> Command[Literal["supervisor"]]:
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
        
    result = await agent.ainvoke(state)
    
    return Command(update={"messages": [HumanMessage(content=result["messages"][-1].content, name="weather_expert")]}, goto="supervisor")

async def place_search_agent(state: State) -> Command[Literal["supervisor"]]:
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
    result = await agent.ainvoke(state)
    return Command(update={"messages": [HumanMessage(content=result["messages"][-1].content, name="place_search_expert")]}, goto="supervisor")

async def flight_fares_search_agent(state: State) -> Command[Literal["supervisor"]]:
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
    result = await agent.ainvoke(state)
    return Command(update={"messages": [HumanMessage(content=result["messages"][-1].content, name="flight_fares_search_expert")]}, goto="supervisor")