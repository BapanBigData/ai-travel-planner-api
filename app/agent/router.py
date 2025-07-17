from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import MessagesState


class Router(TypedDict):
    next: Literal["hotel_search_expert", "weather_expert", "place_search_expert", "flight_fares_search_expert", "geolocation_expert", 'FINISH']

class State(MessagesState):
    next:str