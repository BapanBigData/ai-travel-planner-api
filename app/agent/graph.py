from langgraph.graph import StateGraph, START, END
# from langgraph.checkpoint.memory import MemorySaver
from app.agent.router import State
from app.agent.agent import *

graph = StateGraph(State)

graph.add_node("supervisor", supervisor)
graph.add_node("hotel_search_expert", hotel_search_agent)
graph.add_node("weather_expert", weather_agent)
graph.add_node("place_search_expert", place_search_agent)
graph.add_node("flight_fares_search_expert", flight_fares_search_agent)

graph.add_edge(START, "supervisor")
graph.add_edge("hotel_search_expert", "supervisor")
graph.add_edge("weather_expert", "supervisor")
graph.add_edge("place_search_expert", "supervisor")
graph.add_edge("flight_fares_search_expert", "supervisor")
graph.add_edge("supervisor", END)

# memory = MemorySaver()
app = graph.compile()
