from langgraph.graph import StateGraph, START, END  
from langgraph.checkpoint.memory import MemorySaver
from app.agent.router import State
from app.agent.agent import *

# Define the graph
graph = StateGraph(State)

# Add nodes
graph.add_node("supervisor", supervisor)
graph.add_node("hotel_search_expert", hotel_search_agent)
graph.add_node("weather_expert", weather_agent)
graph.add_node("place_search_expert", place_search_agent)
graph.add_node("flight_fares_search_expert", flight_fares_search_agent)
graph.add_node("geolocation_expert", geolocation_agent)

# Initial entry point
graph.add_edge(START, "supervisor")

# Return control to supervisor after each expert agent runs
graph.add_edge("hotel_search_expert", "supervisor")
graph.add_edge("weather_expert", "supervisor")
graph.add_edge("place_search_expert", "supervisor")
graph.add_edge("flight_fares_search_expert", "supervisor")
graph.add_edge("geolocation_expert", "supervisor")

# ✅ Add this missing edge so that "FINISH" works correctly
graph.add_edge("supervisor", END)

# Compile graph
memory = MemorySaver()
app = graph.compile(checkpointer=memory)
