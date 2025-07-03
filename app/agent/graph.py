from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from app.agent.llm_setup import call_llm_with_tool_bind, tools


workflow = StateGraph(MessagesState)
workflow.add_node("llm_with_tools", call_llm_with_tool_bind)
workflow.add_node("tools", ToolNode(tools))
workflow.add_edge(START, "llm_with_tools")
workflow.add_conditional_edges("llm_with_tools", tools_condition)
workflow.add_edge("tools", "llm_with_tools")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
