from langchain_core.messages import HumanMessage;
from langchain_google_genai import ChatGoogleGenerativeAI;
from langgraph.graph import END, START, StateGraph
from typing import Dict, TypedDict;
from dotenv import load_dotenv;
import os;
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class AgentState(TypedDict):
    messages : list[HumanMessage];
    
def process(state: AgentState):
    """ send data to the ai model and get a response """
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state
    
    
graph = StateGraph(AgentState);
graph.add_node("process", process);
graph.add_edge(START, "process");
graph.add_edge("process", END);
agent = graph.compile();


user = input("Enter: ");
agent.invoke(
    {
        "messages" : [HumanMessage(content=user)]
    }
)
