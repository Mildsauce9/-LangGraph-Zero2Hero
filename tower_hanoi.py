from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, ToolMessage;
from langchain_google_genai import ChatGoogleGenerativeAI; # type: ignore
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from typing_extensions import Dict, TypedDict, Annotated, Sequence, Union;
from dotenv import load_dotenv;
load_dotenv()

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]
    tower_1 : list[int]
    tower_2 : list[int]
    tower_3 : list[int]

# We'll create a custom tool node that has access to state
class StatefulToolNode:
    def __init__(self, tools):
        self.tools = {tool.name: tool for tool in tools}
    
    def __call__(self, state: AgentState):
        # Get the last message which should contain tool calls
        last_message = state["messages"][-1]
        tool_calls = last_message.tool_calls
        
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if tool_name in self.tools:
                # Pass the current state to the tool
                if tool_name == "move":
                    result = self.move_with_state(state, **tool_args)
                elif tool_name == "peek":
                    result = self.peek_with_state(state, **tool_args)
                elif tool_name == "get_all_towers":
                    result = self.get_all_towers_with_state(state)
                else:
                    result = f"Unknown tool: {tool_name}"
                
                results.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                ))
        
        return {"messages": results}
    
    def move_with_state(self, state: AgentState, source: str, destination: str) -> str:
        """Move a disk from source to destination tower with state validation"""
        # Validate tower names
        valid_towers = ["tower_1", "tower_2", "tower_3"]
        if source not in valid_towers or destination not in valid_towers:
            return f"Invalid tower name. Use tower_1, tower_2, or tower_3"
        
        # Check if source tower has disks
        if not state[source]:
            return f"Cannot move from {source}: tower is empty"
        
        # Get the top disk from source
        disk = state[source][-1]
        
        # Check if move is valid (can't place larger disk on smaller one)
        if state[destination] and state[destination][-1] < disk:
            return f"Invalid move: Cannot place disk {disk} on top of disk {state[destination][-1]}"
        
        # Perform the move by updating state
        moved_disk = state[source].pop()
        state[destination].append(moved_disk)
        
        return f"Moved disk {moved_disk} from {source} to {destination}. New state: tower_1={state['tower_1']}, tower_2={state['tower_2']}, tower_3={state['tower_3']}"
    
    def peek_with_state(self, state: AgentState, tower: str) -> str:
        """Check the current state of a tower"""
        valid_towers = ["tower_1", "tower_2", "tower_3"]
        if tower not in valid_towers:
            return f"Invalid tower name. Use tower_1, tower_2, or tower_3"
        
        if not state[tower]:
            return f"{tower} is empty"
        
        return f"{tower} contains disks: {state[tower]} (bottom to top)"
    
    def get_all_towers_with_state(self, state: AgentState) -> str:
        """Get the current state of all towers"""
        return f"Current game state - tower_1: {state['tower_1']}, tower_2: {state['tower_2']}, tower_3: {state['tower_3']}"

# Define tools for the LLM to understand available functions
@tool
def move(source: str, destination: str) -> str:
    """ Use this method to move a disk from one tower to another, mention the towers as \"tower_1\" or \"tower_2\" or \"tower_3\" """
    pass  # Implementation handled by StatefulToolNode

@tool
def peek(tower: str) -> str:
    """ You can see the current state of any tower simply parse the tower name as \"tower_1\" or \"tower_2\" or \"tower_3\" """
    pass  # Implementation handled by StatefulToolNode

@tool
def get_all_towers() -> str:
    """ Get the current state of all towers """
    pass  # Implementation handled by StatefulToolNode

tools = [move, peek, get_all_towers]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"];
    last_message = messages[-1];
    if not last_message.tool_calls:
        return "end";
    return "continue";

# Create our custom stateful tool node
stateful_tool_node = StatefulToolNode(tools)

graph = StateGraph(AgentState);
graph.add_node("model", model_call);
graph.add_node("tools", stateful_tool_node);
graph.add_conditional_edges(
    "model", 
    should_continue, 
    {
        "continue" : "tools", 
        "end" : END
    }, 
    
)
graph.add_edge("tools", "model");
graph.add_edge(START, "model");

app = graph.compile();
from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
            
inputs = {
    "messages": [HumanMessage(content="You are playing the Tower of Hanoi game. There are 3 towers: tower_1, tower_2, and tower_3. Initially tower_1 has 3 disks [3, 2, 1] where 3 is the largest (bottom) and 1 is the smallest (top). Your goal is to move all disks from tower_1 to tower_3. Rules: 1) You can only move one disk at a time (always the top disk). 2) You cannot place a larger disk on top of a smaller disk. Use the available tools to check the current state and make moves. Start by checking the current state!")],
    "tower_1": [3, 2, 1],
    "tower_2": [],
    "tower_3": []
}

print_stream(app.stream(inputs, stream_mode="values"))