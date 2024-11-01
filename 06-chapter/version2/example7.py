from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]

def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}


builder = StateGraph(State)
builder.add_node(node_a)
builder.add_node(node_b)
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
graph.invoke({"foo": ""}, config)


# state_history = list(graph.get_state_history(config))
# for snapshot in state_history[::-1]:
#     print(snapshot)

# 获取指定检查点的状态快照 
config = {"configurable": {"thread_id": "1", "checkpoint_id": "1ef96c6f-4a85-6546-8001-2a071283a5d9"}}
checkpoint_snapshot = graph.get_state(config)
print(checkpoint_snapshot)

# 从检查点重放图执行
config = {"configurable": {"thread_id": "1", "checkpoint_id": "1ef96c6f-4a85-6546-8001-2a071283a5d9"}}
print(graph.invoke({"foo": ""}, config))