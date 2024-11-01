from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# 父图和子图共享状态键
class ParentState(TypedDict):
    shared_data: str
    parent_data: str

class SubState(TypedDict):
    shared_data: str  # 共享键
    sub_data: str

# 定义子图
def process_in_sub(state: SubState):
    return {"shared_data": f"处理: {state['shared_data']}"}

sub_builder = StateGraph(SubState)
sub_builder.add_node("process", process_in_sub)
sub_builder.add_edge(START, "process")
sub_builder.add_edge("process", END)
sub_graph = sub_builder.compile()

# 在父图中使用子图
parent_builder = StateGraph(ParentState)
parent_builder.add_node("sub_process", sub_graph)  # 直接添加编译后的子图
parent_builder.add_edge(START, "sub_process")
parent_builder.add_edge("sub_process", END)
parent_graph = parent_builder.compile()

print(parent_graph.get_graph().draw_mermaid())
