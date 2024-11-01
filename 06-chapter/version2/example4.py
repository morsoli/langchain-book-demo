from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# 定义状态
class State(TypedDict):
    input: str
    priority: str
    result: str

# 定义节点
def check_priority(state: dict):
    # 判断优先级
    if "urgent" in state["input"].lower():
        return {"priority": "high"}
    return {"priority": "normal"}

def handle_high_priority(state: dict):
    return {"result": "紧急处理完成"}

def handle_normal_priority(state: dict):
    return {"result": "常规处理完成"}

# 路由函数
def route_by_priority(state: dict):
    return "high_handler" if state["priority"] == "high" else "normal_handler"

# 构建图
builder = StateGraph(State)

# 添加节点
builder.add_node("priority_check", check_priority)
builder.add_node("high_handler", handle_high_priority)
builder.add_node("normal_handler", handle_normal_priority)

# 添加边
builder.add_edge(START, "priority_check")
builder.add_conditional_edges(
    "priority_check", 
    route_by_priority
)
builder.add_edge("high_handler", END)
builder.add_edge("normal_handler", END)

# 编译和使用
graph = builder.compile()
result = graph.invoke({"input": "urgent task"})
print(result)
print(graph.get_graph().draw_mermaid())