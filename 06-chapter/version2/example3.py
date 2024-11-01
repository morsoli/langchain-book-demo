from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

# 定义状态
class State(TypedDict):
    input: str
    result: str

# 定义节点
def validate_input(state: dict):
    return {"input": state["input"].strip()}

def process_data(state: dict, config: RunnableConfig):
    user_id = config["configurable"]["user_id"]
    return {"result": f"用户 {user_id} 的输入 '{state['input']}' 已处理"}

# 构建图
builder = StateGraph(State)

# 添加节点
builder.add_node("validate", validate_input)
builder.add_node("process", process_data)

# 添加边
builder.add_edge(START, "validate")
builder.add_edge("validate", "process")
builder.add_edge("process", END)

# 编译和使用
graph = builder.compile()
result = graph.invoke(
    {"input": "测试数据"},
    {"configurable": {"user_id": "123"}}
)
print(result)
print(graph.get_graph().draw_mermaid())