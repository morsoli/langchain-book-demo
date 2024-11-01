from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START

# 输入状态
class InputState(TypedDict):
    user_input: str

# 输出状态    
class OutputState(TypedDict):
    result: str


# 内部状态(包含所有必要字段)
class InternalState(TypedDict):
    user_input: str
    intermediate_data: dict
    result: str


# 私有状态(用于节点间特定通信)
class PrivateState(TypedDict):
    temp_data: dict
    

# 定义节点处理函数
def process_input(state: InputState) -> InternalState:
    # 处理输入并返回更新
    return {
        "user_input": state["user_input"],
        "intermediate_data": {"processed": True}
    }

def generate_result(state: InternalState) -> OutputState:
    # 生成最终结果
    return {
        "result": f"Processed: {state}"
    }

# 构建图
builder = StateGraph(InternalState, 
                    input=InputState,
                    output=OutputState)

# 添加节点和边
builder.add_node("process", process_input)
builder.add_node("generate", generate_result)
builder.add_edge(START, "process")
builder.add_edge("process", "generate")
builder.add_edge("generate", END)

# 编译图
graph = builder.compile()

# 调用图
result = graph.invoke({"user_input": "Hello"})

print(result)