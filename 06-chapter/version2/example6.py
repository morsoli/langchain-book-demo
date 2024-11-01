from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END


# 父图和子图使用不同的状态模式
class ParentState(TypedDict):
    input_data: str
    result: str

class SubState(TypedDict):
    sub_input: str  # 完全不同的键
    sub_output: str

# 定义子图
def sub_process(state: SubState):
    return {"sub_output": f"子图处理: {state['sub_input']}"}

sub_builder = StateGraph(SubState)
sub_builder.add_node("process", sub_process)
sub_graph = sub_builder.compile()

# 创建调用子图的函数
def call_subgraph(state: ParentState):
    # 转换状态到子图格式
    sub_result = sub_graph.invoke({
        "sub_input": state["input_data"]
    })
    # 转换结果回父图格式
    return {"result": sub_result["sub_output"]}

# 在父图中使用子图
parent_builder = StateGraph(ParentState)
parent_builder.add_node("sub_process", call_subgraph)