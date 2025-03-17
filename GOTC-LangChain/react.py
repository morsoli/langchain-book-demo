# 导入必要的库和模块
import os
import operator
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, BaseMessage
from langgraph.prebuilt import ToolInvocation, ToolExecutor
from langchain_openai import ChatOpenAI

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_sk_xxx"
os.environ["LANGCHAIN_PROJECT"] = "langgraph-ReAct"


# 设置API密钥
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

# 定义天气查询工具
@tool
def get_weather(query: str):
    """调用搜索功能"""
    # 简单的天气查询逻辑
    if "上海" in query or "上海市" in query.lower():
        return "今天上海多云，气温 30 摄氏度左右"
    return "暂未查询到，重新再试"

# 设置工具执行器
tools = [get_weather]
tool_executor = ToolExecutor(tools)

# 初始化语言模型
model = ChatOpenAI(model="glm-4-air", base_url="https://open.bigmodel.cn/api/paas/v4/", api_key=ZHIPU_API_KEY).bind_tools(tools)
model = model.bind_tools(tools)

# 定义决策函数：判断是否继续执行
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # 如果最后一条消息没有工具调用，则结束
    if not last_message.tool_calls:
        return "end"
    # 否则继续执行
    else:
        return "continue"

# 定义推理节点：模型调用函数
def call_model(state):
    messages = state["messages"]
    # 调用模型生成响应
    response = model.invoke(messages)
    # 返回新的消息列表，这将被添加到现有列表中
    return {"messages": [response]}

# 定义行动节点：工具调用函数
def call_tool(state):
    messages = state["messages"]
    # 获取最后一条消息中的工具调用
    last_message = messages[-1]
    tool_call = last_message.tool_calls[0]
    # 构造工具调用对象
    action = ToolInvocation(
        tool=tool_call["name"],
        tool_input=tool_call["args"],
    )
    # 执行工具调用并获取响应
    response = tool_executor.invoke(action)
    # 创建工具消息
    function_message = ToolMessage(
        content=str(response), name=action.tool, tool_call_id=tool_call["id"]
    )
    # 返回新的消息列表
    return {"messages": [function_message]}


from langgraph.graph import END, StateGraph, START

# 定义全局状态，这个状态将在工作流中传递，在节点间中共享
class AgentState(TypedDict):
    # 状态属性 messages
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
# 定义工作流（状态图）
workflow = StateGraph(AgentState)

# 添加两个主要节点：agent和action
workflow.add_node("reasoning", call_model)  # reasoning 推理节点负责调用语言模型
workflow.add_node("action", call_tool)  # action 行动节点负责调用工具

# 定义决策函数：判断是否继续执行
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # 如果最后一条消息没有工具调用，则结束
    if not last_message.tool_calls:
        return "end"
    # 否则继续执行
    else:
        return "continue"
    
# 设置工作流的入口点为推理（reasoning）节点
workflow.add_edge(START, "reasoning")
# 添加从action到reasoning的普通边
workflow.add_edge("action", "reasoning")
# 添加条件边
workflow.add_conditional_edges(
    # 起始节点为reasoning
    "reasoning",
    # 使用should_continue函数决定下一步操作
    should_continue,
    # 定义条件分支
    {
        # 如果返回"continue"，则执行action节点
        "continue": "action",
        # 如果返回"end"，则结束工作流
        "end": END,
    },
)
# 将工作流编译成可执行的LangChain Runnable对象
app = workflow.compile()
inputs = {"messages": [HumanMessage(content="上海今天天气怎么样")]}
app.invoke(inputs)

for s in app.stream(inputs):
    print(list(s.values())[0])
    print("="*10)