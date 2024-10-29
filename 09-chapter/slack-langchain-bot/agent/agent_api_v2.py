import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.chat_models.tongyi import ChatTongyi
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from tools_v2 import generate_image, generate_voice


tools = [generate_voice, generate_image, TavilySearchResults(max_results=1)]

# 处理对话并使用LLM生成回应
def agent(state: MessagesState) -> MessagesState:
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "与人类对话，尽可能回答问题，你可以使用工具"
        ),
        ("placeholder", "{messages}"),
    ])
    model = ChatTongyi(temperature=0)
    model_with_tools = model.bind_tools(tools)
    bound = prompt | model_with_tools
    prediction = bound.invoke({
        "messages": state["messages"]
    })
    return {
        "messages": [prediction],
    }
# 根据最后一条消息决定下一步操作
def route_tools(state: MessagesState):
    msg = state["messages"][-1]
    # 如果消息需要调用工具,返回"tool"
    if msg.tool_calls:
        return "tools"
    
    # 否则结束当前对话轮次
    return END


# 初始化状态图
builder = StateGraph(MessagesState)  

# 添加节点
builder.add_node(agent)
builder.add_node("tools", ToolNode(tools))  


# START -> agent 进入代理处理
builder.add_edge(START, "agent")

# agent -> [tools, END]: 代理处理后根据route_tools函数的返回结果选择路径
builder.add_conditional_edges(
    "agent",        # 源节点
    route_tools,    # 路由决策函数
    ["tools", END]  # 可能的目标节点
)

# tools -> agent: 工具调用完成后返回代理处理
builder.add_edge("tools", "agent")

graph = builder.compile()


def langchain_agent(user, query):
    config = {"configurable": {"user_id": user, "thread_id": "1"}}
    response = graph.invoke({"messages": [("user", query)]}, 
            config=config)
    return response["messages"][-1].content

if __name__ == "__main__":
    
    while True:
        user_input = input("人类: ")
        # 处理用户输入
        user_query = re.sub(r"^人类: ", "", user_input)
        # 运行代理
        print(f"AI: {langchain_agent("1", user_query)}")
        print("=" * 30)