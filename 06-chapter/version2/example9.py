from typing import Annotated, Literal, Any
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode


# 初始化数据库连接
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
# 初始化数据库工具
db_tool = SQLDatabaseToolkit(db=db, llm=ChatDeepSeek(model="deepseek-chat"))
tools = db_tool.get_tools()


# 获取特定工具
list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")


@tool
def execute_db_query(query: str) -> str:
    """
    执行数据库SQL查询并返回结果。
    如果查询不正确，则返回错误消息。
    如果返回错误消息，则重写查询，检查查询，并重试。
    """
    print("="*10, query)
    result = db.run_no_throw(query)
    if not result:
        return "错误：查询失败。请重写您的查询并重试。"
    return result

# SQL查询检查系统描述
query_check_system = """您是一位注重细节的SQL专家。
请对SQLite查询进行双重检查，以发现常见错误，包括：
- 使用NOT IN与NULL值
- 当应该使用UNION ALL时使用了UNION
- 使用BETWEEN表示不包含的范围
- 谓词中的数据类型不匹配
- 正确引用标识符
- 为函数使用正确数量的参数
- 转换为正确的数据类型
- 使用正确的列进行连接

如果发现上述错误，请重写查询。如果没有错误，请仅复制原始查询。

在运行此检查后，您将调用适当的工具来执行查询。"""

# SQL查询检查提示模板
query_check_prompt = ChatPromptTemplate.from_messages(
    [("system", query_check_system), ("placeholder", "{messages}")]
)
query_check = query_check_prompt | ChatDeepSeek(model="deepseek-chat").bind_tools([execute_db_query])

# print("="*10, query_check.invoke({"messages": [("user", "SELECT * FROM Artist LIMIT 10;")]}))

# 定义代理的状态
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# 定义一个新的工作流图
workflow_graph = StateGraph(AgentState)

# 添加第一个工具调用节点
def first_tool_call(state: AgentState) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }

# 添加模型检查查询节点
def model_check_query(state: AgentState) -> dict[str, list[AIMessage]]:
    """
    使用此工具在执行查询之前双重检查您的查询是否正确。
    """
    print(state["messages"])
    print(query_check.invoke({"messages": [state["messages"][-1]]}))
    return {"messages": [query_check.invoke({"messages": [state["messages"][-1]]})]}


def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    创建一个带有错误处理备选方案的工具节点，以便处理错误并将它们呈现给代理。
    
    参数:
    tools -- 工具列表，包含一个或多个工具。
    
    返回:
    一个配置有错误处理备选方案的工具节点。
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def handle_tool_error(state: dict) -> dict:
    """
    处理工具执行过程中发生的错误。
    
    参数:
    state -- 当前状态，包含错误信息和工具调用信息。
    
    返回:
    包含错误消息的工具消息列表。
    """
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


# 添加选择相关表格的节点
model_get_schema = ChatDeepSeek(model="deepseek-chat").bind_tools(
    [get_schema_tool]
)

workflow_graph.add_node(
    "model_get_schema",
    lambda state: {
        "messages": [model_get_schema.invoke(state["messages"])]
    },
)

# 定义提交最终答案的工具
class SubmitFinalAnswer(BaseModel):
    """根据查询结果向用户提交最终答案。"""

    final_answer: str = Field(..., description="向用户的最终答案")

# 添加模型生成查询的节点
query_gen_system = """您是一位注重细节的SQL专家。

给定一个输入问题，输出一个语法正确的SQLite查询来运行，然后查看查询结果并返回答案。

不要调用除了SubmitFinalAnswer之外的任何工具来提交最终答案。

在生成查询时：

输出回答输入问题而不带工具调用的SQL查询。

除非用户指定他们希望获得特定数量的示例，否则始终将查询限制为最多5个结果。
您可以按相关列对结果进行排序，以返回数据库中最有趣的示例。
永远不要查询特定表的所有列，只根据问题询问相关列。

如果您在执行查询时遇到错误，请重写查询并重试。

如果您得到一个空的结果集，您应该尝试重写查询以获得非空的结果集。
如果您没有足够的信息回答问题...请不要编造信息，只说您没有足够的信息。

如果您有足够的信息回答问题，只需调用适当的工具向用户提交最终答案。

不要对数据库进行任何DML语句（INSERT, UPDATE, DELETE, DROP等）。"""


query_gen_prompt = ChatPromptTemplate.from_messages(
    [("system", query_gen_system), ("placeholder", "{messages}")]
)
query_gen = query_gen_prompt | ChatDeepSeek(model="deepseek-chat").bind_tools(
    [SubmitFinalAnswer]
)

# 添加模型生成查询的节点
def query_gen_node(state: AgentState):
    message = query_gen.invoke(state)

    # 有时，LLM会幻想并调用错误的工具。我们需要捕获这一点并返回错误消息。
    tool_messages = []
    if message.tool_calls:
        for tc in message.tool_calls:
            if tc["name"] != "SubmitFinalAnswer":
                tool_messages.append(
                    AIMessage(
                        content=f"错误：调用了错误的工具：{tc['name']}。请修正您的错误。记得只调用SubmitFinalAnswer来提交最终答案。生成的查询应该不带工具调用输出。",
                        tool_call_id=tc["id"],
                    )
                )
    else:
        tool_messages = []
    return {"messages": [message] + tool_messages}

# 定义条件边以决定是否继续或结束工作流
def should_continue(state: AgentState) -> Literal[END, "correct_query", "query_gen"]: # type: ignore
    messages = state["messages"]
    last_message = messages[-1]
    # 如果有工具调用，则结束
    if getattr(last_message, "tool_calls", None):
        return END
    if last_message.content.startswith("错误："):
        return "query_gen"
    else:
        return "correct_query"

# 添加节点到工作流图
workflow_graph.add_node("first_tool_call", first_tool_call)

# 添加前两个工具的节点
workflow_graph.add_node(
    "list_tables_tool", create_tool_node_with_fallback([list_tables_tool])
)
workflow_graph.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))


workflow_graph.add_node("query_gen", query_gen_node)

# 添加模型检查查询的节点
workflow_graph.add_node("correct_query", model_check_query)

# 添加执行查询的节点
workflow_graph.add_node("execute_query", create_tool_node_with_fallback([execute_db_query]))

# 指定节点之间的边
workflow_graph.add_edge(START, "first_tool_call")
workflow_graph.add_edge("first_tool_call", "list_tables_tool")
workflow_graph.add_edge("list_tables_tool", "model_get_schema")
workflow_graph.add_edge("model_get_schema", "get_schema_tool")
workflow_graph.add_edge("get_schema_tool", "query_gen")
workflow_graph.add_conditional_edges(
    "query_gen",
    should_continue,
)
workflow_graph.add_edge("correct_query", "execute_query")
workflow_graph.add_edge("execute_query", "query_gen")

# 将工作流编译为可运行的应用
app = workflow_graph.compile()
print(app.get_graph().draw_mermaid())

messages = app.invoke(
    {"messages": [("user", "Which sales agent made the most in sales in 2009?")]}
)
json_str = messages["messages"][-1].tool_calls[0]["args"]["final_answer"]
print(json_str)
