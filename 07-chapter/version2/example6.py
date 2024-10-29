from typing import List
import uuid
from langchain_core.documents import Document 
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

# 初始化向量存储,用于存储记忆
recall_vector_store = InMemoryVectorStore(DashScopeEmbeddings())

# 记忆存储和检索的核心工具函数
def get_user_id(config: RunnableConfig) -> str:
    """获取用户ID
    Args:
        config: 运行时配置
    Returns:
        str: 用户ID
    Raises:
        ValueError: 如果未提供用户ID
    """
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("需要提供用户ID来保存记忆")
    return user_id

@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """保存记忆到向量存储以供后续语义检索
    Args:
        memory: 要保存的记忆内容
        config: 运行时配置
    Returns:
        str: 保存的记忆内容
    """
    user_id = get_user_id(config)
    document = Document(
        page_content=memory, 
        id=str(uuid.uuid4()), 
        metadata={"user_id": user_id}
    )
    recall_vector_store.add_documents([document])
    return memory

@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """搜索相关记忆
    Args:
        query: 搜索查询
        config: 运行时配置
    Returns:
        List[str]: 检索到的记忆列表
    """
    user_id = get_user_id(config)
    def _filter_function(doc: Document) -> bool:
        return doc.metadata.get("user_id") == user_id
    documents = recall_vector_store.similarity_search(
        query, k=3, filter=_filter_function
    )
    return [document.page_content for document in documents]

tools = [save_recall_memory, search_recall_memories, TavilySearchResults(max_results=1)]
model = ChatTongyi()
# 系统提示词模板
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个具有长期记忆能力的助手。你需要依赖外部记忆来在对话之间存储信息，请使用可用的记忆工具来存储和检索\n\n"
        "记忆使用指南:\n"
        "1. 积极使用记忆工具来建立对用户的全面理解\n"
        "2. 根据储存的记忆做出推论\n" 
        "3. 定期回顾过往互动以识别用户的偏好\n"
        "4. 根据每个新信息更新对用户的认知模型\n"
        "5. 交叉对照新旧信息以保持一致性\n"
        "6. 利用记忆认识并确认用户情况或观点的变化\n"
        "7. 运用记忆提供个性化的例子和类比\n"
        "8. 回顾过往经验来指导当前问题解决\n\n"
        
        "## 记忆回溯\n"
        "基于当前对话上下文检索的记忆:\n{recall_memories}\n\n"
        
        "## 使用说明\n"
        "与用户交流时无需特意提及你的记忆能力。"
        "要将对用户的理解自然地融入回应中。"
        "使用工具保存想在下次对话中保留的信息。"
        "如果调用工具，工具调用前的文本是内部消息。在工具调用成功确认后再回应。\n\n"
    ),
    ("placeholder", "{messages}"),
])


# 用于存储对话相关的记忆
class State(MessagesState):
    recall_memories: List[str]

# 处理当前状态并使用LLM生成回应
def agent(state: State) -> State:
    """
    Args:
        state: 当前对话状态
    Returns:
        State: 更新后的状态和代理回应
    """
    model_with_tools = model.bind_tools(tools)
    bound = prompt | model_with_tools
    recall_str = (
        "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
    )
    prediction = bound.invoke({
        "messages": state["messages"],
        "recall_memories": recall_str,
    })
    return {
        "messages": [prediction],
    }
    

def load_memories(state: State, config: RunnableConfig) -> State:
    """加载当前对话相关的记忆
    
    Args:
        state: 当前对话状态
        config: 运行时配置,包含用户ID等信息
        
    Returns:
        State: 包含加载记忆的更新状态
        
    工作流程:
    1. 获取当前对话内容
    2. 将对话内容截断到800个字符
    3. 基于对话内容检索相关记忆
    """
    # 获取当前对话内容字符串
    convo_str = get_buffer_string(state["messages"])
    
    # 限制token数量,避免超出模型上下文限制
    convo_str = convo_str[:800]
    
    # 检索相关记忆
    recall_memories = search_recall_memories.invoke(convo_str, config)
    
    return {
        "recall_memories": recall_memories,
    }

def route_tools(state: State):
    """根据最后一条消息决定下一步操作
    
    Args:
        state: 当前对话状态
        
    Returns:
        Literal["tools", "__end__"]: 返回下一步是使用工具还是结束对话
        
    工作流程:
    1. 获取最后一条消息
    2. 检查是否需要调用工具
    3. 根据检查结果决定路由
    """
    # 获取最后一条消息
    msg = state["messages"][-1]
    
    # 如果消息需要调用工具,返回"tools"
    if msg.tool_calls:
        return "tools"
    
    # 否则结束当前对话轮次
    return END

#---------------------------- 构建对话流程图 ----------------------------#

# 1. 创建状态图并添加节点
builder = StateGraph(State)  # 初始化状态图构建器

# 添加三个核心节点:
builder.add_node(load_memories)      # 记忆加载节点
builder.add_node(agent)              # 代理处理节点
builder.add_node("tools", ToolNode(tools))  # 工具调用节点

# 2. 设置图的边(定义节点间的连接关系)
# START -> load_memories: 对话开始时首先加载记忆
builder.add_edge(START, "load_memories")

# load_memories -> agent: 加载记忆后进入代理处理
builder.add_edge("load_memories", "agent")

# agent -> [tools, END]: 代理处理后根据route_tools函数的返回结果选择路径
builder.add_conditional_edges(
    "agent",        # 源节点
    route_tools,    # 路由决策函数
    ["tools", END]  # 可能的目标节点
)

# tools -> agent: 工具调用完成后返回代理处理
builder.add_edge("tools", "agent")

# 3. 编译图并设置检查点
memory = MemorySaver()  # 创建记忆保存器
graph = builder.compile(checkpointer=memory)  # 编译图并启用检查点功能

#---------------------------- 图的处理流程 ----------------------------#
# 1. START -> load_memories: 开始对话时加载相关记忆
# 2. load_memories -> agent: 将加载的记忆传给代理
# 3. agent根据route_tools的判断:
#    - 如果需要使用工具: agent -> tools -> agent
#    - 如果不需要使用工具: agent -> END
# 4. 整个过程中通过memory保存状态


def get_stream_chunk(chunk):
   """
   打印对话流中的数据
   
   Args:
       chunk: 包含节点更新信息的数据块
           格式: {
               节点名: {
                   "messages": [...] 或其他更新数据
               }
           }
   
   工作流程:
   1. 遍历数据块中的所有节点更新
   2. 针对消息类型和非消息类型采用不同的打印格式
   3. 在内容之间添加换行符提高可读性
   """
   for _, updates in chunk.items():
       # 判断是否包含消息类型的更新
       if "messages" in updates:
           # 如果是消息,使用pretty_print()方法美化打印
           updates["messages"][-1].pretty_print()
       else:
           # 如果是其他类型的更新,直接打印
           print(updates)
           
       # 添加空行分隔不同节点的输出
       print("\n")

# 示例对话部分       
config = {"configurable": {"user_id": "1", "thread_id": "1"}}

# 第一轮对话: 告知兴趣爱好    
for chunk in graph.stream(
    {"messages": [("user", "莫尔索爱好写作")]}, 
    config=config
):
    get_stream_chunk(chunk)

# 第二轮对话: 告知正在做的事
for chunk in graph.stream(
    {"messages": [("user", "莫尔索正在写关于 LangGraph 的技术文章")]},
    config=config,
):
    get_stream_chunk(chunk)

#第三轮对话:询问问题挑战
for chunk in graph.stream(
    {"messages": [("user", "提供写作帮助")]},
    config={"configurable": {"user_id": "1", "thread_id": "2"}},
):
    get_stream_chunk(chunk)

    """==== Ai Message ====

好的，现在我记住了莫尔索喜欢写作这一点。这是一个很有趣的信息！写作是一种非常棒的表达自我和探索内心世界的方式。如果莫尔索正在寻找写作灵感，或者想要提升写作技巧，我能提供一些建议或资源。你是否有具体的问题或需要哪方面的帮助呢？例如，他可能需要一些创意激发的方法，或者是关于如何开始写作项目的建议？
Tool Calls:
  save_recall_memory (call_f5c8c1e8879b45078a8f63)
 Call ID: call_f5c8c1e8879b45078a8f63
  Args:
    memory: 莫尔索喜欢写作


==== Tool Message ====
Name: save_recall_memory

莫尔索喜欢写作


==== Ai Message ====

我已经记住了莫尔索喜欢写作这一点。如果有任何关于写作的问题或需要进一步的帮助，随时告诉我！无论是寻找灵感、提高写作技巧还是完成某个项目，我都可以提供支持和建议。你是否希望我分享一些写作资源或提示呢？


{'recall_memories': ['莫尔索喜欢写作']}


==== Ai Message ====

太好了！莫尔索正在撰写关于LangGraph的技术文章。这听起来像是一个既专业又有趣的主题。为了帮助莫尔索更好地完成这篇文章，我可以提供一些写作建议和技术资料的搜索方向。

首先，我们可以考虑以下几个方面：
1. **明确目标读者**：了解目标读者是谁可以帮助确定文章的深度和语言风格。
2. **结构规划**：确保文章有一个清晰的引言、主体和结论。可以包括背景介绍、核心概念解释、案例研究或示例等部分。
3. **技术准确性**：确保所有技术术语和概念都是准确无误的，并且尽可能引用可靠的来源。
4. **可读性**：使用图表、子标题和列表来提高文章的可读性和吸引力。

如果需要更具体的帮助，比如查找有关LangGraph的相关技术资料或参考资料，请告诉我，我可以帮您搜索相关信息。
Tool Calls:
  save_recall_memory (call_b191d25f7ba44232b17840)
 Call ID: call_b191d25f7ba44232b17840
  Args:
    memory: 莫尔索正在写关于LangGraph的技术文章


==== Tool Message ====
Name: save_recall_memory

莫尔索正在写关于LangGraph的技术文章


==== Ai Message ====

我已经记住了莫尔索正在写关于LangGraph的技术文章这一信息。如果您需要任何关于技术文章写作的具体帮助，例如查找相关资料、构建文章结构或提升文章的可读性，请随时告诉我。我在这里为您提供支持！

此外，如果莫尔索有任何特定的技术问题或需要进一步澄清的概念，也可以告诉我，我会尽力提供帮助。希望这些建议能对莫尔索有所帮助！


{'recall_memories': ['莫尔索喜欢写作', '莫尔索正在写关于LangGraph的技术文章']}


==== Ai Message ====

当然可以！我了解到您正在撰写一篇关于LangGraph的技术文章，我可以帮您快速搜索一些关于LangGraph的信息，以便为莫尔索的技术文章提供更多的参考材料。

稍等片刻，我会尽快找到相关信息。
Tool Calls:
  tavily_search_results_json (call_9bc00ab9de824556b5a532)
 Call ID: call_9bc00ab9de824556b5a532
  Args:
    query: LangGraph


==== Tool Message ====
Name: tavily_search_results_json

[{"url": "https://github.com/langchain-ai/langgraph", "content": "LangGraph is a library for creating stateful, multi-actor applications with LLMs, using cycles, controllability, and persistence. Learn how to use LangGraph with LangChain, LangSmith, and Anthropic tools to build agent and multi-agent workflows."}]


==== Ai Message ====

我找到了一些关于LangGraph的信息。LangGraph似乎是一个用于创建包含大型语言模型（LLMs）的多参与者应用程序的库。它支持状态管理、可控性和持久性，并且可以与LangChain、LangSmith和Anthropic等工具一起使用来构建代理和多代理工作流。

这些信息可能有助于增加文章的技术深度和广度。如果莫尔索需要更多详细的技术规格、实际应用案例或其他相关信息，请随时告诉我，我会尽力提供更多帮助。


{'recall_memories': []}
    """