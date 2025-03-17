from typing import List
import uuid
from langchain_core.documents import Document 
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_deepseek import ChatDeepSeek
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

from typing_extensions import TypedDict

class KnowledgeTriple(TypedDict):
    """知识三元组数据结构
    
    字段说明:
    - subject: 主体/主语
    - predicate: 谓语/关系
    - object_: 宾语/客体
    """
    subject: str     # 主体，例如:"小明"
    predicate: str   # 谓语，例如:"喜欢"
    object_: str     # 客体，例如:"篮球"

@tool
def save_recall_memory(memories: List[KnowledgeTriple], config: RunnableConfig) -> str:
    """将记忆存入向量数据库以支持语义化召回
    参数说明:
    memories: 知识三元组列表，每个三元组包含主谓宾结构的记忆片段
    config: 运行时配置信息，包含用户标识等元数据
    
    返回值:
    str: 存储的记忆列表
    
    工作流程:
    1. 获取当前用户ID
    2. 遍历记忆列表
    3. 将每条记忆序列化并添加元数据
    4. 存入向量数据库
    """
    # 获取用户唯一标识
    user_id = get_user_id(config)
    
    # 遍历处理每条记忆
    for memory in memories:
        # 将三元组值拼接成字符串，用于向量化
        serialized = " ".join(memory.values())
        
        # 构建文档对象，包含记忆内容和元数据
        document = Document(
            # 记忆内容
            serialized,
            # 生成唯一文档ID
            id=str(uuid.uuid4()),
            # 元数据信息
            metadata={
                "user_id": user_id,  # 用户标识
                **memory,            # 展开原始三元组数据
            },
        )
        
        # 将文档添加到向量存储中
        recall_vector_store.add_documents([document])
    
    return memories


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
llm = ChatDeepSeek(model="deepseek-chat")
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
    model_with_tools = llm.bind_tools(tools)
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


"""
==== Ai Message ====

了解了，莫尔索确实喜欢写作。这让我想起了他的一些内心独白非常有深度，通过写作他似乎找到了表达自己情感的方式。如果有具体的作品或者情节你想讨论的话，我很乐意一起探讨。需要我为你找一些关于他的写作片段吗？
Tool Calls:
  save_recall_memory (call_e1901821652a4029bd6b6a)
 Call ID: call_e1901821652a4029bd6b6a
  Args:
    memories: [{'subject': '莫尔索', 'predicate': '爱好', 'object_': '写作'}]


==== Tool Message ====
Name: save_recall_memory

[{"subject": "莫尔索", "predicate": "爱好", "object_": "写作"}]

---------------------------------------------------------------------------

==== Ai Message ====

我已经记住了莫尔索爱好写作这一点。如果你有任何关于他的作品或写作习惯的问题，随时可以问我！

==== Ai Message ====
Tool Calls:
  save_recall_memory (call_b61742540a444689bb4506)
 Call ID: call_b61742540a444689bb4506
  Args:
    memories: [{'object_': 'LangGraph', 'predicate': '写作主题', 'subject': '莫尔索'}, {'object_': '技术文章', 'predicate': '写作内容', 'subject': '莫尔索'}]


==== Tool Message ====
Name: save_recall_memory

[{"subject": "莫尔索", "predicate": "写作主题", "object_": "LangGraph"}, {"subject": "莫尔索", "predicate": "写作内容", "object_": "技术文章"}]


==== Ai Message ====

我已经记录下莫尔索正在撰写一篇关于LangGraph的技术文章。如果您有任何关于这篇文章的具体问题或需要进一步的帮助，请随时告诉我！
"""
