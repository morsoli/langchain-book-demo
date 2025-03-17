from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup as Soup
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langgraph.graph import END, StateGraph, START
from langchain_anthropic import ChatAnthropic

# LCEL 文档
url = "https://python.langchain.com/docs/concepts/#langchain-expression-language-lcel"
loader = RecursiveUrlLoader(
    url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
)
docs = loader.load()

# 根据 URL 排序列表并获取文本
d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]
)


# 通义千问提示词
code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个擅长 LCEL（LangChain 表达语言）的编码助手。\n
    这里是完整的 LCEL 文档： \n ------- \n  {context} \n ------- \n 根据上述提供的文档回答用户问题。
    确保你提供的代码可以执行，包含所有必要的导入和变量定义。结构化你的回答，先描述代码解决方案。
    然后列出导入。最后列出功能代码块。以下是用户问题：""",
        ),
        ("placeholder", "{messages}"),
    ]
)
# 数据模型
class Code(BaseModel):
    """LCEL 问题代码解决方案的模式。"""

    prefix: str = Field(description="问题和方法的描述")
    imports: str = Field(description="代码块导入语句")
    code: str = Field(description="不包括导入语句的代码块")
llm = ChatDeepSeek(model="deepseek-chat")
code_gen_chain_tongyi = code_gen_prompt | llm.with_structured_output(Code)

# question = "如何直接将字符串传递给Runnable对象构造提示词？"
# tongyi_solution = code_gen_chain_tongyi.invoke(
#     {"context": concatenated_content, "messages": [("user", question)]}
# )


# 强制使用工具
code_gen_prompt_claude = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """<instructions> 你是一个擅长 LCEL（LangChain 表达语言）的编码助手。\n
    这里是 LCEL 文档： \n ------- \n  {context} \n ------- \n 根据上述提供的文档回答用户问题。
    确保你提供的代码可以执行，包含所有必要的导入和变量定义。结构化你的回答：1）描述代码解决方案的前缀，
    2）导入，3）功能代码块。调用代码工具以正确结构化输出。</instructions> \n 以下是用户问题：""",
        ),
        ("placeholder", "{messages}"),
    ]
)

expt_llm = "claude-3-5-sonnet-20241022"
llm = ChatAnthropic(
    model=expt_llm,
    default_headers={"anthropic-beta": "tools-2024-04-04"},
)
structured_llm_claude = llm.with_structured_output(Code, include_raw=True)
# # 检查工具使用错误
# def check_claude_output(tool_output):
#     """检查解析错误或未能调用工具"""

#     # 解析错误
#     if tool_output["parsing_error"]:
#         # 报告输出和解析错误
#         print("解析错误！")
#         raw_output = str(tool_output["raw"].content)
#         error = tool_output["parsing_error"]
#         raise ValueError(
#             f"解析你的输出时出错！确保调用了工具。输出：{raw_output}。\n 解析错误：{error}"
#         )

#     # 未调用工具
#     elif not tool_output["parsed"]:
#         print("未能调用工具！")
#         raise ValueError(
#             "你没有使用提供的工具！确保调用工具以结构化输出。"
#         )
#     return tool_output

# # 带输出检查的链
# code_gen_chain = code_gen_prompt_claude | structured_llm_claude | check_claude_output
def parse_output(solution):
    """当我们添加 'include_raw=True' 到结构化输出时，
    它将返回一个包含 'raw'、'parsed'、'parsing_error' 的字典。"""

    return solution["parsed"]

code_gen_chain = code_gen_prompt_claude | structured_llm_claude | parse_output
# question = "如何直接将字符串传递给Runnable对象构造提示词？"
# solution = code_gen_chain.invoke(
#     {"context": concatenated_content, "messages": [("user", question)]}
# )

class GraphState(TypedDict):
    """
    表示我们的图的状态。

    属性：
        error : 二进制标志，用于控制流以指示是否触发了测试错误
        messages : 包含用户问题、错误消息、推理
        generation : 代码解决方案
        iterations : 尝试次数
    """

    error: str
    messages: List
    generation: str
    iterations: int

# 最大尝试次数
max_iterations = 3
# 反射
# flag = 'reflect'
flag = "no reflect"

### 节点
def generate(state: GraphState):
    """
    生成代码解决方案

    参数：
        state (dict): 当前图状态

    返回：
        state (dict): 新增键到状态，generation
    """

    print("---生成代码解决方案---")

    # 状态
    messages = state["messages"]
    iterations = state["iterations"]
    error = state["error"]

    # 我们被路由回生成带有错误的状态
    if error == "yes":
        messages += [
            (
                "user",
                "现在，再试一次。调用代码工具以结构化输出，包括前缀、导入和代码块：",
            )
        ]

    # 解决方案
    code_solution = code_gen_chain.invoke(
        {"context": concatenated_content, "messages": messages}
    )
    messages += [
        (
            "assistant",
            f"{code_solution.prefix} \n 导入：{code_solution.imports} \n 代码：{code_solution.code}",
        )
    ]

    # 增量
    iterations = iterations + 1
    return {"generation": code_solution, "messages": messages, "iterations": iterations}


def code_check(state: GraphState):
    """
    检查代码

    参数：
        state (dict): 当前图状态

    返回：
        state (dict): 新增键到状态，error
    """

    print("---检查代码---")

    # 状态
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]

    # 获取解决方案组件
    imports = code_solution.imports
    code = code_solution.code

    # 检查导入
    try:
        exec(imports)
    except Exception as e:
        print("---代码导入检查：失败---")
        error_message = [("user", f"你的解决方案导入测试失败：{e}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    # 检查执行
    try:
        exec(imports + "\n" + code)
    except Exception as e:
        print("---代码块检查：失败---")
        error_message = [("user", f"你的解决方案代码执行测试失败：{e}")]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    # 无错误
    print("---无代码测试失败---")
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
    }


def reflect(state: GraphState):
    """
    反思错误

    参数：
        state (dict): 当前图状态

    返回：
        state (dict): 新增键到状态，generation
    """

    print("---生成代码解决方案---")

    # 状态
    messages = state["messages"]
    iterations = state["iterations"]
    code_solution = state["generation"]
    # 添加反思
    reflections = code_gen_chain.invoke(
        {"context": concatenated_content, "messages": messages}
    )
    messages += [("assistant", f"这里是对错误的反思：{reflections}")]
    return {"generation": code_solution, "messages": messages, "iterations": iterations}


def decide_to_finish(state: GraphState):
    """
    确定是否完成。

    参数：
        state (dict): 当前图状态

    返回：
        str: 下一个要调用的节点
    """
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations == max_iterations:
        print("---决策：完成---")
        return "end"
    else:
        print("---决策：重试解决方案---")
        if flag == "reflect":
            return "reflect"
        else:
            return "generate"


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
           print(updates["messages"][-1][1])
       else:
           print(updates)
           
       # 添加空行分隔不同节点的输出
       print("\n")
       

builder = StateGraph(GraphState)
builder.add_node("gen_code", generate)  # 生成解决方案
builder.add_node("check_code", code_check)  # 检查代码
builder.add_node("reflect_code", reflect)  # 反思
builder.add_edge(START, "gen_code")
builder.add_edge("gen_code", "check_code")
builder.add_conditional_edges(
    "check_code",
    decide_to_finish,
    {
    "end": END,
    "reflect": "reflect_code",
    "regenerate": "gen_code",
    },
)
builder.add_edge("reflect_code", "gen_code")
graph = builder.compile()
question = "如何直接将字符串传递给langchain_core中的Runnable对象？"

# for chunk in graph.stream({"messages": [("user", question)], "iterations": 0, "error": ""}
# ):
#     get_stream_chunk(chunk)
print(graph.get_graph().draw_mermaid())