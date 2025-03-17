import uuid
from typing import Literal
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph
from langgraph.constants import START, END
from langchain_deepseek import ChatDeepSeek

# 创建记忆系统，用于保存对话状态
memory = MemorySaver()

# 扩展State类，添加summary属性用于保存对话摘要
class State(MessagesState):
    summary: str

# 初始化聊天模型
model = ChatDeepSeek(model="deepseek-chat")

# 定义调用模型的逻辑
def call_model(state: State):
    # 如果存在摘要，将其作为系统消息添加
    summary = state.get("summary", "")
    if summary:
        system_message = f"之前对话的摘要：{summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    response = model.invoke(messages)
    # 返回响应消息
    return {"messages": [response]}

# 定义是否继续对话或进行摘要的逻辑
def should_continue(state: State) -> Literal["summarize_conversation", END]: # type: ignore
    """返回下一个执行节点。"""
    messages = state["messages"]
    # 如果消息超过4条，则进行对话摘要
    if len(messages) > 4:
        return "summarize_conversation"
    # 否则结束对话
    return END

# 定义对话摘要的逻辑
def summarize_conversation(state: State):
    # 首先，生成对话摘要
    summary = state.get("summary", "")
    if summary:
        summary_message = f"这是到目前为止的对话摘要：{summary}\n\n" \
                          "考虑上面的新消息，扩展摘要："
    else:
        summary_message = "请创建上面的对话摘要："

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    # 删除不再显示的消息，这里删除除了最后两条之外的所有消息
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

# 定义新的状态图
workflow = StateGraph(State)

# 添加对话节点和摘要节点
workflow.add_node("conversation", call_model)
workflow.add_node("summarize_conversation", summarize_conversation)

# 设置入口点为对话节点
workflow.add_edge(START, "conversation")

# 添加条件边
workflow.add_conditional_edges(
    "conversation",
    should_continue,
)

# 添加摘要节点
workflow.add_edge("summarize_conversation", END)

app = workflow.compile(checkpointer=memory)

# 打印总结信息
def print_summary(update):
    for _, v in update.items():
        for m in v["messages"]:
            m.pretty_print()
        if "summary" in v:
            print(v["summary"])

# 线程ID是一个唯一标识符，用于标识这次特定的对话。
thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}


# 定义初始人类消息
input_messages = [
    HumanMessage("我是李四，你叫什么？"),
    HumanMessage("你好AI，我想聊聊我的爱好和职业。")
]

# 处理初始消息并打印更新
for input_message in input_messages:
    input_message.pretty_print()
    for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
        print_summary(event)

# 第一次检查状态，可以看到还没有进行总结，这是因为列表中只有4条消息
print(app.get_state(config).values)  

input_message = HumanMessage(content="我喜欢阅读和徒步旅行。你呢？")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_summary(event)
# 第二次检查状态，可以看到有一条对话摘要，以及最新两条消息
print(app.get_state(config).values)

input_message = HumanMessage(content="我叫什么名字？")
input_message.pretty_print()
for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
    print_summary(event)
 
# input_message = HumanMessage(content="我最喜欢解决复杂的算法问题。那对你来说，最大的挑战是什么？")
# input_message.pretty_print()
# for event in app.stream({"messages": [input_message]}, config, stream_mode="updates"):
#     print_summary(event)