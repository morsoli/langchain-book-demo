from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    trim_messages,
)

# 定义李四和AI之间的对话消息列表，内容涉及爱好和职业等话题
messages = [
    HumanMessage("你好AI，我想聊聊我的爱好和职业。"),
    AIMessage("你好李四，很高兴和你聊天。你的爱好是什么呢？"),
    HumanMessage("我喜欢阅读和徒步旅行。你呢？"),
    AIMessage("作为一个AI，我‘喜欢’处理数据和帮助解决问题。你从事什么职业？"),
    HumanMessage("我是一名软件工程师。总是有很多问题需要解决。"),
    AIMessage("那听起来很有趣！你最喜欢编程的哪个部分？"),
    HumanMessage("我最喜欢解决复杂的算法问题。那对你来说，最大的挑战是什么？"),
    AIMessage("对我来说，最大的挑战是如何更自然地与人类沟通。"),
]

# 使用trim_messages来选择消息，确保对话历史的有效性和连贯性
# 这里我们假设我们只想保留最后5条消息
selected_messages = trim_messages(
    messages,
    token_counter=len,  # 使用len函数来计算消息数量
    max_tokens=4,  # 允许最多选择4条消息
    strategy="last",  # 选择策略为“最后”，即优先保留列表末尾的消息
    start_on="human",  # 确保对话历史以人类消息开始
    include_system=False,  # 如果不需要系统消息，可以不包含它
    allow_partial=False,  # 不允许部分消息被选中
)

# 打印选中的消息
for msg in selected_messages:
    msg.pretty_print()