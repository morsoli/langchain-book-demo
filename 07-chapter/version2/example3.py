from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    trim_messages,
)
from langchain_community.chat_models.tongyi import ChatTongyi

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

# 使用trim_messages函数来裁剪消息列表，使其不超过特定的token数量限制
selected_messages = trim_messages(
    messages,
    # 使用ChatTongyi来计算消息中的token数量
    token_counter=ChatTongyi(),
    max_tokens=200,  # 设置令牌数量的上限为200
    # 确保对话历史以人类消息开始
    start_on="human",
    # 如果原始对话历史中包含系统消息，则保留它，因为系统消息可能包含对模型的特殊指令
    include_system=True,
    strategy="last",  # 如果需要裁剪消息以适应令牌限制，优先保留列表末尾的消息
)

# 打印选中的消息
for msg in selected_messages:
    msg.pretty_print()