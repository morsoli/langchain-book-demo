from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    filter_messages,
)

# 定义李四和AI之间的对话消息列表，内容涉及爱好和职业等话题
messages = [
    HumanMessage("你好AI，我想聊聊我的爱好和职业。", id="1", name="李四"),
    AIMessage("你好李四，很高兴和你聊天。你的爱好是什么呢？"),
    HumanMessage("我喜欢阅读和徒步旅行。你呢？", id="2",),
    AIMessage("作为一个AI，我‘喜欢’处理数据和帮助解决问题。你从事什么职业？"),
    HumanMessage("我是一名软件工程师。总是有很多问题需要解决。", id="3",),
    AIMessage("那听起来很有趣！你最喜欢编程的哪个部分？"),
    HumanMessage("我最喜欢解决复杂的算法问题。那对你来说，最大的挑战是什么？", id="4",),
    AIMessage("对我来说，最大的挑战是如何更自然地与人类沟通。"),
]

# print(filter_messages(messages, include_types="human"))

print(filter_messages(messages, include_ids=["3"]))