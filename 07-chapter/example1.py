from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain 
from langchain.memory import ConversationBufferMemory  # 导入 ConversationBufferMemory 类，用于管理对话缓冲区内存
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 创建一个 ChatPromptTemplate 实例，用于定义如何提示聊天模型
prompt = ChatPromptTemplate(
    messages=[
        # 这里定义了系统消息模板。它定义了聊天机器人的身份和聊天背景。
        SystemMessagePromptTemplate.from_template(
            "你是一个友好的聊天机器人，正在与人类进行对话。"
        ),
        # MessagesPlaceholder 是对话历史的占位符，它的变量名称需要与内存键对齐
        MessagesPlaceholder(variable_name="chat_history"),
        # 这里定义了人类消息模板。它定义了如何呈现用户的问题。
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

# 创建一个 ConversationBufferMemory 实例。这里的 `return_messages=True` 表明我们需要返回消息列表以适应 MessagesPlaceholder
# 注意 `"chat_history"` 与 MessagesPlaceholder 的名称对齐。
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 创建一个 LLMChain 实例，用于实现整个对话流程
# 这包括使用前面定义的聊天模型、提示模板和内存
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,  # 设置为 True 以输出详细的调试信息
    memory=memory
)


if __name__ == "__main__":
    print(conversation({"question": "你好"}))