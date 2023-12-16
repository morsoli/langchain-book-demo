from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import (
    CombinedMemory,
    ConversationBufferMemory,
    ConversationSummaryMemory,
)
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化ChatOpenAI模型
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# 创建会话缓存内存，用于存储聊天历史
conv_memory = ConversationBufferMemory(
    memory_key="chat_history_lines", input_key="input"
)

# 创建会话摘要内存，用于生成对话摘要
summary_memory = ConversationSummaryMemory(llm=llm, input_key="input")

# 组合两种内存形式
memory = CombinedMemory(memories=[conv_memory, summary_memory])

# 设置对话模板
_DEFAULT_TEMPLATE = """以下是一个人类和AI之间的友好对话。AI很健谈，并提供了来自其上下文的许多具体细节。如果AI不知道某个问题的答案，它会如实说不知道。

对话摘要:
{history}
当前对话:
{chat_history_lines}
人类: {input}
AI:"""

# 创建提示模板
PROMPT = PromptTemplate(
    input_variables=["history", "input", "chat_history_lines"],
    template=_DEFAULT_TEMPLATE,
)

# 创建对话链
conversation = ConversationChain(llm=llm, verbose=True, memory=memory, prompt=PROMPT)

conversation.run("讲个笑话？")
conversation.run("还有吗？")
conversation.run("让我讲一个")