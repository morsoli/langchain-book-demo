from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 创建一个ConversationSummaryBufferMemory实例
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=10)

# 模拟一段对话并保存上下文
memory.save_context({"input": "嗨"}, {"output": "怎么了"})
memory.save_context({"input": "没什么，你呢"}, {"output": "也没什么"})

messages = memory.chat_memory.messages
previous_summary = ""
print(memory.predict_new_summary(messages, previous_summary))

from langchain.chains import ConversationChain

conversation_with_summary = ConversationChain(
    llm=llm,
    memory=ConversationSummaryBufferMemory(llm=llm, max_token_limit=40),
    verbose=True,
)
print(conversation_with_summary.predict(input="什么事?"))