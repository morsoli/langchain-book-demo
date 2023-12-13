from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationTokenBufferMemory
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=20)

# 模拟一段对话并保存上下文
memory.save_context({"input": "嗨"}, {"output": "怎么了"})
memory.save_context({"input": "没什么，你呢"}, {"output": "也没什么"})

variables = memory.load_memory_variables({})
print(variables)