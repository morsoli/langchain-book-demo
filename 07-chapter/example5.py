from langchain.memory import ConversationSummaryMemory
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# 创建一个ConversationSummaryMemory实例
memory = ConversationSummaryMemory(llm=llm, return_messages=True)

# 模拟一段对话并保存上下文
memory.save_context({"input": "今天天气怎么样？"}, {"output": "今天天气晴朗。"})
memory.save_context({"input": "有什么好玩的地方推荐吗？"}, {"output": "附近的公园很不错。"})

# 加载内存变量，获取对话摘要
variables = memory.load_memory_variables({})
print(variables)
messages = memory.chat_memory.messages
previous_summary = "使用中文总结"
print(memory.predict_new_summary(messages, previous_summary))