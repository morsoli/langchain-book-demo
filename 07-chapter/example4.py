from langchain.memory import ConversationKGMemory
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 创建一个ConversationKGMemory实例
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
memory = ConversationKGMemory(llm=llm)

# 模拟一段对话并保存上下文
memory.save_context({"input": "小李是程序员"}, {"output": "知道了，小李是程序员"})
memory.save_context({"input": "莫尔索是小李的笔名"}, {"output": "明白，莫尔索是小李的笔名"})

# 加载内存变量
variables = memory.load_memory_variables({"input": "告诉我关于小李的信息"})

if __name__ == "__main__":
    print(variables) 