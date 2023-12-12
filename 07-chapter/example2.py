from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 创建一个ConversationBufferWindowMemory实例，只保留最后1次互动
memory = ConversationBufferWindowMemory(k=1)

# 保存上下文信息
memory.save_context({"input": "嗨"}, {"output": "怎么了"})
memory.save_context({"input": "没什么，你呢"}, {"output": "也没什么"})

# 加载内存变量
variables = memory.load_memory_variables({})

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
conversation_with_summary = ConversationChain(
    llm=llm,
    memory=ConversationBufferWindowMemory(k=2),
    verbose=True
)

if __name__ == "__main__":
    print(variables)
    # 进行预测
    print(conversation_with_summary.predict(input="你最近怎么样?"))