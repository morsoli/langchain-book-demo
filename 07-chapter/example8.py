from langchain.memory import VectorStoreRetrieverMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 在对话链中使用
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# 这里使用OpenAI的嵌入式模型作为向量化函数
vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
# 创建VectorStoreRetrieverMemory
retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(retriever=retriever)

memory.save_context({"input": "我喜欢吃火锅"}, {"output": "听起来很好吃"})
memory.save_context({"input": "我喜欢打羽毛球"}, {"output": "..."})
memory.save_context({"input": "我不喜欢看摔跤比赛"}, {"output": "我也是"}) #


PROMPT_TEMPLATE = """以下是人类和 AI 之间的友好对话。AI 话语多且提供了许多来自其上下文的具体细节。如果 AI 不知道问题的答案，它会诚实地说不知道。

以前对话的相关片段：
{history}

（如果不相关，你不需要使用这些信息）

当前对话：
人类：{input}
AI：
"""

prompt = PromptTemplate(input_variables=["history", "input"], template=PROMPT_TEMPLATE)
conversation_with_summary = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

print(conversation_with_summary.predict(input="你好，我是莫尔索，你叫什么"))
print(conversation_with_summary.predict(input="我喜欢的食物是什么？"))
print(conversation_with_summary.predict(input="我提到了哪些运动？"))