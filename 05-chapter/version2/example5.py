from langchain_deepseek import ChatDeepSeek
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_community.retrievers.web_research import WebResearchRetriever
from langchain_chroma import Chroma
from langchain_google_community import GoogleSearchAPIWrapper


import os
os.environ["USER_AGENT"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"

def test():
    # 初始化向量存储
    vectorstore = Chroma(
        embedding_function=DashScopeEmbeddings(), persist_directory="./chroma_db_oai"
    )

    # 初始化语言模型
    llm = ChatDeepSeek(model="deepseek-chat")

    # 初始化谷歌搜索API包装器
    search = GoogleSearchAPIWrapper()

    # 初始化WebResearchRetriever
    web_research_retriever = WebResearchRetriever(allow_dangerous_requests=True).from_llm(
        vectorstore=vectorstore,
        llm=llm,
        search=search,
    )

    # 使用WebResearchRetriever检索与查询相关的文档
    user_input = "LLM驱动的自主代理是如何工作的？"
    docs = web_research_retriever.invoke(user_input)
    # 打印检索到的文档
    return docs

if __name__ == "__main__":
    print(test())
