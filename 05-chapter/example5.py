from langchain.chat_models.openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from langchain.vectorstores.chroma import Chroma
from langchain.retrievers.web_research import WebResearchRetriever
from dotenv import load_dotenv

load_dotenv()

def test():
    # 初始化向量存储
    vectorstore = Chroma(
        embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db_oai"
    )

    # 初始化语言模型
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 初始化谷歌搜索API包装器
    search = GoogleSearchAPIWrapper()

    # 初始化WebResearchRetriever
    web_research_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore,
        llm=llm,
        search=search,
    )

    # 使用WebResearchRetriever检索与查询相关的文档
    user_input = "LLM驱动的自主代理是如何工作的？"
    docs = web_research_retriever.get_relevant_documents(user_input)\
    # 打印检索到的文档
    return docs

if __name__ == "__main__":
    print(test())
