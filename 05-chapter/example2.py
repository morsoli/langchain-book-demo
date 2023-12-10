from langchain.document_loaders import WebBaseLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import MultiQueryRetriever
from dotenv import load_dotenv

# 加载环境变量，通常用于配置文件中的API密钥等敏感信息
load_dotenv()


def test_query():
    # 网页加载内容
    loader = WebBaseLoader("https://mp.weixin.qq.com/s/Y0t8qrmU5y6H93N-Z9_efw")
    data = loader.load()

    # 拆分文本
    # 使用递归字符文本分割器将文本分割成小块，每块最大512个字符，不重叠
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    splits = text_splitter.split_documents(data)

    # 创建向量数据库
    # 使用 OpenAI 的嵌入向量模型
    embedding = OpenAIEmbeddings()
    # 使用分割后的文档和嵌入向量创建 Chroma 向量存储
    vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

    # 定义一个查询问题
    question = "程序员如何实现自我成长?"

    # 创建一个基于语言模型的检索器
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    # 使用多查询检索器，结合向量数据库和语言模型
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(), llm=llm
    )

    # 使用检索器获取与查询相关的文档
    unique_docs = retriever_from_llm.get_relevant_documents(query=question)
    print(unique_docs)

if __name__ == "__main__":
    test_query()