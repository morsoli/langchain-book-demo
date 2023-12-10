from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

# 加载环境变量，通常用于配置文件中的API密钥等敏感信息
load_dotenv()

def test():
    # 示例文档列表
    doc_list = [
        "我喜欢苹果",
        "我喜欢橙子",
        "苹果和橙子都是水果",
    ]
    # 初始化 BM25 检索器
    bm25_retriever = BM25Retriever.from_texts(doc_list)
    bm25_retriever.k = 2

    # 使用 OpenAI 嵌入向量初始化 Chroma 检索器
    embedding = OpenAIEmbeddings()
    chroma_vectorstore = Chroma.from_texts(doc_list, embedding)
    chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 2})

    # 初始化 EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5]
    )

    # 检索与查询“苹果”相关的文档
    docs = ensemble_retriever.get_relevant_documents("苹果")
    
    return docs

if __name__ == "__main__":
    print(test())
