from langchain.retrievers import EnsembleRetriever
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma


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
    embedding = DashScopeEmbeddings()
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
