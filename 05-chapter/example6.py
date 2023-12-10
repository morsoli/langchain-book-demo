from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def test():
    # 加载文档
    loader = TextLoader("./test.txt")
    documents = loader.load()

    # 文本分割
    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    texts = text_splitter.split_documents(documents)

    # 初始化嵌入向量
    embeddings = OpenAIEmbeddings()

    # 使用文档和嵌入向量创建 Chroma 向量存储
    db = Chroma.from_documents(texts, embeddings)

    # 将向量存储转换为检索器
    # 将向量存储转换为检索器 - 使用默认的相似性搜索
    retriever = db.as_retriever()
    docs = retriever.get_relevant_documents("LLMOps的含义是什么？")
    print("默认相似性搜索结果：\n", docs)

    # 使用最大边际相关性（MMR）搜索
    retriever_mmr = db.as_retriever(search_type="mmr")
    docs_mmr = retriever_mmr.get_relevant_documents("LLMOps的含义是什么？")
    print("MMR 搜索结果：\n", docs_mmr)

    # 设置相似度分数阈值
    retriever_similarity_threshold = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5})
    docs_similarity_threshold = retriever_similarity_threshold.get_relevant_documents("LLMOps的含义是什么？")
    print("相似度分数阈值搜索结果：\n", docs_similarity_threshold)

    # 指定 top k 搜索
    retriever_topk = db.as_retriever(search_kwargs={"k": 1})
    docs_topk = retriever_topk.get_relevant_documents("LLMOps的含义是什么？")
    print("Top K 搜索结果：\n", docs_topk)


if __name__ == "__main__":
    test()