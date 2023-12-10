from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

# 加载环境变量，通常用于配置文件中的API密钥等敏感信息
load_dotenv()

    
def test_chromadb():
    # 导入所需的模块和类
    # 加载文本文件，这里以《西游记》为例
    raw_documents = TextLoader("./西游记.txt", encoding="utf-8").load()

    # 创建文本分割器，将文本分割成较小的部分
    # chunk_size 定义每个部分的大小，chunk_overlap 定义部分之间的重叠
    text_splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=32)

    # 将原始文档分割成更小的文档
    documents = text_splitter.split_documents(raw_documents)

    # 使用文档和 OpenAI 的嵌入向量创建 Chroma 向量存储
    db = Chroma.from_documents(documents, OpenAIEmbeddings())

    # 定义一个查询，这里查询的是孙悟空被压在五行山下的故事
    query = "孙悟空怎么被压在五行山下的？"

    # 在数据库中进行相似度搜索，k=1 表示返回最相关的一个文档
    docs = db.similarity_search(query, k=1)

    # 打印找到的最相关文档的内容
    print(docs[0].page_content)


if __name__ == "__main__":
    test_chromadb()