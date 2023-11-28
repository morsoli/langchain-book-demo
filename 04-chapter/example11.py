# 引入所需的类和方法
from langchain.chains import RetrievalQA
from langchain.llms.openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv 

# 加载环境变量
load_dotenv()

def test_qa():
    # 加载文档
    loader = TextLoader("./test.txt")
    documents = loader.load()
    # 将文档分割为小块
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # 使用OpenAI的嵌入模型
    embeddings = OpenAIEmbeddings()
    # 使用Chroma构建文档搜索索引
    docsearch = Chroma.from_documents(texts, embeddings)
    # 加载问答链
    qa_chain = load_qa_chain(OpenAI(temperature=0), chain_type="map_reduce")
    # 创建RetrievalQA实例
    qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=docsearch.as_retriever())
    # 运行问答系统
    qa.run("LangChain 支持哪些编程语言?")
    
if __name__ == "__main__":
    test_qa()
