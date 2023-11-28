# 引入所需的类和方法
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv 

# 加载环境变量
load_dotenv()

# 测试函数
def test_converstion():
    # 加载文档
    loader = TextLoader("./test.txt")
    documents = loader.load()
    # 将文档分割为较小的段落
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)
    # 使用OpenAI生成文档的嵌入
    embeddings = OpenAIEmbeddings()
    # 使用Chroma构建向量存储，便于后续检索
    vectorstore = Chroma.from_documents(documents, embeddings)
    # 设置对话历史存储
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # 创建ConversationalRetrievalChain实例
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever(), memory=memory)
    # 进行第一次查询
    query = "这本书包含哪些内容？"
    result = qa({"question": query})
    print(result)
    # 保存聊天历史，用于下一次查询
    chat_history = [(query, result["answer"])]
    # 进行第二次查询，包括之前的聊天历史
    query = "还有要补充的吗"
    result = qa({"question": query, "chat_history": chat_history})
    print(result["answer"])
    
    
if __name__ == "__main__":
    test_converstion()