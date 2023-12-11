import re
import time
import logging
import pickle

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationTokenBufferMemory
from langchain.vectorstores.chroma import Chroma
from conversation import ConversationRetrievalChain

from retrivers import MyRetriever
from config import *

# 设置日志记录器
logger = logging.getLogger(__name__)
# 初始化语言模型
llm = ChatOpenAI(model=MODEL_NAME, temperature=0, max_tokens=1200)

def load_embedding(store_name, embedding, suffix, path):
    """加载chroma嵌入"""
    vector_store = Chroma(
        persist_directory=f"{path}/chroma_{store_name}_{suffix}",
        embedding_function=embedding,
    )
    return vector_store

# 加载检索数据库
db_embedding_chunks_small = load_embedding(
    store_name="openAIEmbeddings",
    embedding=OpenAIEmbeddings(),
    suffix="chunks_small",
    path=DB_DIR,
)
db_embedding_chunks_medium = load_embedding(
    store_name="openAIEmbeddings",
    embedding=OpenAIEmbeddings(),
    suffix="chunks_medium",
    path=DB_DIR,
)

def load_pickle(prefix, suffix, path):
    """从pickle文件加载数据"""
    with open(f"{path}/{prefix}_{suffix}.pkl", "rb") as file:
        return pickle.load(file)

# 加载文档块
db_docs_chunks_small = load_pickle(
    prefix="docs_pickle", suffix="small_chunks", path=DB_DIR
)
db_docs_chunks_medium = load_pickle(
    prefix="docs_pickle", suffix="medium_chunks", path=DB_DIR
)
# 加载文件名
file_names = load_pickle(prefix="file", suffix="names", path=DB_DIR)

# 初始化检索器
my_retriever = MyRetriever(
    llm=llm,
    embedding_chunks_small=db_embedding_chunks_small,
    embedding_chunks_medium=db_embedding_chunks_medium,
    docs_chunks_small=db_docs_chunks_small,
    docs_chunks_medium=db_docs_chunks_medium,
    first_retrieval_k=FIRST_RETRIEVAL_K,
    second_retrieval_k=SECOND_RETRIEVAL_K,
    num_windows=NUM_WINDOWS,
    retriever_weights=RETRIEVER_WEIGHTS,
)

# 初始化内存
memory = ConversationTokenBufferMemory(
    llm=llm,
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    return_messages=True,
    max_token_limit=MAX_CHAT_HISTORY,
)

# 初始化基于检索的问答链
qa = ConversationRetrievalChain.from_llm(
    llm,
    my_retriever,
    file_names=file_names,
    memory=memory,
    return_source_documents=False,
    return_generated_question=False,
)

if __name__ == "__main__":
    while True:
        user_input = input("人类: ")
        start_time = time.time()
        # 处理用户输入
        user_input_ = re.sub(r"^人类: ", "", user_input)
        # 运行问答链
        resp = qa({"question": user_input_})
        # 输出回答
        print(f"AI:{resp['answer']}")
        # 输出处理时间
        print(f"耗时: {time.time() - start_time}")
        print("=" * 66)
