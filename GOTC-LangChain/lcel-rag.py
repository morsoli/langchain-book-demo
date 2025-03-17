import os
import bs4
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_sk_xxx"
os.environ["LANGCHAIN_PROJECT"] = "lcel-rag"


ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
zhipu_embedding = OpenAIEmbeddings(model="embedding-2", base_url="https://open.bigmodel.cn/api/paas/v4/", api_key=ZHIPU_API_KEY)

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://liduos.com/llm-secure.html",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("content", "card-content", "article")
        )
    ),
)
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=zhipu_embedding)

retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template(
        """你是一个回答问题的助手，使用以下检索到的上下文片段回答问题，如果你不知道答案，就说你不知道，最多使用三句话，答案要简明扼要。
        问题: {question} \
        上下文: {context} \
        答案: """)
])

llm = ChatOpenAI(model="glm-4-air", base_url="https://open.bigmodel.cn/api/paas/v4/", api_key=ZHIPU_API_KEY)

rag_chain = RunnableParallel(context = retriever, question = RunnablePassthrough() ) | rag_prompt | llm 

print(rag_chain.invoke("什么是提示词攻击?").content)