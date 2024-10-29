from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter,LLMChainExtractor
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


def pretty_print_docs(docs):
    # 格式化打印文档
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

def test():
    # 从网页加载内容
    loader = WebBaseLoader("https://mp.weixin.qq.com/s/Y0t8qrmU5y6H93N-Z9_efw")
    data = loader.load()

    # 拆分文本
    # 使用递归字符文本分割器将文本分割成小块，每块最大512个字符，不重叠
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    splits = text_splitter.split_documents(data)

    # 创建语言模型实例
    llm = ChatTongyi()

    # 创建向量数据库检索器
    retriever = Chroma.from_documents(documents=splits, embedding=DashScopeEmbeddings()).as_retriever()
    question = "LLMOps指的是什么?"

    # 未压缩时查询的结果
    docs = retriever.get_relevant_documents(query=question)
    pretty_print_docs(docs)

    # 创建链式提取器
    compressor = LLMChainExtractor.from_llm(llm)
    # 创建上下文压缩检索器
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    # 压缩后的查询结果
    docs = compression_retriever.get_relevant_documents(query=question)
    pretty_print_docs(docs)

    # 创建嵌入向量过滤器
    embeddings_filter = EmbeddingsFilter(embeddings=DashScopeEmbeddings(), similarity_threshold=0.76)
    # 使用过滤器创建上下文压缩检索器
    compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=retriever)
    # 过滤后的查询结果
    docs = compression_retriever.get_relevant_documents(query=question)
    pretty_print_docs(docs)

if __name__ == "__main__":
    test()