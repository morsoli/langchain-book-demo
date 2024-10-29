from langchain_core.documents import Document
from langchain_community.llms.tongyi import Tongyi
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

def test_summary():
    text_splitter = CharacterTextSplitter()
    # 读取文件中的文本内容
    with open("./test.txt") as f:
        state_of_the_union = f.read()
    # 利用文本分割器将长文本分割成更小的部分
    texts = text_splitter.split_text(state_of_the_union)
    # 将每段文本转换为Document对象
    docs = [Document(page_content=t) for t in texts[:3]]
    # 使用load_summarize_chain函数加载摘要处理链
    chain = load_summarize_chain(Tongyi(), chain_type="map_reduce")
    chain.run(docs)
    
if __name__ == "__main__":
    test_summary()