from langchain.schema import Document
from langchain.llms.openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv 

# 加载环境变量
load_dotenv()


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
    chain = load_summarize_chain(OpenAI(temperature=0), chain_type="map_reduce")
    chain.run(docs)
    
if __name__ == "__main__":
    test_summary()