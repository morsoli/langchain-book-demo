import os
from langchain.llms.openai import OpenAI
from langchain.chains.api import podcast_docs
from langchain.chains import APIChain
from dotenv import load_dotenv 

# 加载环境变量
load_dotenv()

LISTENNOTES_API_KEY = os.environ.get("LISTENNOTES_API_KEY")

if __name__ == "__main__":
    # 创建OpenAI模型实例，并设置温度参数为0。设置播客API的访问密钥。
    llm = OpenAI(temperature=0)
    headers = {"X-ListenAPI-Key": LISTENNOTES_API_KEY}
    chain = APIChain.from_llm_and_api_docs(llm, podcast_docs.PODCAST_DOCS, headers=headers, verbose=True)
    # 使用chain.run方法执行APIChain，传入自然语言查询，搜索关于ChatGPT的节目，要求超过30分钟，且只返回一条结果。
    chain.run("搜索关于ChatGPT的节目, 要求超过30分钟，只返回一条")