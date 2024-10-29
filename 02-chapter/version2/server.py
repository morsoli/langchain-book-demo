from typing import List

from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
from langserve import add_routes

# 链定义
class CommaSeparatedListOutputParser(BaseOutputParser[List[str]]):
    """将 LLMs 逗号分隔格式输出内容解析为的列表"""

    def parse(self, text: str) -> List[str]:
        """解析LLMs调用的输出。"""
        return text.strip().split(", ")

template = """你是一个能生成逗号分隔列表的助手，用户会传入一个类别，你应该生成该类别中的5个对象，并以逗号分隔形式返回。
只返回逗号分隔的内容，不要包含其他内容。"""
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])
first_chain = chat_prompt | ChatTongyi() | CommaSeparatedListOutputParser()

# 应用定义
app = FastAPI(
  title="第一个LangChain 应用",
  version="0.0.1",
  description="LangChain应用接口",
)

# 添加链路由
add_routes(app, first_chain, path="/first_app")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)