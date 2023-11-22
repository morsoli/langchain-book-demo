from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
class CommaSeparatedListOutputParser(BaseOutputParser[List[str]]):
    """将LLMs 逗号分隔格式输出内容解析为的列表"""

    def parse(self, text: str) -> List[str]:
        """解析LLMs调用的输出"""
        return text.strip().split(", ")

template = """你是一个能生成逗号分隔列表的助手，用户会传入一个类别，你应该生成该类别中的5个对象，并以逗号分隔列表形式返回。
只返回逗号分隔的列表，不要包含其他内容。"""
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

if __name__ == "__main__":
    chain = chat_prompt | ChatOpenAI() | CommaSeparatedListOutputParser()
    print(chain.invoke({"text": "动物"}))
    # 输出：['狗,猫,鸟,鱼,兔子']