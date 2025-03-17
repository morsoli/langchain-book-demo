from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

prompt = ChatPromptTemplate.from_template("给售卖 {topic} 的店铺起一个好听的店名")
model =  ChatOpenAI(model="glm-4-air", base_url="https://open.bigmodel.cn/api/paas/v4/", api_key=ZHIPU_API_KEY)
output_parser = StrOutputParser()

chain = prompt | model | output_parser
chain.invoke({"topic": "咖啡"})