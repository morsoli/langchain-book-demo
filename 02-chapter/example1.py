from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

llm = OpenAI()
chat_model = ChatOpenAI()

text = "给生产杯子的公司取一个名字。"
messages = [HumanMessage(content=text)]


if __name__ == "__main__":
    print(llm.invoke(text))
    # 输出： 茶杯屋

    print(chat_model.invoke(messages))
    # 输出： content='杯享'
    