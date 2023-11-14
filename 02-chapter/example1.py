from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

llm = OpenAI()
chat_model = ChatOpenAI()

text = "给生产杯子的公司取一个名字。"
messages = [HumanMessage(content=text)]


if __name__ == "__main__":
    print(llm.invoke(text))
    # >> 茶杯屋

    print(chat_model.invoke(messages))
    # >> content='杯享'
    