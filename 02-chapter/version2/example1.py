from langchain_core.messages import HumanMessage
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.llms.tongyi import Tongyi
from langchain_openai import OpenAI, ChatOp

llm = Tongyi()
chat_model = ChatTongyi()

text = "给生产杯子的公司取一个名字，直接输出最终名字，无需做额外解释"
messages = [HumanMessage(content=text)]


if __name__ == "__main__":
    print(llm.invoke(text))
    # 输出： 茶杯屋

    print(chat_model.invoke(messages), type)
    # 输出： content='杯享'
    