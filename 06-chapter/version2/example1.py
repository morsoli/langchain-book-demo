# 导入langchain库中的相关模块
from langchain_core.tools import tool
from langchain_community.chat_models import ChatTongyi
from langgraph.prebuilt import create_react_agent

model = ChatTongyi()

# 定义一个工具函数，用于获取句子中不同汉字的数量
@tool
def count_unique_chinese_characters(sentence):
    """用于计算句子中不同汉字的数量"""
    unique_characters = set()

    # 遍历句子中的每个字符
    for char in sentence:
        # 检查字符是否是汉字
        if '\u4e00' <= char <= '\u9fff':
            unique_characters.add(char)

    # 返回不同汉字的数量
    return len(unique_characters)

tools = [count_unique_chinese_characters]
langgraph_agent_executor = create_react_agent(model, tools)

# 主函数
if __name__ == "__main__":
    query = "‘如何用LangChain实现一个代理’这句话共包含几个不同的汉字"
    messages = langgraph_agent_executor.invoke({"messages": [("human", query)]})
    print(messages["messages"][-1].content)
    messages2 = langgraph_agent_executor.invoke({"messages": messages["messages"]+[("human", "")]})
    print(messages2["messages"][-1].content)

