# 引入 LangChain 库的相关模块和类
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.agents.react.base import DocstoreExplorer
from langchain.docstore import Wikipedia
from langchain.chat_models import ChatOpenAI

# 初始化一个文档存储浏览器，以探索 Wikipedia 数据
docstore = DocstoreExplorer(Wikipedia())

# 定义一个工具列表，包括搜索和查找工具
tools = [
    Tool(
        name="Search",              # 工具名称为“搜索”
        func=docstore.search,       # 指定搜索功能
        description="在你需要进行搜索式提问时非常实用",
    ),
    Tool(
        name="Lookup",              # 工具名称为“查找”
        func=docstore.lookup,       # 指定查找功能
        description="在你需要进行查找式提问时非常实用",
    ),
]

# 初始化 OpenAI 类的实例，设置温度参数为 0，并指定模型名称
llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)

# 初始化代理，并将工具、LLM 和代理类型配置进去
react = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=True)

if __name__ == "__main__":
    question = "哪位运动员与林丹交手次数最多，被誉为羽毛球历史上最精彩的交锋，他第一次赢林丹是什么时候？用中文回答"
    react.run(question)