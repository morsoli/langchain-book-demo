# 引入 LangChain 库中相关的模块和类
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from dotenv import load_dotenv

# 加载环境变量，通常用于从 .env 文件中读取配置
load_dotenv()

# 初始化一个 ChatOpenAI 类的实例，指定使用 GPT-4 模型，并设置温度参数为 0，以提高输出的确定性
llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)

# 创建一个 Google 搜索 API 的包装器实例
search = GoogleSearchAPIWrapper()

# 定义一个工具列表，其中包括一个搜索工具，这个工具将用于执行搜索任务
tools = [
    Tool(
        name="Intermediate Answer",  # 工具的名称, 这个不可以变
        func=search.run,            # 指定工具执行的函数
        description="在你需要进行搜索式提问时非常实用",  # 对工具的描述
    )
]

# 主程序入口
if __name__ == "__main__":
    # 初始化一个代理实例，该代理结合了 LLM 和定义的工具
    agent = initialize_agent(
        tools, llm, 
        agent=AgentType.SELF_ASK_WITH_SEARCH,  # 代理类型为自问搜索式代理
        verbose=True,                          # 开启详细输出模式
        handle_parsing_errors=True             # 开启解析错误处理
    )
    # 运行代理
    agent.run("现任中国羽毛球单打组主教练是哪个省的？用中文回答")
