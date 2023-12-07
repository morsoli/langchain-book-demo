from langchain.chains import LLMMathChain
from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# 加载环境变量，通常用于配置文件中的API密钥等敏感信息
load_dotenv()

# 初始化一个基于ChatGPT的语言模型，设置模型和温度参数
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# 设置一个会话缓冲记忆体，用于存储和返回聊天历史
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# 初始化并加载数学工具，它将用于代理进行数学运算
tools = load_tools(["llm-math"], llm=llm) 

if __name__ == "__main__":
    # 初始化一个对话型代理，设置其使用的工具、语言模型、代理类型、最大迭代次数、记忆体和其他参数
    conversational_agent  = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, 
                                             max_iterations=5, memory=memory,verbose=True,handle_parsing_errors=True)

    # 运行代理，解决一个简单的数学问题
    print(conversational_agent.run("3加5等于几？"))

    # 运行代理，询问最后一个问过的问题是什么
    print(conversational_agent.run("我问的最后一个问题是什么?用中文回答"))
