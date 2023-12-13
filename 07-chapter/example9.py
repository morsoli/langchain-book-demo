from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.utilities.google_search import GoogleSearchAPIWrapper
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 创建搜索工具
search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="在你需要进行搜索式提问时非常实用",
    )
]

# 创建代理的提示模板
prefix = "请与人类进行对话，并尽可能最好地回答以下问题。你可以访问以下工具:"
suffix = "开始!\n{chat_history}\n问题: {input}\n{agent_scratchpad}"
prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

# 创建内存
message_history = RedisChatMessageHistory(
    url="redis://localhost:6379/0", ttl=600, session_id="my-session"
)
memory = ConversationBufferMemory(memory_key="chat_history")

# 构建LLMChain并创建代理
llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)

if __name__ == "__main__":
    # 运行代理
    agent_chain.run(input="中国有多少人?")
    agent_chain.run(input="它的国歌叫什么")