from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import FileChatMessageHistory


from agent.tools.image import GenerateImageTool
from agent.tools.search import SearchTool
from agent.tools.speech import GenerateVoiceTool

from utils import file_cache_dir

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = [SearchTool(), GenerateImageTool(), GenerateVoiceTool()]

# 定义提示模板
prefix = "Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"
suffix = "Begin!\n{chat_history}\nQuestion: {input}\n{agent_scratchpad}"
prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

def langchain_agent(user, query):
    # 创建具有记忆的代理执行器
    message_history = FileChatMessageHistory(file_cache_dir/user)
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history)   
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory
    )
    return agent_chain.run(query)


if __name__ == "__main__":
    print(langchain_agent("test", ""))