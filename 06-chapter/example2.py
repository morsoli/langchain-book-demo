from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


tools = load_tools(["wikipedia","terminal"], llm=llm) 
agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                         verbose=True)


def get_tool_info():
    for tool in tools:
        print(f"工具名称: {tool.name}")
        print(f"工具描述: {tool.description}")

def get_agent_prompt():
    print(agent.agent.llm_chain.prompt.template)


if __name__ == "__main__":
    # get_tool_info()
    get_agent_prompt()
