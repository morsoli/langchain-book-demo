from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import  Tool
import json
from dotenv import load_dotenv

load_dotenv()


def get_current_weather(location: str, unit: str = "celsius"):
    """根据输入地点获取天气情况"""
    weather_info = {
        "location": location,
        "temperature": "28",
        "unit": unit,
        "forecast": ["温暖", "晴朗"],
    }
    return json.dumps(weather_info)

tools = [
    Tool.from_function(
        name="get_current_weather",
        func=get_current_weather,
        description="""根据输入地点获取天气情况""",
    ),
]

if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    mrkl = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
    mrkl.run("今天北京天气怎么样?")