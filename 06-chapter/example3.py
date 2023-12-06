from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def record_recommendations(actions: str) -> str:
    "记录类似健身请求的行动和反馈，以便未来使用"
    # 信息记录到数据库逻辑
    return "插入成功"

@tool
def search_recommendations(query: str) -> str:
    "为类似的健身请求搜索相关的行动和反馈"
    # 向量数据库或外部知识库检索逻辑
    
    results_list = [[["搜索了 '跑步爱好者的健身计划'，找到了 '跑步者的终极力量训练计划：7个高效练习'", '该计划主要针对力量训练，可能不适用于所有跑步者。建议加入一些有氧运动和灵活性训练。'], 
                     ["搜索了 '跑步者的健身计划'，找到了 '跑步者的核心锻炼：6个基本练习'", '该计划包含对跑步者有益的核心锻炼。不过，建议也加入有氧运动和灵活性训练。']]]
    return "按照 [[action, recommendation], ...] 的格式，继续列出相关的行动和反馈列表:\n" + str(results_list)



llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = load_tools(["google-search"], llm=llm)
tools.extend([record_recommendations, search_recommendations])

def create_prompt(info: str) -> str:
    prompt_start = (
        "根据下面提供的用户信息及其兴趣，作为健身教练来执行相应的动作。\n\n"+
        "用户提供的信息：\n\n"
    )
    prompt_end = (
        "\n\n1. 利用用户信息来搜索并复查之前的行动和反馈（如果有的话）\n"+
        "2. 在给出回答之前，务必使用先把你采取的行动和反馈记录到数据库中，以便未来能提供更好的健身计划。\n"+
        "3. 在为用户研究健身计划时，要记住之前的行动和反馈，并据此来回答用户\n"
    )   
    return prompt_start + info + prompt_end

def run_agent(info):
    agent = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    agent.run(input=create_prompt(info))
    
if __name__ == "__main__":
    run_agent("我是小李，今年23岁，喜欢跑步")