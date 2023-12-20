from langchain.agents import initialize_agent
from langchain.agents import load_tools
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

def dataset_test(unique_id):
    # 创建基准数据集
    client = Client()
    outputs = ["2", "6"]
    dataset_name = f"agent-qa-{unique_id}"
    dataset = client.create_dataset(dataset_name, description="agent测试数据集")

    for query, answer in zip(inputs, outputs):
        client.create_example(inputs={"input": query}, outputs={"output": answer}, dataset_id=dataset.id)

    # 使用LangSmith评估代理
    evaluation_results = client.run_on_dataset(dataset_name, agent)
    print(evaluation_results)
    
    # 查看评估结果
    project_name = f"runnable-agent-test-{unique_id}"
    runs = client.list_runs(project_name=project_name)
    for run in runs:
        print("评估结果: ", run)

if __name__ == "__main__":
    inputs = ["1+1等于几", "3+3等于几?"]
    # 创建代理
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    tools = load_tools(["llm-math"], llm=llm) 
    agent  = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                            handle_parsing_errors=True)
    # 运行代理并记录结果
    results = agent.batch([{"input": x} for x in inputs], return_exceptions=True)
    print(results)
    dataset_test("001")
    
    
