from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_community.chat_models import ChatTongyi
from langsmith import Client

def dataset_test(unique_id):
    # 创建基准数据集
    client = Client()
    outputs = ["2", "6"]
    dataset_name = f"agent-qa-{unique_id}"
    dataset = client.create_dataset(dataset_name, description="agent测试数据集")

    for query, answer in zip(inputs, outputs):
        client.create_example(inputs={"text": query}, outputs={"output": answer}, dataset_id=dataset.id)

    # 使用LangSmith评估代理
    evaluation_results = client.run_on_dataset(dataset_name, agent)
    print(evaluation_results)
    
    # 查看评估结果
    project_name = f"runnable-agent-test-{unique_id}"
    runs = client.list_runs(project_name=project_name)
    for run in runs:
        print("评估结果: ", run)

@tool
def math_add(a: int, b: int) -> int:
    "计算两数之和"
    return a + b
    
if __name__ == "__main__":
    template = "回答下列问题，尽量使用工具"
    human_template = "{text}"
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])
    model = ChatTongyi(temperature=0)
    model_with_tools = model.bind_tools([math_add])
    # 创建代理
    agent = chat_prompt | model_with_tools | StrOutputParser()
    
    inputs = ["1+1等于几", "3+3等于几?"]
    config = RunnableConfig(max_concurrency=3)
    # 运行代理并记录结果
    results = agent.batch([{"text": x} for x in inputs], config=config)
    print(results)
    dataset_test("001")
    
    
