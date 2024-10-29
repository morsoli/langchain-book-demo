from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate

example_prompt = PromptTemplate(input_variables=["input", "output"], template="问题: {input}\n{output}")
# 创建FewShotPromptTemplate实例
# 示例中包含了一些教模型如何回答问题的样本
template = FewShotPromptTemplate(
    examples=[
        {"input": "1+1等于多少?", "output": "2"},
        {"input": "3+2等于多少?", "output": "5"}
    ],
    example_prompt=example_prompt,
    input_variables=["input"],
    suffix="问题: {input}"
)
prompt = template.format(input="5-3等于多少?")


if __name__ == "__main__":
    print(prompt)