from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi


if __name__ =="__main__":
    prompt_template = "给做 {product} 的公司起一个名字，无需做额外输出"
    prompt=PromptTemplate.from_template(prompt_template)

    llm = Tongyi()
    chain = prompt | llm
    print(chain.invoke("儿童玩具"))