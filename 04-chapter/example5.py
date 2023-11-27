from langchain.prompts import PromptTemplate
from langchain.llms.openai import OpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv 

# 加载环境变量
load_dotenv()


if __name__ =="__main__":
    prompt_template = "给做 {product} 的公司起一个名字?"

    llm = OpenAI(temperature=0)
    llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template)
    )
    print(llm_chain("儿童玩具"))
    print(llm_chain.run("儿童玩具"))
    llm_chain.apply([{"product":"儿童玩具"}])
    llm_chain.generate([{"product":"儿童玩具"}])
    llm_chain.predict(product="儿童玩具")