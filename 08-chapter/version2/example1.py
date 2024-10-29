from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage
from langchain_community.chat_models.tongyi import ChatTongyi

prompt = ChatPromptTemplate.from_template("给做{product}的公司起一个名字,不超过5个字")

class MyConstructorCallbackHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized, prompts, **kwargs):
        print("构造器回调：链开始运行")

    def on_chain_end(self, response, **kwargs):
        print("构造器回调：链结束运行")
        
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("构造器回调：模型开始运行")

    def on_llm_end(self, response, **kwargs):
        print("构造器回调：模型结束运行")

class MyRequestCallbackHandler(BaseCallbackHandler):
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        print(f"请求回调：{serialized.get('name') if serialized else ""}开始运行")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        print(f"请求回调：结束运行 {outputs}")

    def on_llm_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs
    ) -> None:
        print("请求回调：模型开始运行")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        print("请求回调：模型结束运行")

def constructor_test():
    callbacks = [MyConstructorCallbackHandler()]
    # 在构造器中使用回调处理器
    llm = ChatTongyi(callbacks=callbacks)
    chain = prompt | llm
    # 这次运行将使用构造器中定义的回调
    chain.invoke({"product": "杯子"})
    
def request_test():
    callbacks = [MyRequestCallbackHandler()]
    # 初始化 Chain，不在构造器中传递回调处理器
    llm = ChatTongyi()
    chain = prompt | llm
    # 在请求中使用回调处理器
    chain.invoke({"product": "杯子"}, config={"callbacks": callbacks})
    
if __name__ == "__main__":
    #constructor_test()
    request_test()