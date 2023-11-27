from typing import Any, Dict, List, Optional

from langchain.pydantic_v1 import Extra

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.prompts.base import BasePromptTemplate

class MyCustomChain(Chain):
    prompt: BasePromptTemplate
    llm: BaseLanguageModel
    output_key: str = "text"

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """pompt中的动态变量
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """允许直接输出的动态变量.
        """
        return [self.output_key]
    
    # 同步调用
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # 下面是一个自定义逻辑实现
        prompt_value = self.prompt.format_prompt(**inputs)
        # 调用一个语言模型或另一个链时，传递一个回调处理。这样内部运行可以通过这个回调（进行逻辑处理）。
        response = self.llm.generate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )
        # 回调出发时的日志输出
        if run_manager:
            run_manager.on_text("Log something about this run")

        return {self.output_key: response.generations[0][0].text}
    # 异步调用
    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        prompt_value = self.prompt.format_prompt(**inputs)
        response = await self.llm.agenerate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )
        if run_manager:
            await run_manager.on_text("Log something about this run")

        return {self.output_key: response.generations[0][0].text}

    @property
    def _chain_type(self) -> str:
        return "my_custom_chain"

if __name__ == "__main__":
    from langchain.prompts import PromptTemplate
    from langchain.llms.openai import OpenAI
    from dotenv import load_dotenv 

    # 加载环境变量
    load_dotenv()
    
    prompt_template = "给生产 {product} 的公司起一个名字。"
    llm = OpenAI(temperature=0)
    custom_chain = MyCustomChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))
    print(custom_chain("杯子"))