# 使用此文件创建您自己的工具
import logging

from langchain import LLMChain, PromptTemplate
from langchain.agents import Tool
# 根据您的具体实现替换 Steamship 和 OpenAI 的引用
# 从 langchain 导入 LLMChain 和 PromptTemplate

NAME = "我的工具"

DESCRIPTION = """
在需要创建待办事项列表时有用。
输入：需要创建待办事项列表的目标。
输出：针对该目标的待办事项列表。请明确说明目标是什么！
"""

PROMPT = """
你是一个规划师，擅长为给定目标制定待办事项列表。
为这个目标制定一个待办事项列表：{objective}"
"""

class MyTool(Tool):
    """用于管理待办事项列表的工具。"""

    def __init__(self):
        super().__init__(
            name=NAME, func=self.run, description=DESCRIPTION
        )

    def _get_chain(self):
        todo_prompt = PromptTemplate.from_template(PROMPT)
        # 请在这里替换为您的 LLM 实现
        return LLMChain(llm=OpenAI(temperature=0), prompt=todo_prompt)

    @property
    def is_single_input(self) -> bool:
        """工具是否只接受单一输入。"""
        return True

    def run(self, prompt: str, **kwargs) -> str:
        """响应 LLM 提示。"""
        chain = self._get_chain()
        return chain.predict(objective=prompt)
