from typing import Optional, Union
from math import sqrt, cos, sin
from langchain.tools import BaseTool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 工具描述
descriptions = (
    "当您需要计算直角三角形的斜边长度时使用此工具，"
    "给定三角形的一边或两边和/或一个角度（以度为单位）。"
    "使用此工具时，您必须提供以下参数中的至少两个："
    "['adjacent_side', 'opposite_side', 'angle']。"
)

class HypotenuseTool(BaseTool):
    name = "Hypotenuse calculator"  # 工具名称
    description = descriptions  # 工具描述
    
    def _run(
        self,
        adjacent_side: Optional[Union[int, float]] = None,
        opposite_side: Optional[Union[int, float]] = None,
        angle: Optional[Union[int, float]] = None
    ):
        # 检查值
        if adjacent_side and opposite_side:
            # 如果提供了邻边和对边，计算斜边
            return sqrt(float(adjacent_side)**2 + float(opposite_side)**2)
        elif adjacent_side and angle:
            # 如果提供了邻边和角度，使用余弦计算斜边
            return adjacent_side / cos(float(angle))
        elif opposite_side and angle:
            # 如果提供了对边和角度，使用正弦计算斜边
            return opposite_side / sin(float(angle))
        else:
            # 如果参数不足，返回错误信息
            return "无法计算三角形的斜边。需要提供两个或更多的参数：`adjacent_side`, `opposite_side`, 或 `angle`。"
    
    def _arun(self, query: str):
        # 异步运行，计算逻辑同理
        pass


if __name__ == "__main__":
    tools = [HypotenuseTool()]
    agent = initialize_agent(tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=3,
        handle_parsing_errors=True)
    agent.run("如果我有一个三角形，其两边的长度分别是3厘米和3厘米，那么斜边的长度是多少？")
    agent.run("如果我有一个三角形，对边长度为3厘米，角度为45度，那么斜边的长度是多少？")
    agent.run("如果我有一个三角形，邻边长度为3厘米，角度为45度，那么斜边的长度是多少？")