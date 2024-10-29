from langchain_core.output_parsers import BaseOutputParser
from langchain_core.messages import HumanMessage
from langchain_community.llms.tongyi import Tongyi

llm = Tongyi()

text = "给生产杯子的公司取三个合适的中文名字,以逗号分隔符的方式输出"
messages = [HumanMessage(content=text)]
    
class CommaSeparatedListOutputParser(BaseOutputParser):
    """将LLMs输出解析为逗号分隔的列表"""

    def parse(self, text: str):
        """解析LLMs调用的输出"""
        return text.strip().split(",")

if __name__ == "__main__":
    llms_response = llm.invoke(text)
    print(CommaSeparatedListOutputParser().parse(llms_response))
    # 输出： ['杯子之家', '瓷杯工坊', '品质杯子']
