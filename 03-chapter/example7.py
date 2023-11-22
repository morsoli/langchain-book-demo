from typing import List
import json
import re
from typing import Type, TypeVar
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, ValidationError, Field
from langchain.schema import BaseOutputParser, OutputParserException

# 加载环境变量
load_dotenv()

CUSTOM_FORMAT_INSTRUCTIONS  = """输出内容需要被格式化成一个 JSON 实例，这个实例应该符合下面提到的 JSON 模式。
输出模式如下：
```
{schema}
```"""

T = TypeVar("T", bound=BaseModel)

class CustomOutputParser(BaseOutputParser[BaseModel]):
    pydantic_object: Type[T]
    
    def parse(self, text: str) -> BaseModel:
        """
        解析文本到Pydantic模型。

        Args:
            text: 要解析的文本。

        Returns:
            Pydantic模型的一个实例。
        """
        try:
            # 贪婪搜索第一个json候选。
            match = re.search(
                r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL
            )
            json_str = ""
            if match:
                json_str = match.group()
            json_object = json.loads(json_str, strict=False)
            return self.pydantic_object.parse_obj(json_object)
        
        except (json.JSONDecodeError, ValidationError) as e:
            name = self.pydantic_object.__name__
            msg = f"从输出中解析{name}失败 {text}。错误信息: {e}"
            raise OutputParserException(msg, llm_output=text)

    def get_format_instructions(self) -> str:
        """
        获取格式说明。

        Returns:
            格式说明的字符串。
        """
        schema = self.pydantic_object.schema()

        # 移除不必要的字段。
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # 确保json在上下文中格式正确（使用双引号）。
        schema_str = json.dumps(reduced_schema)
        
        return CUSTOM_FORMAT_INSTRUCTIONS.format(schema=schema_str)


    @property
    def _type(self) -> str:
        """
        获取解析器类型。

        Returns:
            解析器的类型字符串。
        """
        return "custom output parser"
    

if __name__ == "__main__":
    # 定义花费记录的数据模型
    class ExpenseRecord(BaseModel):
        amount: float = Field(description="花费金额")
        category: str = Field(description="花费类别")
        date: str = Field(description="花费日期")
        description: str = Field(description="花费描述")

    # 创建Pydantic输出解析器实例
    parser = CustomOutputParser(pydantic_object=ExpenseRecord)

    # 定义获取花费记录的提示模板
    expense_template = '''
    请将这些花费记录在我的预算中。
    我的花费记录是：{query}
    格式说明：
    {format_instructions}
    '''

    # 使用提示模板创建实例
    prompt = PromptTemplate(
        template=expense_template,
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    ) 

    # 格式化提示
    _input = prompt.format_prompt(query="昨天,我在超市花了45元买日用品。晚上我又花了20元打车。")

    # 创建OpenAI模型实例
    model = OpenAI(model_name="text-davinci-003", temperature=0)

    # 使用模型处理格式化后的提示
    output = model(_input.to_string())

    # 解析输出结果
    expense_record = parser.parse(output)
    # 遍历并打印花费记录的各个参数
    for parameter in expense_record.__fields__:
        print(f"{parameter}: {expense_record.__dict__[parameter]},  {type(expense_record.__dict__[parameter])}")
