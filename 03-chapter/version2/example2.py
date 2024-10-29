import json
from pydantic import BaseModel, field_validator
from langchain_core.prompts import StringPromptTemplate

delimiter="####"

PROMPT = f"""将每个用户的信息用{delimiter}字符分割，并按照下面示例提取姓名，职业和爱好信息。
示例如下:
"""

class PersonInfoPromptTemplate(StringPromptTemplate, BaseModel):
    """一个自定义的提示词模板, 用于生成关于人物的JSON格式信息。"""
    
    @field_validator("input_variables")
    def validate_input_variables(cls, v):
        """验证输入变量是否正确。"""
        if "name" not in v:
            raise ValueError("name must be in input_variable.")
        if "occupation" not in v:
            raise ValueError("occupation must be in input_variable.")
        if "fun_fact" not in v:
            raise ValueError("fun_fact must be in input_variable.")
        return v

    def format(self, **kwargs) -> str:
        """格式化输入并生成JSON格式的输出。"""
        person_info = {
            "name": kwargs.get("name"),
            "occupation": kwargs.get("occupation"),
            "fun_fact": kwargs.get("fun_fact")
        }
        return PROMPT+json.dumps(person_info, ensure_ascii=False)

    def _prompt_type(self):
        return "person-info"


if __name__ == "__main__":
    # 初始化模板实例
    person_info_template = PersonInfoPromptTemplate(input_variables=["name", "occupation", "fun_fact"])
    # 生成JSON格式的提示
    prompt_output = person_info_template.format(
        name="张三",
        occupation="软件工程师",
        fun_fact="喜欢攀岩"
    )
    print(prompt_output)
