from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek

def test():
    # 创建一个 PromptTemplate 实例，用于生成提示词。
    # 这里的模板是为生产特定产品的公司取名。
    prompt = PromptTemplate.from_template(
        "给生产{product}的公司取一个名字。"
    )

    # 创建一个 Runnable 链，包括上述提示词模板、聊天模型和字符串输出解析器。
    # 这个链首先生成提示词，然后通过 ChatDeepSeek 聊天模型进行处理，最后通过 StrOutputParser 转换成字符串。
    runnable = prompt | ChatDeepSeek(model="deepseek-chat") | StrOutputParser()

    # 打印输入模式的 JSONSchema
    print(runnable.input_schema.model_json_schema())

    # 打印输出模式的 JSONSchema。这说明了 Runnable 执行后的输出数据结构。
    print(runnable.output_schema.model_json_schema())


if __name__ == "__main__":
    test()