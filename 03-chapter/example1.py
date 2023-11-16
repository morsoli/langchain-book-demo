from langchain.prompts import PromptTemplate
# 定义模板
template = PromptTemplate.from_template("翻译这段文字: {text}，风格: {style}")
# 定义变量style和text的内容
template.format(text="我爱编程", style="诙谐有趣")


if __name__ == "__main__":
    print(template.format(text="我爱编程", style="诙谐有趣"))
