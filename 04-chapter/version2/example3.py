from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate

def test():
    # 这个模型用于处理聊天或对话类的语言生成任务。
    model = ChatTongyi()

    # 创建一个 PromptTemplate 实例。
    # 这里的模板用于生成一个故事，其中故事类型由变量 {story_type} 决定。
    prompt = PromptTemplate.from_template(
        "讲一个{story_type}的故事。"
    )

    # 创建一个处理链（Runnable），包含上述提示词模板和 ChatOpenAI 聊天模型。
    # 这个链将使用 PromptTemplate 生成提示词，然后通过 ChatOpenAI 模型进行处理。
    chain = prompt | model
    # 使用流式处理生成故事。
    # 这里传入的 story_type 为 "悲伤"，模型将根据这个类型生成一个悲伤的故事。
    # 这个方法返回一个迭代器，可以逐步获取模型生成的每个部分。
    chain.astream()
    for s in chain.stream({"story_type": "悲伤"}):
        # 打印每个生成的部分，end="" 确保输出连续，无额外换行。
        print(s.content, end="", flush=True)

if __name__ == "__main__":
    test()