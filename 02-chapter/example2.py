from langchain.prompts.chat import ChatPromptTemplate


template = "你是一个能将{input_language}翻译成{output_language}的助手。"
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])


if __name__ == "__main__":
    print(chat_prompt.format_messages(input_language="汉语", output_language="英语", text="我爱编程。"))
    