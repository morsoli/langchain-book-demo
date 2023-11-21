from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


template="你是一个翻译助手，可以将 {input_language} 翻译为 {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

human_template="{talk}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])


if __name__ == "__main__":
    print(chat_prompt.format_prompt(input_language="中文", output_language="英语", talk="我喜欢编程").to_messages())
    