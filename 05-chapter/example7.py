from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema import Document

# 创建文档提示模板
doc_prompt = PromptTemplate.from_template("{page_content}")

# 构建 StuffDocumentsChain
chain = (
    {
        "content": lambda docs: "\n\n".join(
            format_document(doc, doc_prompt) for doc in docs
        )
    }
    | PromptTemplate.from_template("总结下面的内容:\n\n{content}")
    | ChatOpenAI()
    | StrOutputParser()
)

# 示例文本
text = """
2022 年 11 月 30 日，OpenAI 正式发布 ChatGPT，在短短一年时间里，ChatGPT 不仅成为了生成式 AI 领域的热门话题，
更是开启了新一轮技术浪潮，每当 OpenAI 有新动作，就可以占据国内外各大科技媒体头条。
从最初的 GPT-3.5 模型，到如今的 GPT-4.0 Turbo 模型，OpenAI 的每一次更新都不断拓宽着我们对于人工智能可能性的想象，
最开始，ChatGPT 只是通过文字聊天进行互动，而现在，已经能够借助 GPT-4V 解说足球视频。
"""

# 将文本分割成文档
docs = [
    Document(
        page_content=split,
        metadata={"source": "https://mp.weixin.qq.com/s/Y0t8qrmU5y6H93N-Z9_efw"},
    )
    for split in text.split()
]


if __name__ == "__main__":
    # 调用链并打印结果
    print(chain.invoke(docs))