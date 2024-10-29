from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_models.tongyi import ChatTongyi

# 创建文档提示模板
llm = ChatTongyi()
prompt = ChatPromptTemplate.from_template("总结下面的内容: {context}")


# 示例文本
text = """
2022 年 11 月 30 日，OpenAI 正式发布 ChatGPT，在短短一年时间里，ChatGPT 不仅成为了生成式 AI 领域的热门话题，更是开启了新一轮技术浪潮，每当 OpenAI 有新动作，就可以占据国内外各大科技媒体头条。从最初的 GPT-3.5 模型，到如今的 GPT-4.0 Turbo 模型，OpenAI 的每一次更新都不断拓宽着我们对于人工智能可能性的想象，最开始，ChatGPT 只是通过文字聊天进行互动，而现在，已经能够借助 GPT-4V 解说足球视频。
文字是思想的载体，第一次看到 ChatGPT 的演示效果，我就被震撼到了，看着对话框中的文字逐个跳现，流畅的内容表达、尽情展现想象力（虽然后面了解到实质是概率模型），这与以往任何智能对话机器人截然不同，真正展示了智能的可能性。随后，我便开始搜寻一切关于 ChatGPT 的信息，频繁刷新 reddit 上的 ChatGPT 话题讨论，检索 X 平台的 ChatGPT 关键词，查看科技媒体是否有相关报道......
在接下来的一个月里，我既兴奋又焦虑，兴奋源于每当有空闲时间，我就能与 ChatGPT 这款“真正的人工智能”进行对话；而焦虑则缘于工作时总忍不住去刷新新闻，生怕错过其他用户分享的新玩法展示和有趣的提示词，当然彼时在国内没引起太大的关注，微信指数、百度指数以及新浪指数也只有浅浅的波动。
"""

# 将文本分割成文档
docs = [
    Document(
        page_content=split,
        metadata={"source": "https://mp.weixin.qq.com/s/Y0t8qrmU5y6H93N-Z9_efw"},
    )
    for split in text.split("\n")
]

if __name__ == "__main__":
    chain = create_stuff_documents_chain(llm, prompt)
    # 调用链并打印结果
    print(chain.invoke({"context": docs}))