from enum import Enum
from typing import List, Optional

from langchain.agents import Tool, initialize_agent, AgentType, AgentExecutor
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import PyPDFLoader, YoutubeLoader
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage, Document
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from pydantic import Field
from libs.cache import KeyValueStore
from libs.usage import UsageTracker

TEMPERATURE = 0.2
VERBOSE = True
PERSONALITY = """You are a Youtuber 

You NEVER:
- Say you're an assistant 
- Call me Morty

To get access to personal anecdotes and advise you always look up information using the knowledge_base tool.
You ALWAYS look up information in the knowledge_base before responding.

The knowledge base will give you back sources. Make sure to end your answers with the correct sources when you've used them.
Cite sources as follows: Source("your_source")
"""

class ChatbotConfig():
    # 配置类，用于定义 LangChain Slackbot 的配置选项
    elevenlabs_api_key: Optional[str] = Field(
        default="", description="Optional API KEY for ElevenLabs Voice Bot"
    )
    elevenlabs_voice_id: Optional[str] = Field(
        default="", description="Optional voice_id for ElevenLabs Voice Bot"
    )
    use_gpt4: bool = Field(
        True,
        description="If True, use GPT-4. Use GPT-3.5 if False. "
        "GPT-4 generates better responses at higher cost and latency.",
    )
    bot_token: Optional[str] = Field(
        "",
        description="Your telegram bot token.\nLearn how to create one here: "
        "https://github.com/steamship-packages/langchain-agent-production-starter/blob/main/docs/register-telegram-bot.md",
    )
    n_free_messages: Optional[int] = Field(
        -1, description="Number of free messages assigned to new users."
    )
    api_base: str = Field(
        "https://api.telegram.org/bot", description="The root API for Telegram"
    )
    


class FileType(str, Enum):
    # 枚举类，定义了支持的文件类型
    YOUTUBE = "YOUTUBE"
    PDF = "PDF"
    WEB = "WEB"
    TEXT = "TEXT"


class MyBot:
    # LangChain Slackbot 主类
    config: ChatbotConfig

    def __init__(self, **kwargs):
        # 初始化函数
        self.model_name = "gpt-4" if self.config.use_gpt4 else "gpt-3.5-turbo"
        self.store = KeyValueStore(self.client, store_identifier="config")
        bot_token = self.store.get("bot_token")
        if bot_token:
            bot_token = bot_token.get("token")
        self.config.bot_token = bot_token or self.config.bot_token

        self.usage = UsageTracker(
            self.client, n_free_messages=self.config.n_free_messages
        )

    def chunk(self, text: List[Document], chunk_size: int = 1_000, chunk_overlap: int = 300) -> List[Document]:
        # 分块处理长文本
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(text)

    def load(self, file_type: FileType, content_or_url: str) -> List[Document]:
        # 加载不同类型的文件
        loaders = {
            FileType.YOUTUBE: YoutubeLoader.from_youtube_url(content_or_url, add_video_info=True),
            FileType.PDF: PyPDFLoader(content_or_url),
            # 其他文件类型加载器...
        }
        return loaders[file_type].load()

    def index(self, chunks: List[Document]):
        self.get_vectorstore().add_documents(chunks)
        # 索引文档，用于后续的搜索和检索
        # 这里可以用伪代码替换原有的 Steamship 索引逻辑
        # 示例：index_documents_in_vectorstore(chunks)

    def get_agent(self, chat_id: str) -> AgentExecutor:
        # 获取 LangChain 代理执行器
        # 定义使用的语言模型、工具和记忆体
        llm = ChatOpenAI(model_name=self.model_name, temperature=TEMPERATURE, verbose=VERBOSE)
        tools = self.get_tools(chat_id=chat_id)
        memory = self.get_memory(chat_id=chat_id)
        return initialize_agent(tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=VERBOSE, memory=memory, agent_kwargs={"system_message": SystemMessage(content=PERSONALITY), "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")]})

    def get_memory(self, chat_id: str):
        # 获取对话的记忆体，用于维持对话上下文
        memory = ConversationBufferMemory(
            memory_key="memory",
            chat_memory=ChatMessageHistory(key=f"history-{chat_id or 'default'}"),
            return_messages=True,
        )
        return memory

    def get_tools(self, chat_id: str) -> List[Tool]:
        # 定义和返回 LangChain 工具列表
        # 这里的 RetrievalQAWithSourcesChain 可以用于从记忆体中检索信息
        qa = RetrievalQAWithSourcesChain.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-4", temperature=0, verbose=VERBOSE),
            chain_type="stuff",
            retriever=self.get_vectorstore().as_retriever(k=3),
        )

        qa_tool = Tool(
            name="knowledge_base",
            func=lambda x: qa({"question": x}, return_only_outputs=False),
            description="always use this to answer questions. Input should be a fully formed question."
        )

        return [qa_tool]
        # 可以根据需要添加更多工具

    def get_vectorstore(self) -> VectorStore:
        pass
        # 获取向量存储
        # 此处需要根据实际情况定义向量存储的实现
        # 示例：return MyVectorStore()
        
    def voice_tool(self) -> Optional[Tool]:
        """Return tool to generate spoken version of output text."""
        return None
        # return GenerateSpeechTool(
        #     client=self.client,
        #     voice_id=self.config.elevenlabs_voice_id,
        #     elevenlabs_api_key=self.config.elevenlabs_api_key,
        # )
