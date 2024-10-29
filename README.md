# langchain-book-demo
《Langchain编程：从入门到实践》书稿及演示代码仓库

## 第二版更新
1. 移除 Chain 编程接口内容介绍，对StuffDocumentsChain、ConversationalRetrievalChain等传统文档检索组件进行迁移https://python.langchain.com/docs/versions/migrating_chains/
2. 对ConversationBufferMemory、ConversationStringBufferMemory等传统记忆组件进行迁移https://python.langchain.com/docs/versions/migrating_memory/
3. 增加对LangChain扩展的Agent编排框架LangGraph库的详细介绍
4. 迁移至 Pydantic 2，移除langchain_core.pydantic_v1 https://python.langchain.com/docs/versions/v0_3/
5. langchain 已被拆分为以下组件包： langchain-core（包含涉及 LangChain 可运行性、可观测性工具以及重要抽象基实现（例如，聊天模型）的核心抽象。
） 、 langchain（包含通用代码，这些代码使用在 langchain-core 中定义的接口构建。此包适用于跨特定接口的不同实现通用性良好的代码。例如， create_tool_calling_agent 在支持工具调用功能的聊天模型中工作。） 、 langchain-community（社区维护的第三方集成。包含基于 langchain-core 中定义的接口实现的集成。由 LangChain 社区维护） 、 langchain-[partner]（作伙伴包是专门针对特别受欢迎的集成（例如， langchain-openai ， langchain-anthropic 等等）的包。通常，专用包会从更好的可靠性和支持中获益。
）
6. 代码示例和说明全面支持 LangChain 0.3