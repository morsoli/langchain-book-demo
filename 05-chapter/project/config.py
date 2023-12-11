import os
import glob
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model


current_file_path = os.path.abspath(__file__)
# 获取当前脚本所在目录的路径
current_directory = os.path.dirname(current_file_path)

# 存储嵌入向量和 Langchain 文档的目录
DB_DIR = os.path.join(current_directory, "database_store")

# 存储源文件的目录
DOCS_DIR = os.path.join(current_directory, "data")
FILE_PATH = glob.glob(DOCS_DIR + "/*")

# 模型名称
MODEL_NAME = "gpt-3.5-turbo" 

# 用于存储聊天历史的最大token数。
MAX_CHAT_HISTORY = 800

# 用于检索信息的 LLM 上下文的最大token数。
MAX_LLM_CONTEXT = 1200

# LLM 生成的最大token数。
MAX_LLM_GENERATION = 1000 

# 第一阶段文档检索的基本（小型）块大小。
BASE_CHUNK_SIZE = 100 

# 设置为 0 以禁用重叠。
CHUNK_OVERLAP = 0

# 最终检索（中型）块大小将为 BASE_CHUNK_SIZE * CHUNK_SCALE。
CHUNK_SCALE = 3  # 块缩放比例

WINDOW_STEPS = 3  # 窗口步数

# 窗口块的标记数将为 BASE_CHUNK_SIZE * WINDOW_SCALE。
WINDOW_SCALE = 18  # 窗口缩放比例

# BM25 检索器与 Chroma 向量存储检索器的权重比例。
RETRIEVER_WEIGHTS = 0.5, 0.5  # 检索器权重

# 第一次检索中检索的块数量将在 FIRST_RETRIEVAL_K 到 2*FIRST_RETRIEVAL_K 之间，因为使用了集成检索器。
FIRST_RETRIEVAL_K = 3  # 第一次检索块数量

# 第二次检索中检索的块数量将在 SECOND_RETRIEVAL_K 到 2*SECOND_RETRIEVAL_K 之间，因为使用了集成检索器。
SECOND_RETRIEVAL_K = 3  # 第二次检索块数量

# 第三个检索器的窗口数（大块）。
NUM_WINDOWS = 2  # 窗口数（大块数量）

REFINE_QA_TEMPLATE = """
对后续输入进行细化或重新表述，将其分解为少于 3 个异构的单跳查询，作为检索工具的输入。
如果后续输入是多跳、多步骤、复杂或比较性查询，并且与聊天历史和文档名称相关，则进行分解。否则保持后续输入不变。
输出格式必须严格遵循以下格式，每个查询只能包含一个文档名称：
```
1. One-hop standalone query
...
3. One-hop standalone query
...
```
数据库中的文档名称：
```
{database}
```
聊天历史：
```
{chat_history}
```
开始:
后续输入: {question}
One-hop standalone queries(s):
"""

DOCS_SELECTION_TEMPLATE = """
下面是一些验证过的来源和用户输入。如果你认为其中任何一条与用户输入相关，请列出所有可能的上下文编号。
```
{snippets}
```
输出格式必须如下所示，不得有其他内容。否则，你将输出[]:
[0, ..., n]

用户输入: {query}
"""

RETRIEVAL_QA_SYS = """你是由莫尔索设计的一个助手。
如果你认为以下信息与用户输入相关，请根据相关检索到的来源回答用户；否则，仅根据用户输入进行回复。"""

RETRIEVAL_QA_TEMPLATE = """
数据库中的文件名称:
```
{database}
```
聊天历史:
```
{chat_history}
```
验证过的来源:
```
{context}
```
用户输入: {question}
"""


RETRIEVAL_QA_CHAT_TEMPLATE = """
数据库中的文件名称:
```
{database}
```
聊天历史:
```
{chat_history}
```
验证过的来源:
```
{context}
```
"""


class PromptTemplates:
    def __init__(self):
        # 类的初始化方法，用于设置各种提示模板
        self.refine_qa_prompt = REFINE_QA_TEMPLATE
        self.docs_selection_prompt = DOCS_SELECTION_TEMPLATE
        self.retrieval_qa_sys = RETRIEVAL_QA_SYS
        self.retrieval_qa_prompt = RETRIEVAL_QA_TEMPLATE
        self.retrieval_qa_chat_prompt = RETRIEVAL_QA_CHAT_TEMPLATE

    def get_refine_qa_template(self, llm: str):
        """
        获取修正问答(prompt refinement)的模板。

        参数:
            llm (str): 使用的语言模型名称。

        返回值:
            PromptTemplate: 修正问答的模板对象。

        功能:
            - 根据语言模型的类型来调整修正问答的模板。
        """
        if "llama" in llm.lower():
            temp = f"[INST] {self.refine_qa_prompt} [/INST]"
        else:
            temp = self.refine_qa_prompt

        return PromptTemplate(
            input_variables=["database", "chat_history", "question"],
            template=temp,
        )

    def get_docs_selection_template(self):
        """
        获取文档选择的模板。

        返回值:
            PromptTemplate: 文档选择的模板对象。

        功能:
            - 返回用于文档选择的模板。
        """
        return PromptTemplate(
            input_variables=["snippets", "query"],
            template=self.docs_selection_prompt,
        )

    def get_retrieval_qa_template_selector(self):
        """
        获取检索问答(retrieval QA)的模板选择器。

        参数:
            llm (str): 使用的语言模型名称。

        返回值:
            ConditionalPromptSelector: 检索问答的模板选择器对象。

        功能:
            - 根据语言模型的类型来调整检索问答的模板。
            - 对于不同的模型类型，使用不同的提示格式。
            - 提供一个条件选择器，用于在聊天模型和非聊天模型间选择合适的模板。
        """
        temp = f"{self.retrieval_qa_sys}\n{self.retrieval_qa_prompt}"
        messages = [
            SystemMessagePromptTemplate.from_template(
                f"{self.retrieval_qa_sys}\n{self.retrieval_qa_chat_prompt}"
            ),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]

        prompt_temp = PromptTemplate(
            template=temp,
            input_variables=["database", "chat_history", "context", "question"],
        )
        prompt_temp_chat = ChatPromptTemplate.from_messages(messages)

        return ConditionalPromptSelector(
            default_prompt=prompt_temp,
            conditionals=[(is_chat_model, prompt_temp_chat)],
        )