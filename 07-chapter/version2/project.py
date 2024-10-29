import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# 导入基于时间加权的向量存储检索器
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_core.memory import BaseMemory
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel

# 导入时间模拟工具
from langchain_core.utils import mock_now
from langchain_community.chat_models.tongyi import ChatTongyi

# 自定义的智能代理记忆
class CustomAgentMemory(BaseMemory):

    llm: BaseLanguageModel
    # 检索相关记忆的检索器
    memory_retriever: TimeWeightedVectorStoreRetriever
    # 是否输出详细信息的
    verbose: bool = False
    # 智能代理当前计划，一个字符串列表
    current_plan: List[str] = []
    # 与记忆重要性关联的权重因素，当这个数值偏低，表明它与记忆的相关度及时效性相比较显得不那么重要
    importance_weight: float = 0.15
    # 追踪近期记忆的“重要性”累计值
    aggregate_importance: float = 0.0
    # 反思的阈值，当近期记忆的“重要性”累计值一旦达到反思的阈值，便触发反思过程
    reflection_threshold: Optional[float] = None
    # 最大令牌限制
    max_tokens_limit: int = 1200
    # 查询内容的键
    queries_key: str = "queries"
    # 最近记忆的令牌的键
    most_recent_memories_token_key: str = "recent_memories_token"
    # 添加记忆的键
    add_memory_key: str = "add_memory"
    # 相关记忆的键
    relevant_memories_key: str = "relevant_memories"
    # 简化的相关记忆的键
    relevant_memories_simple_key: str = "relevant_memories_simple"
    # 最近记忆的键
    most_recent_memories_key: str = "most_recent_memories"
    # 当前时间的键
    now_key: str = "now"
    # 是否触发反思的标志
    reflecting: bool = False
    
    @property
    def memory_variables(self) -> List[str]:
        pass

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        pass

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        pass

    def clear(self) -> None:
        """
        清除内存内容。
        """
        pass
    
    def invoke(self, prompt, input_var: dict):
        chain = prompt | ChatTongyi() | StrOutputParser()
        return chain.invoke(input_var)

    # 将一个由换行符分隔的字符串转换为一个字符串列表
    @staticmethod
    def _parse_list(text: str) -> List[str]:
        lines = re.split(r"\n", text.strip())  # 根据换行符分割文本并去除首尾空白
        lines = [line for line in lines if line.strip()]  # 移除空行
        # 移除行首的数字和点，然后去除首尾空白
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    def _get_topics_of_reflection(self, last_k: int = 50) -> List[str]:
        """
        返回和最近记忆内容最相关的 3 个高级问题。
        参数 last_k：采样的最近的记忆数量，默认为50。
        返回值：一个字符串列表，包含三个问题。
        """
        # 创建一个提示模板，询问基于给定观察可以回答的三个最相关的高级问题。
        prompt = PromptTemplate.from_template(
            "```{observations}\n```\n 基于上述信息，提出和这些内容最为相关的3个高级问题。请将每个问题分别写在新的一行上。"

        )
        # 从内存检索器中获取最近的记忆，并转换成字符串形式
        observations = self.memory_retriever.memory_stream[-last_k:]
        observation_str = "\n".join(
            [self._format_memory_detail(o) for o in observations]
        )
        # 运行提示模板并获取结果
        result = self.invoke(prompt, {"observations": observation_str})
        # 将结果解析为列表并返回
        return self._parse_list(result)

    def _get_insights_on_topic(
        self, topic: str, now: Optional[datetime] = None
    ) -> List[str]:
        """
        基于与反思主题相关的记忆生成见解
        参数 topic：反思的主题
        参数 now：可选的当前时间，用于检索记忆
        返回值：一个字符串列表，包含生成的见解
        """
        # 创建一个提示模板，基于相关记忆生成针对特定问题的五个高级新见解。
        prompt = PromptTemplate.from_template(
            "关于‘{topic}’的陈述```\n{related_statements}\n```\n"
            "根据上述陈述，提出与解答下面这个问题最相关的 5 个高级见解。"
            "请不要包含与问题无关的见解，请不要重复已经提出的见解。"
            "问题：{topic}"
            "（示例格式：见解（基于1、5、3的原因）)"
        )

        # 从内存检索器中获取与主题相关的记忆
        related_memories = self.fetch_memories(topic, now=now)
        # 格式化相关记忆为字符串
        related_statements = "\n".join(
            [
                self._format_memory_detail(memory, prefix=f"{i+1}. ")
                for i, memory in enumerate(related_memories)
            ]
        )
        # 运行提示模板并获取结果
        result = self.invoke(prompt, {"topic": topic, "related_statements": related_statements})
        # 将结果解析为列表并返回。
        return self._parse_list(result)

    
    def pause_to_reflect(self, now: Optional[datetime] = None) -> List[str]:
        """
        反思最近的观察并生成“见解”
        参数 now：可选的，表示当前时间的datetime对象
        返回值：一个包含新见解的字符串列表
        """
        # 如果处于详细模式，记录日志信息表示正在进行反思
        if self.verbose:
            print("角色正在进行反思")
        # 初始化一个新见解的列表
        new_insights = []
        # 获取反思的主题
        topics = self._get_topics_of_reflection()
        # 遍历每个主题，生成见解，并添加到内存中
        for topic in topics:
            insights = self._get_insights_on_topic(topic, now=now)
            for insight in insights:
                self.add_memory(insight, now=now)
            new_insights.extend(insights)
        # 返回新生成的见解列表
        return new_insights

    def _score_memory_importance(self, memory_content: str) -> float:
        """
        为给定的记忆打分，评估其绝对重要性
        参数 memory_content：记忆内容
        返回值：记忆重要性的分数，为浮点数
        """
        prompt = PromptTemplate.from_template(
            "在1到10的范围内评分，其中1表示日常琐事(例如刷牙，起床），而10表示极其重要的事情（例如分手，大学录取），请评估下面这段记忆"
            "的重要程度，用一个整数回答。```\n记忆: {memory_content}```\n评分: "
        )
        # 运行提示模板并获取结果
        score = self.invoke(prompt, {"memory_content": memory_content}).strip()
        # 如果处于详细模式，记录日志信息显示重要性分数
        if self.verbose:
            print.info(f"重要性分数: {score}")
        # 使用正则表达式从结果中提取分数
        match = re.search(r"^\D*(\d+)", score)
        # 如果匹配成功，则返回计算后的分数，否则返回0.0
        if match:
            return (float(match.group(1)) / 10) * self.importance_weight
        else:
            return 0.0

    def _score_memories_importance(self, memory_content: str) -> List[float]:
        """
        为给定的记忆打分，评估它们的绝对重要性
        参数 memory_content：一个包含多个记忆内容的字符串
        返回值：一个包含记忆重要性分数的浮点数列表
        """
        prompt = PromptTemplate.from_template(
            "在1到10的范围内评分，其中1表示日常琐事(例如刷牙，起床），而10表示极其重要的事情（例如分手，大学录取），请评估下面这段记忆"
            "的重要程度，始终只回答一个数字列表，如果只给出了一段记忆，仍然以列表形式回答。记忆之间用分号（;）分隔。```\n记忆: {memory_content}```\n评分: "
        )
        scores = self.invoke(prompt, {"memory_content": memory_content}).strip()
        if self.verbose:
            print.info(f"重要性分数: {scores}")
        scores_list = [float(x) for x in scores.split(";")]
        return scores_list


    def add_memories(
        self, memory_content: str, now: Optional[datetime] = None
    ) -> List[str]:
        """
        将一系列观察或记忆添加到智能代理的的记忆中
        参数 memory_content: 字符串形式的多个记忆内容，用分号分隔
        参数 now: 可选的当前时间参数，如果提供，将用于时间戳记。
        返回值: 添加记忆的结果，列表形式
        """
        # 为传入的记忆内容评分，以确定它们的重要性
        importance_scores = self._score_memories_importance(memory_content)

        # 累加最重要的记忆到总重要性
        self.aggregate_importance += max(importance_scores)
        # 将记忆内容分割成记忆列表
        memory_list = memory_content.split(";")
        documents = []

        # 为每个记忆创建一个Document对象，包含记忆内容和其重要性
        for i in range(len(memory_list)):
            documents.append(
                Document(
                    page_content=memory_list[i],
                    metadata={"importance": importance_scores[i]},
                )
            )
        # 向记忆检索器添加这些记忆
        result = self.memory_retriever.add_documents(documents, current_time=now)

        # 如果累计重要性超过了反思阈值，并且智能代理当前不在反思状态，则启动反思过程，并生成新的合成记忆
        if (
            self.reflection_threshold is not None
            and self.aggregate_importance > self.reflection_threshold
            and not self.reflecting
        ):
            self.reflecting = True
            self.pause_to_reflect(now=now)
            # 重置累计重要性，用于在反思后清空重要性
            self.aggregate_importance = 0.0
            self.reflecting = False
        # 返回添加记忆的结果
        return result

    def add_memory(
        self, memory_content: str, now: Optional[datetime] = None
    ) -> List[str]:
        """
        将单个观察或记忆添加到智能体的记忆中
        参数 memory_content: 字符串形式的单个记忆内容
        参数 now: 可选的当前时间参数，如果提供，将用于时间戳记忆
        返回值: 添加记忆的结果，列表形式
        """
        importance_score = self._score_memory_importance(memory_content)
        self.aggregate_importance += importance_score
        document = Document(
            page_content=memory_content, metadata={"importance": importance_score}
        )
        result = self.memory_retriever.add_documents([document], current_time=now)
        if (
            self.reflection_threshold is not None
            and self.aggregate_importance > self.reflection_threshold
            and not self.reflecting
        ):
            self.reflecting = True
            self.pause_to_reflect(now=now)
            self.aggregate_importance = 0.0
            self.reflecting = False
        return result
    
    def fetch_memories(
        self, observation: str, now: Optional[datetime] = None
    ) -> List[Document]:
        """
        根据观察内容检索相关的记忆
        参数 observation: 表示要检索记忆的观察内容
        参数 now: 可选的当前时间参数，用于时间过滤
        返回值: 一个文档列表，每个文档代表一个相关记忆
        """
        # 如果提供了当前时间，使用该时间作为模拟的“现在”，以获取相关记忆
        if now is not None:
            with mock_now(now):
                return self.memory_retriever.get_relevant_documents(observation)
        else:
            # 否则直接获取相关记忆
            return self.memory_retriever.get_relevant_documents(observation)

    def format_memories_detail(self, relevant_memories: List[Document]) -> str:
        """
        格式化相关记忆的详细信息为字符串
        参数 relevant_memories: 一个包含相关记忆文档的列表
        返回值: 一个字符串，包含所有相关记忆的详细信息
        """
        # 初始化内容列表
        content = []
        # 遍历相关记忆，将每个记忆的详细信息格式化后添加到内容列表
        for mem in relevant_memories:
            content.append(self._format_memory_detail(mem, prefix="- "))
        # 将内容列表合并成一个字符串，并返回
        return "\n".join([f"{mem}" for mem in content])

    def _format_memory_detail(self, memory: Document, prefix: str = "") -> str:
        """
        格式化单个记忆的详细信息
        参数 memory: 要格式化的记忆文档
        参数 prefix: 添加到每个记忆详细信息前的前缀字符串，默认为空
        返回值: 格式化后的记忆详细信息
        """
        # 获取记忆创建的时间
        created_time = memory.metadata["created_at"].strftime("%Y-%m-%d %H:%M:%S")
        # 返回格式化后的记忆详细信息，包含创建时间和记忆内容
        return f"{prefix}[{created_time}] {memory.page_content.strip()}"

    def format_memories_simple(self, relevant_memories: List[Document]) -> str:
        """
        简化记忆
        参数 relevant_memories: 一个包含相关记忆文档的列表
        返回值: 一个字符串，以分号分隔的所有相关记忆内容
        """
        return "; ".join([f"{mem.page_content}" for mem in relevant_memories])

    def _get_memories_until_limit(self, consumed_tokens: int) -> str:
        """
        减少文档中的令牌数量直至达到限制
        参数 consumed_tokens: 已经消耗的令牌数量
        返回值: 达到最大令牌限制的记忆内容字符串
        """
        result = []
        # 从最近的记忆开始遍历，直到达到最大令牌限制
        for doc in self.memory_retriever.memory_stream[::-1]:
            if consumed_tokens >= self.max_tokens_limit:
                break
            # 累计消耗的令牌数量
            consumed_tokens += self.llm.get_num_tokens(doc.page_content)
            # 如果消耗的令牌数量仍未达到限制，将文档添加到结果列表
            if consumed_tokens < self.max_tokens_limit:
                result.append(doc)
        return self.format_memories_simple(result)
