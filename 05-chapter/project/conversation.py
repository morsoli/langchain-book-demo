import inspect
import re
import logging
from typing import Any, Dict, List, Optional
from typing import List, Tuple, Union
from pydantic import Field

from langchain.schema import BasePromptTemplate, BaseRetriever, Document, BaseMessage
from langchain.schema.language_model import BaseLanguageModel
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
)
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)

from retrivers import MyRetriever
from config import *

logger = logging.getLogger(__name__)


ChatTurnType = Union[Tuple[str, str], BaseMessage]


def _get_standalone_questions_list(standalone_questions_str: str) -> List[str]:
    # 定义匹配独立问题的正则表达式
    pattern = r"\d+\.\s(.*?)(?=\n\d+\.|\n|$)"

    # 使用正则表达式查找匹配的问题
    matches = [
        match.group(1) for match in re.finditer(pattern, standalone_questions_str)
    ]
    if matches:
        return matches

    # 如果没有匹配到问题，尝试从 "standalone" 后的文本中提取问题
    match = re.search(
        r"(?i)standalone[^\n]*:[^\n](.*)", standalone_questions_str, re.DOTALL
    )
    sentence_source = match.group(1).strip() if match else standalone_questions_str
    sentences = sentence_source.split("\n")

    # 对提取的句子进行处理，移除编号和格式化字符
    return [
        re.sub(
            r"^\((\d+)\)\.? ?|^\d+\.? ?\)?|^(\d+)\) ?|^(\d+)\) ?|^[Qq]uery \d+: ?|^[Qq]uery: ?",
            "",
            sentence.strip(),
        )
        for sentence in sentences
        if sentence.strip()
    ]
    
def _get_chat_history(chat_history: List[ChatTurnType]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        if isinstance(dialogue_turn, BaseMessage):
            # 根据对话类型（人类或助手）添加前缀
            role_prefix = {"human": "Human: ", "ai": "Assistant: "}.get(dialogue_turn.type, f"{dialogue_turn.type}: ")
            buffer += f"\n{role_prefix}{dialogue_turn.content}"
        elif isinstance(dialogue_turn, tuple):
            # 对话轮次是一个元组时，分别处理人类和助手的消息
            human = "Human: " + dialogue_turn[0]
            ai = "Assistant: " + dialogue_turn[1]
            buffer += "\n" + "\n".join([human, ai])
        else:
            # 不支持的聊天历史格式抛出异常
            raise ValueError(
                f"Unsupported chat history format: {type(dialogue_turn)}."
                f" Full chat history: {chat_history} "
            )
    return buffer


class ConversationRetrievalChain(BaseConversationalRetrievalChain):
    """
    基于对话的检索链类，用于处理对话形式的检索任务。

    属性:
        retriever: MyRetriever 类型，用于获取文档的检索器。
        file_names: 文件名列表，用于检索的文件名。
    """

    retriever: MyRetriever = Field(exclude=True)
    file_names: List = Field(exclude=True)

    def _get_docs(self, question: str, inputs: Dict[str, Any], num_query: int, *, run_manager: Optional[CallbackManagerForChainRun] = None) -> List[Document]:
        """
        根据问题获取相关文档。

        参数:
            question: 提出的问题。
            inputs: 输入参数字典。
            num_query: 查询的数量。
            run_manager: 运行管理器，用于回调管理。

        返回:
            List[Document]: 返回相关的文档列表。
        """
        try:
            docs = self.retriever.get_relevant_documents(question, num_query=num_query, run_manager=run_manager)
            return docs
        except (IOError, FileNotFoundError) as error:
            logger.error("在 _get_docs 中发生错误: %s", error)
            return []

    def _retrieve(self, question_list: List[str], inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> List[str]:
        """
        对问题列表执行检索。

        参数:
            question_list: 问题列表。
            inputs: 输入参数字典。
            run_manager: 运行管理器。

        返回:
            List[str]: 检索到的文档片段列表。
        """
        num_query = len(question_list)
        accepts_run_manager = "run_manager" in inspect.signature(self._get_docs).parameters

        total_results = {}
        for question in question_list:
            docs_dict = self._get_docs(question, inputs, num_query=num_query, run_manager=run_manager) if accepts_run_manager else self._get_docs(question, inputs, num_query=num_query)
            for file_name, docs in docs_dict.items():
                if file_name not in total_results:
                    total_results[file_name] = docs
                else:
                    total_results[file_name].extend(docs)

            logger.info("-----step_done--------------------------------------------------")

        snippets = ""
        redundancy = set()
        for file_name, docs in total_results.items():
            sorted_docs = sorted(docs, key=lambda x: x.metadata["medium_chunk_index"])
            temp = "\n".join(doc.page_content for doc in sorted_docs if doc.metadata["page_content_md5"] not in redundancy)
            redundancy.update(doc.metadata["page_content_md5"] for doc in sorted_docs)
            snippets += f"\nContext about {file_name}:\n{{{temp}}}\n"

        return snippets, docs_dict

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        """
        执行对话式检索链的主要调用方法。

        参数:
            inputs: 包含输入数据的字典，例如问题和聊天历史。
            run_manager: 运行管理器，用于处理回调和管理执行流程。

        返回:
            字典，包含回答和其他可能的输出信息。
        """
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs["question"]
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])

        # 生成新问题
        callbacks = _run_manager.get_child()
        new_questions = self.question_generator.run(question=question, chat_history=chat_history_str, database=self.file_names, callbacks=callbacks)

        # 日志记录
        logger.info("new_questions: %s", new_questions)
        new_question_list = _get_standalone_questions_list(new_questions)[:3]
        logger.info("user_input: %s", question)
        logger.info("new_question_list: %s", new_question_list)

        # 检索相关文档片段
        snippets, source_docs = self._retrieve(new_question_list, inputs, run_manager=_run_manager)

        # 组合检索结果生成回答
        docs = [Document(page_content=snippets, metadata={})]
        new_inputs = inputs.copy()
        new_inputs["chat_history"] = chat_history_str
        answer = self.combine_docs_chain.run(input_documents=docs, database=self.file_names, callbacks=_run_manager.get_child(), **new_inputs)

        # 构造输出
        output: Dict[str, Any] = {self.output_key: answer}
        if self.return_source_documents:
            output["source_documents"] = source_docs
        if self.return_generated_question:
            output["generated_question"] = new_questions

        logger.info("*****response*****: %s", output["answer"])
        logger.info("=====epoch_done============================================================")
        return output

    async def _aget_docs(
        self,
        question: str,
        inputs: Dict[str, Any],
        num_query: int,
        *,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> List[Document]:
        """Get docs."""
        try:
            docs = await self.retriever.aget_relevant_documents(
                question, num_query=num_query, run_manager=run_manager
            )
            return docs
        except (IOError, FileNotFoundError) as error:
            logger.error("An error occurred in _get_docs: %s", error)
            return []

    async def _aretrieve(
        self,
        question_list: List[str],
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        num_query = len(question_list)
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._get_docs).parameters
        )

        total_results = {}
        for question in question_list:
            docs_dict = (
                await self._aget_docs(
                    question, inputs, num_query=num_query, run_manager=run_manager
                )
                if accepts_run_manager
                else await self._aget_docs(question, inputs, num_query=num_query)
            )

            for file_name, docs in docs_dict.items():
                if file_name not in total_results:
                    total_results[file_name] = docs
                else:
                    total_results[file_name].extend(docs)

            logger.info(
                "-----step_done--------------------------------------------------",
            )

        snippets = ""
        redundancy = set()
        for file_name, docs in total_results.items():
            sorted_docs = sorted(docs, key=lambda x: x.metadata["medium_chunk_index"])
            temp = "\n".join(
                doc.page_content
                for doc in sorted_docs
                if doc.metadata["page_content_md5"] not in redundancy
            )
            redundancy.update(doc.metadata["page_content_md5"] for doc in sorted_docs)
            snippets += f"\nContext about {file_name}:\n{{{temp}}}\n"

        return snippets, docs_dict

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        question = inputs["question"]
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])

        callbacks = _run_manager.get_child()
        new_questions = await self.question_generator.arun(
            question=question,
            chat_history=chat_history_str,
            database=self.file_names,
            callbacks=callbacks,
        )
        new_question_list = _get_standalone_questions_list(new_questions)[:3]
        logger.info("new_questions: %s", new_questions)
        logger.info("new_question_list: %s", new_question_list)

        snippets, source_docs = await self._aretrieve(
            new_question_list, inputs, run_manager=_run_manager
        )

        docs = [
            Document(
                page_content=snippets,
                metadata={},
            )
        ]

        new_inputs = inputs.copy()
        new_inputs["chat_history"] = chat_history_str
        answer = await self.combine_docs_chain.arun(
            input_documents=docs,
            database=self.file_names,
            callbacks=_run_manager.get_child(),
            **new_inputs,
        )
        output: Dict[str, Any] = {self.output_key: answer}
        if self.return_source_documents:
            output["source_documents"] = source_docs
        if self.return_generated_question:
            output["generated_question"] = new_questions

        logger.info("*****response*****: %s", output["answer"])
        logger.info(
            "=====epoch_done============================================================",
        )

        return output

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, retriever: BaseRetriever, 
                 condense_question_prompt: BasePromptTemplate = PromptTemplates().get_refine_qa_template(MODEL_NAME), 
                 chain_type: str = "stuff", verbose: bool = False, 
                 condense_question_llm: Optional[BaseLanguageModel] = None, 
                 combine_docs_chain_kwargs: Optional[Dict] = None, 
                 callbacks: Callbacks = None, **kwargs: Any) -> BaseConversationalRetrievalChain:
        """
        从语言模型创建问答检索链的类方法。

        参数:
            llm: 语言模型。
            retriever: 文档检索器。
            condense_question_prompt: 精简问题的提示模板。
            chain_type: 链类型。
            verbose: 是否显示详细日志。
            condense_question_llm: 精简问题的语言模型。
            combine_docs_chain_kwargs: 组合文档链的额外参数。
            callbacks: 回调函数集。
            kwargs: 其他参数。

        返回:
            BaseConversationalRetrievalChain: 对话检索链实例。
        """
        combine_docs_chain_kwargs = combine_docs_chain_kwargs or {
            "prompt": PromptTemplates().get_retrieval_qa_template_selector().get_prompt(llm)
        }
        doc_chain = load_qa_chain(llm, chain_type=chain_type, verbose=verbose, callbacks=callbacks, **combine_docs_chain_kwargs)

        _llm = condense_question_llm or llm
        condense_question_chain = LLMChain(llm=_llm, prompt=condense_question_prompt, verbose=verbose, callbacks=callbacks)
        return cls(retriever=retriever, combine_docs_chain=doc_chain, question_generator=condense_question_chain, callbacks=callbacks, **kwargs)