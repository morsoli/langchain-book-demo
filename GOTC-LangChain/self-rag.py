import os
import bs4
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_sk_xxx"
os.environ["LANGCHAIN_PROJECT"] = "langgraph-selfrag"

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
zhipu_embedding = OpenAIEmbeddings(model="embedding-2", base_url="https://open.bigmodel.cn/api/paas/v4/", api_key=ZHIPU_API_KEY)

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://liduos.com/how-to-build-llm-agent-2024.html",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("content", "card-content", "article")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=zhipu_embedding)

retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="glm-4-air", base_url="https://open.bigmodel.cn/api/paas/v4/", api_key=ZHIPU_API_KEY)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import hub


# 文档和问题是否相关
class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="文档和问题是否相关, 'yes' or 'no'"
    )
grade_prompt = hub.pull("efriis/self-rag-retrieval-grader")
structured_llm_grader = llm.with_structured_output(GradeDocuments)
retrieval_grader = grade_prompt | structured_llm_grader

# question = "AI产品经理的职责?"
# docs = retriever.invoke(question)
# doc_txt = docs[0].page_content
# print(doc_txt)
# print(retrieval_grader.invoke({"question": question, "document": doc_txt}))


# 答案是否有事实根据
class GradeHallucinations(BaseModel):
    binary_score: str = Field(
        description="答案是否有事实根据, 'yes' or 'no'"
    )
hallucination_prompt = hub.pull("efriis/self-rag-hallucination-grader")
structured_llm_grader = llm.with_structured_output(GradeHallucinations)
hallucination_grader = hallucination_prompt | structured_llm_grader

# 答案是否解答了这个问题
class GradeAnswer(BaseModel):

    binary_score: str = Field(
        description="答案是否解答了这个问题, 'yes' or 'no'"
    )
answer_prompt = hub.pull("efriis/self-rag-answer-grader")
structured_llm_grader = llm.with_structured_output(GradeAnswer)
answer_grader = answer_prompt | structured_llm_grader

# 重写问题
re_write_prompt = hub.pull("efriis/self-rag-question-rewriter")
question_rewriter = re_write_prompt | llm | StrOutputParser()

# 常规 RAG
prompt = hub.pull("rlm/rag-prompt")
rag_chain = prompt | llm | StrOutputParser()


from typing import List

from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]
    
def retrieve(state):
    """
    检索文档

    参数:
        state (dict): 当前图状态

    返回:
        state (dict): 新增加了"documents"键的状态字典,它包含检索到的文档
    """
    print("---检索---")
    question = state["question"]

    # 检索文档
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    """
    生成答案

    参数:
        state (dict): 当前图状态

    返回:
        state (dict): 新增加了"generation"键的状态字典,它包含LLM生成的内容
    """
    print("---生成---")
    question = state["question"]
    documents = state["documents"]

    # RAG 生成
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    确定检索到的文档是否与问题相关。

    参数:
        state (dict): 当前图状态

    返回:
        state (dict): 更新"documents"键,只保留相关的文档
    """

    print("---检查文档与问题的相关性---")
    question = state["question"]
    documents = state["documents"]

    # 评分每个文档
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        if not score:
            print("---评分: 文档不相关---")
            continue
        grade = score.binary_score
        if grade == "yes":
            print("---评分: 文档相关---")
            filtered_docs.append(d)
        else:
            print("---评分: 文档不相关---")
            continue
    return {"documents": filtered_docs, "question": question}


def transform_query(state):
    """
    转换查询以生成更好的问题。

    参数:
        state (dict): 当前图状态

    返回:
        state (dict): 更新"question"键,使用重新表述的问题
    """

    print("---转换查询---")
    question = state["question"]
    documents = state["documents"]

    # 重写问题
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

def decide_to_generate(state):
    """
    确定是生成答案还是重新生成问题。

    参数:
        state (dict): 当前图状态

    返回:
        str: 下一个节点的二进制决策
    """

    print("---评估已过滤的文档---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # 所有文档都已被过滤
        # 我们将重新生成一个新的查询
        print(
            "---决策: 所有文档与问题无关, 转换查询---"
        )
        return "transform_query"
    else:
        # 我们有相关的文档, 所以生成答案
        print("---决策: 生成---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    确定生成的内容是否与文档相关并回答了问题。

    参数:
        state (dict): 当前图状态

    返回:
        str: 下一个节点的决策
    """

    print("---检查幻觉---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # 检查幻觉
    if grade == "yes":
        print("---决策: 生成内容与文档相关---")
        # 检查问题回答
        print("---评估生成内容与问题的关系---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---决策: 生成内容回答了问题---")
            return "useful"
        else:
            print("---决策: 生成内容没有回答问题---")
            return "not useful"
    else:
        print("---决策: 生成内容与文档不相关, 重试---")
        return "not supported"
    
from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile
app = workflow.compile()


from pprint import pprint

# Run
inputs = {"question": "大模型在toB有哪些落地形式?"}
for output in app.stream(inputs)       :
    for key, value in output.items():
        pprint(f"Node '{key}':")
    pprint("\n---\n")

# Final generation
pprint(value["generation"])