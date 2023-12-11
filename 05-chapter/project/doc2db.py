import os
import pickle
from typing import List
from collections import deque
import hashlib
import tiktoken
from tqdm import tqdm
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain.document_loaders import PyPDFLoader, TextLoader

from config import *
from utils import clean_text

def calculate_token_length(text: str):
    """计算给定文本字符串使用TikToken的令牌长度。

    参数:
        text (str): 需要被令牌化的文本。

    返回值:
        int: 令牌化文本的长度。
    """
    tokenizer_name = tiktoken.encoding_for_model(MODEL_NAME)
    tokenizer = tiktoken.get_encoding(tokenizer_name.name)
    tokens = tokenizer.encode(text, disallowed_special=())

    return len(tokens)


def string_to_md5(text: str):
    """将字符串转换为其MD5哈希值。

    参数:
        text (str): 需要被哈希的文本。

    返回值:
        str: 输入字符串的MD5哈希值。
    """
    md5_hash = hashlib.md5()
    md5_hash.update(text.encode("utf-8"))

    return md5_hash.hexdigest()


def load_document(file_path):
    """加载文件并将其内容作为文档对象返回。

    参数:
        file_path (str): 文件的路径。

    返回值:
        Document: 加载的文档。
    """
    loader_mapping = {
        "pdf": (PyPDFLoader, {}),
        "txt": (TextLoader, {}),
    }
    extension = file_path.split(".")[-1]
    if extension in loader_mapping:
        loader_class, args = loader_mapping[extension]
        loader = loader_class(file_path, **args)
        document = loader.load()

        return document

    raise ValueError(f"不支持的扩展名 {extension}")

def documents_to_vector_store(documents: List[Document], embedding_name: str, suffix: str = ""):
    """
    将文档列表转换为向量格式并存储在指定的目录中。

    参数:
        documents (List[Document]): 要转换的文档对象列表。
        embedding_name (str): 使用的嵌入名称，用于标识不同的嵌入方法或模型。
        suffix (str, 可选): 用于生成存储名称的后缀，以区分不同的存储。默认为空字符串。

    功能:
        - 根据嵌入名称和后缀创建存储名称。
        - 检查存储目录是否存在，如果不存在则创建它。
        - 使用Chroma库将文档列表转换为向量格式，并持久化到指定目录。
    """
    store_name = f"{embedding_name}_{suffix}"
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
    Chroma.from_documents(
        documents,
        OpenAIEmbeddings(),
        persist_directory=f"{DB_DIR}/chroma_{store_name}",
    )


def file_names_to_pickle(file_names: list, save_name: str = ""):
    """
    将文件名列表保存为Pickle格式的文件。

    参数:
        file_names (list): 需要保存的文件名列表。
        save_name (str, 可选): 保存的Pickle文件的名称。默认为空字符串。

    功能:
        - 根据提供的保存名称创建Pickle文件的完整路径。
        - 检查存储目录是否存在，如果不存在则创建它。
        - 将文件名列表以Pickle格式写入文件。
    """
    pickle_name = f"{save_name}"
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
    with open(f"{DB_DIR}/{pickle_name}.pkl", "wb") as file:
        pickle.dump(file_names, file)


def documents_to_pickle(documents: List[Document], suffix: str = ""):
    """
    将文档对象列表清理后保存为Pickle格式的文件。

    参数:
        documents (List[Document]): 要保存的文档对象列表。
        suffix (str, 可选): 用于生成Pickle文件名称的后缀。默认为空字符串。

    功能:
        - 对每个文档对象的页面内容进行清理。
        - 根据后缀创建Pickle文件的完整路径。
        - 检查存储目录是否存在，如果不存在则创建它。
        - 将处理后的文档对象列表以Pickle格式写入文件。
    """
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    pickle_name = f"pickle_{suffix}"
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
    with open(f"{DB_DIR}/docs_{pickle_name}.pkl", "wb") as file:
        pickle.dump(documents, file)



def split_document(
    document: List[Document], chunk_size: int, chunk_overlap: int, chunk_index_name: str
):
    """根据提供的大小和重叠将文档切割成更小的块。

    参数:
        document (List[Document]): 要被切割的文档。
        chunk_size (int): 每个块的大小。
        chunk_overlap (int): 相邻块之间的重叠。
        chunk_index_name (str): 用于存储块索引的元数据键。

    返回值:
        list: 表示块的文档对象列表。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=calculate_token_length,
    )
    split_documents = splitter.split_documents(document)
    chunk_index = 0

    for split_doc in split_documents:
        split_doc.metadata[chunk_index_name] = chunk_index
        chunk_index += 1

    return split_documents


def update_document_metadata(documents: List[Document]):
    """处理并更新文档对象列表的元数据。

    参数:
        documents (List[Document]): 文档对象列表。
    """
    file_name_with_extension = os.path.basename(documents[0].metadata["source"])
    file_name, _ = os.path.splitext(file_name_with_extension)

    for _, doc in enumerate(documents):
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = str(value)
        doc.metadata["page_content"] = doc.page_content
        doc.metadata["page_content_md5"] = string_to_md5(doc.page_content)
        doc.metadata["source_md5"] = string_to_md5(doc.metadata["source"])
        doc.page_content = f"{file_name}\n{doc.page_content}"


def add_window_to_document(
    document: Document, window_steps: int, window_size: int, window_index_name: str
):
    """在文档列表的每个文档中添加窗口化信息。

    参数:
        document (Document): 文档对象列表。
        window_steps (int): 窗口化的步长。
        window_size (int): 每个窗口的大小。
        window_index_name (str): 用于存储窗口索引的元数据键。
    """
    window_id = 0
    window_queue = deque()

    for index, doc in enumerate(document):
        if index % window_steps == 0 and index != 0 and index < len(document) - window_size:
            window_id += 1
        window_queue.append(window_id)

        if len(window_queue) > window_size:
            for _ in range(window_steps):
                window_queue.popleft()

        window = set(window_queue)
        doc.metadata[f"{window_index_name}_lower_bound"] = min(window)
        doc.metadata[f"{window_index_name}_upper_bound"] = max(window)


def merge_metadata_dicts(metadata_dicts: List[dict]):
    """将元数据字典列表合并为单个字典。

    参数:
        metadata_dicts (List[dict]): 元数据字典列表。

    返回值:
        dict: 合并后的元数据字典。
    """
    merged_dict = {}
    bounds_dict = {}
    keys_to_remove = set()

    for dic in metadata_dicts:
        for key, value in dic.items():
            if key in merged_dict:
                if value not in merged_dict[key]:
                    merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]

    for key, values in merged_dict.items():
        if len(values) > 1 and all(isinstance(x, (int, float)) for x in values):
            bounds_dict[f"{key}_lower_bound"] = min(values)
            bounds_dict[f"{key}_upper_bound"] = max(values)
            keys_to_remove.add(key)

    merged_dict.update(bounds_dict)

    for key in keys_to_remove:
        del merged_dict[key]

    return {
        k: v[0] if isinstance(v, list) and len(v) == 1 else v
        for k, v in merged_dict.items()
    }


def merge_document_chunks(document: Document, scale_factor: int, chunk_index_name: str):
    """根据缩放因子将相邻的小块合并成较大的块。

    参数:
        document (Document): 文档对象列表。
        scale_factor (int): 要合并成较大块的小块数量。
        chunk_index_name (str): 用于存储块索引的元数据键。

    返回值:
        list: 表示合并块的文档对象列表。
    """
    merged_documents = []
    content_aggregate = ""
    metadata_aggregate = []
    chunk_index = 0

    for index, item in enumerate(document):
        content_aggregate += item.page_content
        metadata_aggregate.append(item.metadata)

        if (index + 1) % scale_factor == 0 or index == len(document) - 1:
            metadata = merge_metadata_dicts(metadata_aggregate)
            metadata[chunk_index_name] = chunk_index
            merged_documents.append(
                Document(
                    page_content=content_aggregate,
                    metadata=metadata,
                )
            )
            chunk_index += 1
            content_aggregate = ""
            metadata_aggregate = []

    return merged_documents


def process_file_paths():
    small_chunks = []
    medium_chunks = []
    processed_file_names = []
    with tqdm(total=len(FILE_PATH), desc="处理文件", ncols=80) as progress_bar:
        for path in FILE_PATH:
            # 加载文件
            document = load_document(path)

            file_name_with_extension = os.path.basename(document[0].metadata["source"])

            # 对文档进行分割并添加窗口
            small_chunk_split = split_document(
                document=document,
                chunk_size=BASE_CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                chunk_index_name="small_chunk_index",
            )
            add_window_to_document(
                document=small_chunk_split,
                window_steps=WINDOW_STEPS,
                window_size=WINDOW_SCALE,
                window_index_name="large_chunks_index",
            )

            # 合并文档并处理元数据
            medium_chunk_split = merge_document_chunks(
                document=small_chunk_split,
                scale_factor=CHUNK_SCALE,
                chunk_index_name="medium_chunk_index",
            )
            update_document_metadata(small_chunk_split)
            update_document_metadata(medium_chunk_split)

            # 保存文件名和处理后的文档块
            processed_file_names.append(file_name_with_extension)
            small_chunks.extend(small_chunk_split)
            medium_chunks.extend(medium_chunk_split)

            progress_bar.update()

    # 将文件名保存为pickle文件
    file_names_to_pickle(processed_file_names, save_name="file_names")

    # 将处理后的文档块保存为向量存储和pickle文件
    documents_to_vector_store(small_chunks, "openAIEmbeddings", suffix="small_chunks")
    documents_to_vector_store(medium_chunks, "openAIEmbeddings", suffix="medium_chunks")
    documents_to_pickle(small_chunks, suffix="small_chunks")
    documents_to_pickle(medium_chunks, suffix="medium_chunks")



if __name__ == "__main__":
    process_file_paths()