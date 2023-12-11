import re
import string
from enum import Enum
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from langchain.schema import Document

def clean_text(text):
    """
    清洗文本：将文本转换为小写，去除标点符号和停用词，并进行词形还原。

    参数:
        text (str): 要清洗的文本。

    返回:
        str: 清洗和词形还原后的文本。
    """
    # 删除文本中的 [SEP] 标记
    text = text.replace("[SEP]", "")
    # 分词
    tokens = word_tokenize(text)
    # 转换为小写
    tokens = [w.lower() for w in tokens]
    # 去除标点符号
    table = str.maketrans("", "", string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # 保留字母、数字或同时包含字母和数字的单词
    words = [
        word
        for word in stripped
        if word.isalpha()
        or word.isdigit()
        or (re.search("\d", word) and re.search("[a-zA-Z]", word))
    ]
    # 去除停用词
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if w not in stop_words]
    # 词形还原
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w) for w in words]
    # 将词列表转换回字符串
    lemmatized_ = " ".join(lemmatized)

    return lemmatized_


class IndexerOperator(Enum):
    """
    枚举类：用于索引中不同查询操作符的枚举。

    属性:
        EQ (str): 等于 "=="
        GT (str): 大于 ">"
        GTE (str): 大于等于 ">="
        LT (str): 小于 "<"
        LTE (str): 小于等于 "<="
    """

    EQ = "=="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="


class DocIndexer:
    """
    文档索引处理类：用于处理文档的索引和搜索。

    属性:
        documents (List[Document]): 需要索引的文档列表。
    """

    def __init__(self, documents):
        self.documents = documents
        self.index = self.build_index(documents)

    def build_index(self, documents):
        """
        为给定的文档列表构建索引。

        参数:
            documents (List[Document]): 需要被索引的文档列表。

        返回:
            dict: 构建的索引。
        """
        index = {}
        for doc in documents:
            for key, value in doc.metadata.items():
                if key not in index:
                    index[key] = {}
                if value not in index[key]:
                    index[key][value] = []
                index[key][value].append(doc)
        return index

    def retrieve_metadata(self, search_dict):
        """
        根据提供的 search_dict 中的搜索条件检索文档。

        参数:
            search_dict (dict): 指定搜索条件的字典。可以包含 "AND" 或 "OR" 运算符进行复杂查询。

        返回:
            List[Document]: 符合搜索条件的文档列表。
        """
        if "AND" in search_dict:
            return self._handle_and(search_dict["AND"])
        elif "OR" in search_dict:
            return self._handle_or(search_dict["OR"])
        else:
            return self._handle_single(search_dict)

    def _handle_and(self, search_dicts):
        # 使用 "AND" 条件进行复杂查询的处理
        results = [self.retrieve_metadata(sd) for sd in search_dicts]
        if results:
            # 取交集
            intersection = set.intersection(
                *[set(map(self._hash_doc, r)) for r in results]
            )
            return [self._unhash_doc(h) for h in intersection]
        else:
            return []

    def _handle_or(self, search_dicts):
        # 使用 "OR" 条件进行复杂查询的处理
        results = [self.retrieve_metadata(sd) for sd in search_dicts]
        union = set.union(*[set(map(self._hash_doc, r)) for r in results])
        return [self._unhash_doc(h) for h in union]

    def _handle_single(self, search_dict):
        # 处理单一搜索条件
        unions = []
        for key, query in search_dict.items():
            operator, value = query
            union = set()
            # 根据不同的操作符来过滤数据
            if operator == IndexerOperator.EQ:
                if key in self.index and value in self.index[key]:
                    union.update(map(self._hash_doc, self.index[key][value]))
            else:
                if key in self.index:
                    for k, v in self.index[key].items():
                        if (
                            (operator == IndexerOperator.GT and k > value)
                            or (operator == IndexerOperator.GTE and k >= value)
                            or (operator == IndexerOperator.LT and k < value)
                            or (operator == IndexerOperator.LTE and k <= value)
                        ):
                            union.update(map(self._hash_doc, v))
            if union:
                unions.append(union)

        if unions:
            intersection = set.intersection(*unions)
            return [self._unhash_doc(h) for h in intersection]
        else:
            return []

    def _hash_doc(self, doc):
        # 为文档创建哈希值，以便在集合操作中使用
        return (doc.page_content, frozenset(doc.metadata.items()))

    def _unhash_doc(self, hashed_doc):
        # 从哈希值重构文档对象
        page_content, metadata = hashed_doc
        return Document(page_content=page_content, metadata=dict(metadata))
