from langchain.schema import BaseMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from typing import Any, Dict, List
from langchain.chains import ConversationChain
from langchain.schema import BaseMemory
from dotenv import load_dotenv
import spacy

# 加载spaCy的中文模型
nlp = spacy.load("zh_core_web_lg")
# 加载环境变量
load_dotenv()

# 初始化ChatOpenAI模型
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

class SimpleEntityMemory(BaseMemory):
    """用于存储实体信息的内存类"""

    # 定义字典来存储有关实体的信息
    entities: dict = {}
    # 定义键名，用于将实体信息传递到提示中
    memory_key: str = "entities"

    def clear(self):
        """清空实体信息"""
        self.entities = {}

    @property
    def memory_variables(self) -> List[str]:
        """定义我们提供给提示的变量"""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """加载内存变量，即实体键"""
        # 获取输入文本并通过spaCy处理
        doc = nlp(inputs[list(inputs.keys())[0]])
        # 提取已知实体的信息（如果存在）
        entities = [
            self.entities[str(ent)] for ent in doc.ents if str(ent) in self.entities
        ]
        # 返回合并的实体信息，放入上下文
        return {self.memory_key: "\n".join(entities)}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """将此次对话的上下文保存到缓冲区"""
        # 获取输入文本并通过spaCy处理
        text = inputs[list(inputs.keys())[0]]
        doc = nlp(text)
        # 对于每个提到的实体，将此信息保存到字典中
        for ent in doc.ents:
            ent_str = str(ent)
            if ent_str in self.entities:
                self.entities[ent_str] += f"\n{text}"
            else:
                self.entities[ent_str] = text
        print(self.entities)

if __name__ == "__main__":
    # 创建自定义内存实例
    memory = SimpleEntityMemory()
    # 设置模板
    template = """以下是一个人类和AI之间的友好对话。AI很健谈，从其上下文中提供了很多具体细节。如果AI不知道答案，它会如实说不知道。如果相关，你会得到有关人类提到的实体的信息。

    相关实体信息:
    {entities}

    对话:
    人类: {input}
    AI:"""
    prompt = PromptTemplate(input_variables=["entities", "input"], template=template)
    # 创建对话链
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=True
    )

    # 使用对话链进行对话
    print(conversation.predict(input="小李在大学时对数学感兴趣"))
    print(conversation.predict(input="小李对什么感兴趣？"))
