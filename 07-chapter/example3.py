from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationEntityMemory
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
memory = ConversationEntityMemory(llm=llm)
# 示例输入
_input = {"input": "小李和莫尔索正在参加一场AI领域的黑客马拉松"}

# 加载内存变量
memory.load_memory_variables(_input)

# 保存上下文信息
memory.save_context(
    _input,
    {"output": "听起来真不错，他们在做什么项目"}
)

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=ConversationEntityMemory(llm=llm)
)

if __name__ == "__main__":
    # 查询特定实体信息
    # print(memory.load_memory_variables({"input": "莫尔索在干嘛？"}))
    print(conversation.predict(input="小李和莫尔索正在参加一场AI领域的黑客马拉松"))