from langchain.llms.openai import OpenAI
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from dotenv import load_dotenv 

# 加载环境变量
load_dotenv()

# 测试数据库链的功能
def test_db_chain():
    # 创建一个SQL数据库实例，连接到SQLite数据库
    db = SQLDatabase.from_uri("sqlite:///../user.db")

    # 创建一个OpenAI的LLM实例，设置温度参数和详细模式
    llm = OpenAI(temperature=0, verbose=True)

    # 创建SQLDatabaseChain实例，结合LLM和数据库，开启详细模式和查询检查器
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, use_query_checker=True)

    # 运行链并提出查询：“有多少用户？”
    db_chain.run("有多少用户?")

if __name__ == "__main__":
    test_db_chain()