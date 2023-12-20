from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

app = FastAPI(
  title="LangServer",
  version="0.1",
  description="A simple api server by langsercer",
)

add_routes(app, ChatOpenAI(), path="/openai")

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("讲一个关于 {topic} 的笑话。")
add_routes(app, prompt | model, path="/joke")

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="localhost", port=8000)