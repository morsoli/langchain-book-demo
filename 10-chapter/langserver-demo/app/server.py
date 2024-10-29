from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langserve import add_routes

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