from fastapi import FastAPI
from langserve import add_routes
from rag_conversation import chain as rag_conversation_chain

app = FastAPI()

# Edit this to add the chain you want to add
add_routes(app, NotImplemented)
add_routes(app, rag_conversation_chain, path="/rag-conversation")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
