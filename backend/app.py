import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from chat_history import ChatHistoryStore, ChunkSummarizer, TopicClusterer, classify_query, retrieve_for_query
from helpers.log import log_debug, log_message
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, trim_messages
from rag import DocumentRAGStore, SessionRAGStore
from rag.embeddings import get_embeddings
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "1.0"))
DEFAULT_SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

HISTORY_STRATEGY = os.getenv("HISTORY_STRATEGY", "full")   # full | trim | summarize | rag | smart
MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "2000"))
SUMMARIZE_AFTER = int(os.getenv("SUMMARIZE_AFTER", "6"))    # message count
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))               # top-k retrieved messages
DOC_TOP_K = int(os.getenv("DOC_TOP_K", "3"))               # top-k document chunks
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "6"))           # smart: sliding window length
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "6"))             # smart: messages per summary chunk
TOPIC_NAMING_MODEL = os.getenv("TOPIC_NAMING_MODEL", "gpt-4o-mini")
TOPIC_SIMILARITY_THRESHOLD = float(os.getenv("TOPIC_SIMILARITY_THRESHOLD", "0.65"))

log_debug("app.py:startup", "backend started", {
    "LITELLM_BASE_URL": str(LITELLM_BASE_URL),
    "DEFAULT_MODEL": DEFAULT_MODEL,
    "HISTORY_STRATEGY": HISTORY_STRATEGY,
})

doc_store = DocumentRAGStore()


class State(TypedDict):
    messages: Annotated[list, add_messages]
    summary: str
    routing_intent: str


def _token_count(messages: list) -> int:
    """Rough token estimate: 1 token ≈ 4 chars. Works for all models."""
    return sum(len(m.content) // 4 for m in messages if isinstance(m.content, str))


def _make_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(model=model, api_key=OPENAI_API_KEY, base_url=LITELLM_BASE_URL, temperature=LLM_TEMPERATURE)


def build_graph() -> StateGraph:
    rag_store = SessionRAGStore() if HISTORY_STRATEGY == "rag" else None
    chat_store = None
    summarizer = None
    if HISTORY_STRATEGY == "smart":
        clusterer = TopicClusterer(
            embeddings=get_embeddings(),
            llm=_make_llm(TOPIC_NAMING_MODEL),
            similarity_threshold=TOPIC_SIMILARITY_THRESHOLD,
        )
        chat_store = ChatHistoryStore(clusterer)
        summarizer = ChunkSummarizer(llm=_make_llm(TOPIC_NAMING_MODEL), chunk_size=CHUNK_SIZE)

    async def call_model(state: State, config: RunnableConfig):
        model_name = config["configurable"].get("model", DEFAULT_MODEL)
        llm = _make_llm(model_name)
        session_id = config["configurable"]["thread_id"]

        log_debug("app.py:call_model", "invoking LLM", {"model": model_name, "strategy": HISTORY_STRATEGY})

        summary = state.get("summary", "")
        system_content = DEFAULT_SYSTEM_PROMPT
        if summary:
            system_content += f"\n\nSummary of earlier conversation:\n{summary}"

        # Inject relevant document chunks into system prompt
        if doc_store.has_documents(session_id):
            current_text = state["messages"][-1].content if state["messages"] else ""
            chunks = doc_store.retrieve(session_id, current_text, k=DOC_TOP_K)
            if chunks:
                context_block = "\n\n---\n".join(c.page_content for c in chunks)
                system_content += f"\n\nRelevant context from uploaded documents:\n{context_block}"

        messages = state["messages"]

        if HISTORY_STRATEGY == "trim":
            messages = trim_messages(
                messages,
                max_tokens=MAX_HISTORY_TOKENS,
                token_counter=_token_count,
                strategy="last",
                start_on="human",
                include_system=False,
            )
        elif HISTORY_STRATEGY == "rag":
            current = messages[-1]
            rag_store.index_messages(session_id, messages[:-1])
            docs = rag_store.retrieve(session_id, current.content, k=RAG_TOP_K)
            role_map = {"human": HumanMessage, "ai": AIMessage}
            retrieved = [role_map.get(d.metadata["role"], HumanMessage)(content=d.page_content) for d in docs]
            messages = retrieved + [current]
        elif HISTORY_STRATEGY == "smart":
            current_text = messages[-1].content
            available_topics = chat_store.list_topics(session_id)
            decision = classify_query(current_text, available_topics)
            log_debug("app.py:call_model", "routing decision", {
                "intent": decision.intent.value,
                "reason": decision.reason,
                "topic_filter": decision.topic_filter,
            })
            messages, extra_context = await retrieve_for_query(
                decision=decision,
                messages=messages,
                store=chat_store,
                session_id=session_id,
                query=current_text,
                rag_k=RAG_TOP_K,
                window_size=WINDOW_SIZE,
                summarizer=summarizer,
            )
            if extra_context:
                system_content += f"\n\n{extra_context}"

        response = await llm.ainvoke([SystemMessage(content=system_content)] + messages)

        log_debug("app.py:call_model", "LLM responded", {
            "response_model": getattr(response, "response_metadata", {}).get("model_name", "unknown"),
        })
        result: dict = {"messages": [response]}
        if HISTORY_STRATEGY == "smart":
            result["routing_intent"] = decision.intent.value
        return result

    async def summarize_conversation(state: State, config: RunnableConfig):
        model_name = config["configurable"].get("model", DEFAULT_MODEL)
        llm = _make_llm(model_name)

        existing_summary = state.get("summary", "")
        if existing_summary:
            prompt = f"Extend this summary with the new conversation:\n\n{existing_summary}"
        else:
            prompt = "Summarize the conversation above concisely:"

        response = await llm.ainvoke(state["messages"] + [HumanMessage(content=prompt)])

        # Keep only the last 2 messages (most recent exchange), compress the rest
        delete = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete}

    def route_after_agent(state: State) -> Literal["summarize_conversation", "__end__"]:
        if HISTORY_STRATEGY == "summarize" and len(state["messages"]) > SUMMARIZE_AFTER:
            return "summarize_conversation"
        return END

    builder = StateGraph(State)
    builder.add_node("agent", call_model)
    builder.add_edge(START, "agent")

    if HISTORY_STRATEGY == "summarize":
        builder.add_node("summarize_conversation", summarize_conversation)
        builder.add_conditional_edges("agent", route_after_agent)
        builder.add_edge("summarize_conversation", END)
    else:
        builder.add_edge("agent", END)

    return builder


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncRedisSaver.from_conn_string(REDIS_URL) as checkpointer:
        await checkpointer.asetup()
        app.state.graph = build_graph().compile(checkpointer=checkpointer)
        yield


app = FastAPI(title="Test Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str
    model: str = DEFAULT_MODEL


class ChatResponse(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str
    model: str
    usage: dict
    strategy: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    log_debug("app.py:chat", "request received", {
        "model": request.model,
        "session_id": request.session_id,
    })

    config = {
        "configurable": {
            "thread_id": request.session_id,
            "model": request.model,
        }
    }

    try:
        result = await app.state.graph.ainvoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    ai_message = result["messages"][-1]
    usage = ai_message.usage_metadata or {}

    response_usage = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }

    log_message(
        session_id=request.session_id,
        model=request.model,
        question=request.message,
        usage=response_usage,
        strategy=HISTORY_STRATEGY,
        temperature=LLM_TEMPERATURE,
        routing_intent=result.get("routing_intent", ""),
    )

    return ChatResponse(
        content=ai_message.content,
        model=request.model,
        usage=response_usage,
        strategy=HISTORY_STRATEGY,
    )


@app.post("/api/documents/upload")
async def upload_document(session_id: str, file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")

    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")

    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty")

    chunks = doc_store.add_document(session_id, file.filename, text)
    return {
        "filename": file.filename,
        "chunks": chunks,
        "documents": doc_store.list_documents(session_id),
    }


@app.delete("/api/documents")
async def clear_documents(session_id: str):
    doc_store.clear(session_id)
    return {"status": "cleared"}
