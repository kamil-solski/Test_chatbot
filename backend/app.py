from contextlib import asynccontextmanager
from typing import Annotated, Literal, TypedDict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from chat_history import ChatHistoryStore, ChunkSummarizer, TopicClusterer, classify_query, retrieve_for_query
from config import CONFIG, LITELLM_BASE_URL, OPENAI_API_KEY, REDIS_URL
from helpers.log import log_message
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, trim_messages
from rag import DocumentRAGStore, SessionRAGStore
from rag.embeddings import get_embeddings
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

DEFAULT_MODEL: str = CONFIG["llm"]["default_model"]
LLM_TEMPERATURE: float = CONFIG["llm"]["temperature"]
DEFAULT_SYSTEM_PROMPT: str = CONFIG["llm"]["system_prompt"]

HISTORY_STRATEGY: str = CONFIG["history"]["strategy"]
MAX_HISTORY_TOKENS: int = CONFIG["history"]["max_tokens"]
SUMMARIZE_AFTER: int = CONFIG["history"]["summarize_after"]
RAG_TOP_K: int = CONFIG["history"]["rag_top_k"]
WINDOW_SIZE: int = CONFIG["history"]["window_size"]
CHUNK_SIZE: int = CONFIG["history"]["chunk_size"]
TOPIC_NAMING_MODEL: str = CONFIG["history"]["topic_naming_model"]
TOPIC_SIMILARITY_THRESHOLD: float = CONFIG["history"]["topic_similarity_threshold"]

DOC_TOP_K: int = CONFIG["rag"]["doc_top_k"]

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
