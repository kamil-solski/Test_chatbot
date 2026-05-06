import os
from functools import lru_cache


@lru_cache(maxsize=1)
def get_embeddings():
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()

    if provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings
        model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        model_path = os.getenv("EMBEDDING_MODEL_PATH", "/app/models")
        return HuggingFaceEmbeddings(model_name=model_name, cache_folder=model_path)

    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    )
