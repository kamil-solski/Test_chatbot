from functools import lru_cache

from config import CONFIG, EMBEDDING_MODEL_PATH, OPENAI_API_KEY


@lru_cache(maxsize=1)
def get_embeddings():
    provider = str(CONFIG["embeddings"]["provider"]).lower()

    if provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=CONFIG["embeddings"]["model_name"],
            cache_folder=EMBEDDING_MODEL_PATH,
        )

    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model=CONFIG["embeddings"]["model"],
    )
