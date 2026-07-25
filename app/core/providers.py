from app.core.config import Settings


def get_langchain_llm(settings: Settings):
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_key(),
            temperature=0,
        )
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_key(),
            temperature=0,
        )

    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_llm_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )


def get_langchain_embeddings(settings: Settings):
    if settings.embed_provider in {"semantic", "local-lite"}:
        from app.core.local_embeddings import build_local_embeddings

        return build_local_embeddings(
            mode=settings.embed_provider,
            model_name=settings.local_semantic_model,
            dimension=settings.local_embed_dimension,
        )
    if settings.embed_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embed_model,
            api_key=settings.openai_key(),
        )

    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )


def get_llamaindex_llm(settings: Settings):
    if settings.llm_provider == "anthropic":
        from llama_index.llms.anthropic import Anthropic

        return Anthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_key(),
            temperature=0,
        )
    if settings.llm_provider == "openai":
        from llama_index.llms.openai import OpenAI

        return OpenAI(
            model=settings.llm_model,
            api_key=settings.openai_key(),
            temperature=0,
        )

    from llama_index.llms.ollama import Ollama

    return Ollama(
        model=settings.ollama_llm_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        request_timeout=120,
    )


def get_llamaindex_embeddings(settings: Settings):
    if settings.embed_provider in {"semantic", "local-lite"}:
        from app.core.local_embeddings import build_llamaindex_embedding

        return build_llamaindex_embedding(
            mode=settings.embed_provider,
            model_name=settings.local_semantic_model,
            dimension=settings.local_embed_dimension,
        )
    if settings.embed_provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding

        return OpenAIEmbedding(
            model=settings.embed_model,
            api_key=settings.openai_key(),
        )

    from llama_index.embeddings.ollama import OllamaEmbedding

    return OllamaEmbedding(
        model_name=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
    )
