import requests

from structs import MLEBEvaluationModelConfig

MODEL_CONFIGS = [
    # Embedders
    # | Isaacus
    MLEBEvaluationModelConfig(
        id="kanon-2-embedder",
        provider="isaacus",
        model_framework="isaacus",
        model_type="embedder",
        batch_size=16,
        mteb_metadata={
            "name": "isaacus/kanon-2-embedder",
            "revision": "1",
            "release_date": None,
            "languages": None,
            "n_parameters": None,
            "memory_usage_mb": None,
            "max_tokens": 16_384,
            "embed_dim": 1_792,
            "license": None,
            "open_weights": None,
            "public_training_data": None,
            "public_training_code": None,
            "framework": ["API"],
            "reference": None,
            "similarity_fn_name": "cosine",
            "use_instructions": None,
            "training_datasets": None,
            "adapted_from": None,
            "superseded_by": None,
            "loader": None,
        },
    ),
    # | Qwen
    MLEBEvaluationModelConfig(
        id="Qwen/Qwen3-Embedding-0.6B",
        provider="huggingface",
        model_framework="sentence-transformer",
        model_type="embedder",
        batch_size=1,
        mteb_metadata=None,
        dtype="bfloat16",
    ),
    MLEBEvaluationModelConfig(
        id="Qwen/Qwen3-Embedding-4B",
        provider="huggingface",
        model_framework="sentence-transformer",
        model_type="embedder",
        batch_size=1,
        mteb_metadata=None,
        dtype="bfloat16",
    ),
    MLEBEvaluationModelConfig(
        id="Qwen/Qwen3-Embedding-8B",
        provider="huggingface",
        model_framework="sentence-transformer",
        model_type="embedder",
        batch_size=1,
        mteb_metadata=None,
        dtype="bfloat16",
    ),
    # | BGE
    MLEBEvaluationModelConfig(
        id="BAAI/bge-m3",
        provider="huggingface",
        model_framework="sentence-transformer",
        model_type="embedder",
        batch_size=1,
        mteb_metadata=None,
        amp_dtype="float16",
    ),
    # | Microsoft
    MLEBEvaluationModelConfig(
        id="intfloat/multilingual-e5-large-instruct",
        provider="huggingface",
        model_framework="sentence-transformer",
        model_type="embedder",
        batch_size=1,
        mteb_metadata=None,
        amp_dtype="float16",
    ),
    # | Mixedbread
    MLEBEvaluationModelConfig(
        id="mixedbread-ai/mxbai-embed-large-v1",
        provider="huggingface",
        model_framework="sentence-transformer",
        model_type="embedder",
        batch_size=1,
        mteb_metadata=None,
        dtype="float16",
    ),
    # | Google
    MLEBEvaluationModelConfig(
        id="models/gemini-embedding-001",
        provider="google",
        model_framework="langchain",
        model_type="embedder",
        batch_size=16,
        mteb_metadata=requests.get(
            "https://raw.githubusercontent.com/embeddings-benchmark/results/refs/heads/main/results/google__gemini-embedding-001/1/model_meta.json"
        ).json(),
    ),
    MLEBEvaluationModelConfig(
        id="google/embeddinggemma-300m",
        provider="huggingface",
        model_framework="sentence-transformer",
        model_type="embedder",
        batch_size=1,
        mteb_metadata=None,
    ),
    # | Snowflake
    MLEBEvaluationModelConfig(
        id="Snowflake/snowflake-arctic-embed-l-v2.0",
        provider="huggingface",
        model_framework="sentence-transformer",
        model_type="embedder",
        batch_size=1,
        mteb_metadata=None,
        trust_remote_code=True,
    ),
    MLEBEvaluationModelConfig(
        id="Snowflake/snowflake-arctic-embed-m-v2.0",
        provider="huggingface",
        model_framework="sentence-transformer",
        model_type="embedder",
        batch_size=1,
        mteb_metadata=None,
        trust_remote_code=True,
    ),
    MLEBEvaluationModelConfig(
        id="Snowflake/snowflake-arctic-embed-l",
        provider="huggingface",
        model_framework="sentence-transformer",
        model_type="embedder",
        batch_size=1,
        mteb_metadata=None,
        trust_remote_code=True,
    ),
    # | Voyage
    MLEBEvaluationModelConfig(
        id="voyage-3-large",
        provider="voyage",
        model_framework="langchain",
        model_type="embedder",
        batch_size=16,
        mteb_metadata=requests.get(
            "https://raw.githubusercontent.com/embeddings-benchmark/results/refs/heads/main/results/voyageai__voyage-3-large/1/model_meta.json"
        ).json()
        | {"loader": None},
    ),
    MLEBEvaluationModelConfig(
        id="voyage-3.5",
        provider="voyage",
        model_framework="langchain",
        model_type="embedder",
        batch_size=16,
        mteb_metadata={
            "name": "voyageai/voyage-3.5",
            "revision": "1",
            "release_date": "2025-05-20",
            "languages": None,
            "n_parameters": None,
            "memory_usage_mb": None,
            "max_tokens": 32000.0,
            "embed_dim": 1024,
            "license": None,
            "open_weights": None,
            "public_training_data": None,
            "public_training_code": None,
            "framework": ["API"],
            "reference": "https://blog.voyageai.com/2025/05/20/voyage-3-5/",
            "similarity_fn_name": "cosine",
            "use_instructions": None,
            "training_datasets": None,
            "adapted_from": None,
            "superseded_by": None,
            "loader": None,
        },
    ),
    MLEBEvaluationModelConfig(
        id="voyage-3.5-lite",
        provider="voyage",
        model_framework="langchain",
        model_type="embedder",
        batch_size=16,
        mteb_metadata={
            "name": "voyageai/voyage-3.5-lite",
            "revision": "1",
            "release_date": "2025-05-20",
            "languages": None,
            "n_parameters": None,
            "memory_usage_mb": None,
            "max_tokens": 32000.0,
            "embed_dim": 1024,
            "license": None,
            "open_weights": None,
            "public_training_data": None,
            "public_training_code": None,
            "framework": ["API"],
            "reference": "https://blog.voyageai.com/2025/05/20/voyage-3-5/",
            "similarity_fn_name": "cosine",
            "use_instructions": None,
            "training_datasets": None,
            "adapted_from": None,
            "superseded_by": None,
            "loader": None,
        },
    ),
    MLEBEvaluationModelConfig(
        id="voyage-law-2",
        provider="voyage",
        model_framework="langchain",
        model_type="embedder",
        batch_size=16,
        mteb_metadata=requests.get(
            "https://raw.githubusercontent.com/embeddings-benchmark/results/refs/heads/main/results/voyageai__voyage-law-2/1/model_meta.json"
        ).json()
        | {"loader": None},
    ),
    # | OpenAI
    MLEBEvaluationModelConfig(
        id="text-embedding-3-large",
        provider="openai",
        model_framework="langchain",
        model_type="embedder",
        batch_size=16,
        mteb_metadata=requests.get(
            "https://raw.githubusercontent.com/embeddings-benchmark/results/refs/heads/main/results/openai__text-embedding-3-large/1/model_meta.json"
        ).json()
        | {"loader": None},
    ),
    MLEBEvaluationModelConfig(
        id="text-embedding-3-small",
        provider="openai",
        model_framework="langchain",
        model_type="embedder",
        batch_size=16,
        mteb_metadata=requests.get(
            "https://raw.githubusercontent.com/embeddings-benchmark/results/refs/heads/main/results/openai__text-embedding-3-small/1/model_meta.json"
        ).json()
        | {"loader": None},
    ),
    MLEBEvaluationModelConfig(
        id="text-embedding-ada-002",
        provider="openai",
        model_framework="langchain",
        model_type="embedder",
        batch_size=16,
        mteb_metadata=requests.get(
            "https://raw.githubusercontent.com/embeddings-benchmark/results/refs/heads/main/results/openai__text-embedding-ada-002/2/model_meta.json"
        ).json()
        | {"loader": None},
    ),
    # | IBM
    MLEBEvaluationModelConfig(
        id="ibm-granite/granite-embedding-english-r2",
        provider="huggingface",
        model_framework="sentence-transformer",
        model_type="embedder",
        batch_size=1,
        mteb_metadata=None,
        amp_dtype="bfloat16",
    ),
    MLEBEvaluationModelConfig(
        id="ibm-granite/granite-embedding-small-english-r2",
        provider="huggingface",
        model_framework="sentence-transformer",
        model_type="embedder",
        batch_size=1,
        mteb_metadata=None,
        amp_dtype="bfloat16",
    ),
]

MODEL_CONFIGS = {model_config.id: model_config for model_config in MODEL_CONFIGS}