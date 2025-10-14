from typing import Literal

import msgspec


class MLEBEvaluationDatasetConfig(msgspec.Struct):
    name: str
    """The name of the MLEB evaluation dataset."""

    id: str
    """The ID or path of the MLEB evaluation dataset."""

    revision: str = "main"
    """The revision of the MLEB evaluation dataset. Defaults to `main`."""

    main_score: str = "ndcg_at_10"
    """The main score to use for the MLEB evaluation dataset. Defaults to `ndcg_at_10`."""


class MLEBEvaluationModelConfig(msgspec.Struct):
    id: str
    """The model's ID."""

    provider: Literal["huggingface", "google", "isaacus", "voyage", "openai", "cohere"]
    """The model's provider."""

    model_framework: Literal["sentence-transformer", "langchain", "isaacus"]
    """The model's framework."""
    
    model_type: Literal["embedder", "reranker"]
    """The model's type."""

    batch_size: int
    """The batch size to use when encoding sentences with the model."""
    
    mteb_metadata: dict | None
    """MTEB metadata for the model. `None` can only be used if the model type is `sentence-transformer`."""
    
    dtype: Literal["float32", "float16", "bfloat16", None] = None
    """The dtype to use when encoding sentences with the model. Defaults to `None`, which does not cast the embeddings. No other options can be used if the `model_type` is not `sentence-transformer`."""
    
    amp_dtype: Literal["float16", "bfloat16", False] = False
    """The dtype to use for automatic mixed precision (AMP) when encoding sentences with the model. Defaults to `False`, in which case, AMP is not used."""
    
    trust_remote_code: bool = False
    """Whether to trust remote code when loading the model. Defaults to `False`."""
    
    def __post_init__(self):
        if not self.mteb_metadata and self.model_framework != "sentence-transformer":
            raise ValueError("`mteb_metadata` must be provided if the model type is not `sentence-transformer`.")
        
        if self.dtype and self.model_framework != "sentence-transformer":
            raise ValueError("`dtype` can only be set if the model type is `sentence-transformer`.")