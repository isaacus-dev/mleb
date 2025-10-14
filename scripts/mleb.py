import gc
import os
import logging
import itertools

from contextlib import nullcontext

import mteb
import numpy as np
import torch
import isaacus

from mteb import ModelMeta, TaskMetadata
from tqdm import tqdm
from dotenv import load_dotenv
from models import MODEL_CONFIGS
from structs import MLEBEvaluationModelConfig, MLEBEvaluationDatasetConfig
from mteb.overview import TASKS_REGISTRY
from sentence_transformers import SentenceTransformer
from mteb.encoder_interface import PromptType

logger = logging.getLogger(__name__)

MODEL_IDS = [
    # Embedders
    # | Isaacus
    "kanon-2-embedder",
    # | Voyage
    "voyage-3-large",
    "voyage-3.5",
    "voyage-3.5-lite",
    "voyage-law-2",
    # | Qwen
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
    # | BGE
    "BAAI/bge-m3",
    # | Microsoft
    "intfloat/multilingual-e5-large-instruct",
    # | Mixedbread
    "mixedbread-ai/mxbai-embed-large-v1",
    # | Google
    "models/gemini-embedding-001",
    "google/embeddinggemma-300m",
    # | Snowflake
    "Snowflake/snowflake-arctic-embed-l-v2.0",
    "Snowflake/snowflake-arctic-embed-m-v2.0",
    # | OpenAI
    "text-embedding-3-large",
    "text-embedding-3-small",
    "text-embedding-ada-002",
    # | IBM
    "ibm-granite/granite-embedding-english-r2",
    "ibm-granite/granite-embedding-small-english-r2",
]

EVALUATION_DATASET_CONFIGS = (
    MLEBEvaluationDatasetConfig(
        name="bar-exam-qa",
        id="isaacus/mteb-barexam-qa",
        revision="dd157bbfa479359488c656981e3999da6f42e4e9",
    ),
    MLEBEvaluationDatasetConfig(
        name="scalr",
        id="isaacus/mleb-scalr",
        revision="319b6cc4b012d733f126a943a8a66bdf9df5dc40",
    ),
    MLEBEvaluationDatasetConfig(
        name="singaporean-judicial-keywords",
        id="isaacus/singaporean-judicial-keywords",
        revision="427e2ae4b22cd9ad990ef8dd4647c16d79c89198",
    ),
    MLEBEvaluationDatasetConfig(
        name="gdpr-holdings-retrieval",
        id="isaacus/gdpr-holdings-retrieval",
        revision="8d41f3d22bb73685b6f42b62ad95940ea3dfbf84",
    ),
    MLEBEvaluationDatasetConfig(
        name="uk-legislative-long-titles",
        id="isaacus/uk-legislative-long-titles",
        revision="436d6a79d06cac556799e9e0be54a6fb90bf7182",
    ),
    MLEBEvaluationDatasetConfig(
        name="australian-tax-guidance-retrieval",
        id="isaacus/australian-tax-guidance-retrieval",
        revision="c64c3baac6bfd5f934d2df6e5d42dcb7d87c8ba8",
    ),
    MLEBEvaluationDatasetConfig(
        name="irish-legislative-summaries",
        id="isaacus/irish-legislative-summaries",
        revision="bbf8b2d84b7de5970b2ba4ea843c791285fdb1df",
    ),
    MLEBEvaluationDatasetConfig(
        name="contractual-clause-retrieval",
        id="isaacus/contractual-clause-retrieval",
        revision="48ed7bcb1f50896a0f71461a04b2df0ca84329d9",
    ),
    MLEBEvaluationDatasetConfig(
        name="license-tldr-retrieval",
        id="isaacus/license-tldr-retrieval",
        revision="ec00129f88e9476e582131dc3a5db9220dfefa48",
    ),
    MLEBEvaluationDatasetConfig(
        name="consumer-contracts-qa",
        id="isaacus/mleb-consumer-contracts-qa",
        revision="2095f248902963b4480ac96b774ba64b2104cbee",
    ),
)


def _get_mteb_task(dataset_config: MLEBEvaluationDatasetConfig) -> mteb.AbsTaskRetrieval:
    """Resolve an MLEB evaluation dataset config to an MTEB task."""

    if dataset_config.name in TASKS_REGISTRY:
        return mteb.get_task(dataset_config.name)

    class MLEBEvaluationTaskRetrieval(mteb.AbsTaskRetrieval):
        metadata = TaskMetadata(
            name=dataset_config.name,
            dataset={
                "path": dataset_config.id,
                "revision": dataset_config.revision,
            },
            main_score=dataset_config.main_score,
            # Supply dummy values for the rest of the metadata.
            description="An MLEB evaluation dataset.",
            type="Retrieval",
            category="t2t",
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["eng-Latn"],
            date=("2021-06-06", "2025-07-28"),
            domains=["Legal"],
            task_subtypes=[],
            license="cc-by-4.0",
            annotations_creators="expert-annotated",
            dialect=[],
            sample_creation="found",
        )

    TASKS_REGISTRY[dataset_config.name] = MLEBEvaluationTaskRetrieval

    return mteb.get_task(dataset_config.name)


class MTEBEmbedderForLangchain(torch.nn.Module):
    """An MTEB-compatible model wrapper for LangChain-compatible embedding models."""

    def __init__(self, model_config: MLEBEvaluationModelConfig) -> None:
        load_dotenv()

        self.model_config = model_config

        match self.model_config.provider:
            case "google":
                from langchain_google_genai import GoogleGenerativeAIEmbeddings

                self.client = GoogleGenerativeAIEmbeddings(model=self.model_config.id)

            case "openai":
                from langchain_openai import OpenAIEmbeddings

                self.client = OpenAIEmbeddings(model=self.model_config.id)

            case "cohere":
                from langchain_cohere import CohereEmbeddings

                self.client = CohereEmbeddings(model=self.model_config.id)

            case "voyage":
                from langchain_voyageai import VoyageAIEmbeddings

                self.client = VoyageAIEmbeddings(model=self.model_config.id)

            case _:
                raise ValueError(f"Unsupported model provider: {self.model_config.provider}")

        # Fix old metadata field names.
        if "memory_usage" in self.model_config.mteb_metadata:
            self.model_config.mteb_metadata["memory_usage_mb"] = self.model_config.mteb_metadata.pop("memory_usage")

        if "use_instuctions" in self.model_config.mteb_metadata:
            self.model_config.mteb_metadata["use_instructions"] = self.model_config.mteb_metadata.pop("use_instuctions")

        if "zero_shot_benchmarks" in self.model_config.mteb_metadata:
            self.model_config.mteb_metadata["training_datasets"] = self.model_config.mteb_metadata.pop(
                "zero_shot_benchmarks"
            )

        self.mteb_model_meta = ModelMeta(**self.model_config.mteb_metadata)

    def encode(
        self, sentences: list[str], prompt_type: PromptType | None = None, convert_to_tensor: bool = False, **kwargs
    ) -> np.ndarray | torch.Tensor:
        match prompt_type:
            case PromptType.query:
                embeddings = [self.client.embed_query(sentence) for sentence in sentences]

            case PromptType.document:
                embeddings = [
                    embedding
                    for batch in itertools.batched(sentences, self.model_config.batch_size)
                    for embedding in self.client.embed_documents(batch)
                ]

            case _:
                raise ValueError(f"Unsupported prompt type: {prompt_type}")

        if convert_to_tensor:
            return torch.tensor(embeddings)

        return np.array(embeddings)


class MTEBEmbedderForIsaacus(torch.nn.Module):
    """An MTEB-compatible model wrapper for Isaacus embedding models."""

    def __init__(self, model_config: MLEBEvaluationModelConfig) -> None:
        load_dotenv()

        self.model_config = model_config

        api_key = None
        base_url = None

        if os.getenv("ISAACUS_ENV") == "dev":
            api_key = os.getenv("ISAACUS_DEV_API_KEY")
            base_url = os.getenv("ISAACUS_DEV_BASE_URL")

        self.client = isaacus.Isaacus(
            max_retries=10,
            api_key=api_key,
            base_url=base_url,
        )

        if self.model_config.mteb_metadata:
            self.mteb_model_meta = ModelMeta(**self.model_config.mteb_metadata)

    def _get_task_type(
        self,
        prompt_type: PromptType | None,
    ) -> str:
        match prompt_type:
            case PromptType.query:
                return "retrieval/query"

            case PromptType.document:
                return "retrieval/document"

            case _:
                raise ValueError(f"`MTEBEmbedderForIsaacus` does not support the prompt type `{prompt_type}`.")

    def encode(
        self, sentences: list[str], prompt_type: PromptType | None = None, convert_to_tensor: bool = False, **kwargs
    ) -> np.ndarray | torch.Tensor:
        task_type = self._get_task_type(prompt_type)

        embeddings = [
            embedding.embedding
            for batch in itertools.batched(
                sentences, self.model_config.batch_size if task_type == "retrieval/document" else 1
            )  # NOTE It is actually much less efficient to not be batching queries, however, we do so to ensure inference time comparisons are fair with other models using the Langchain API which, for whatever reason, unfortunately, does not support batching queries.
            for embedding in self.client.embeddings.create(
                model=self.model_config.id,
                texts=batch,
                task=task_type,
            ).embeddings
        ]

        if convert_to_tensor:
            embeddings = torch.tensor(embeddings)

        else:
            embeddings = np.array(embeddings)

        return embeddings


def _get_mteb_evaluator(model_config: MLEBEvaluationModelConfig) -> torch.nn.Module:
    """Get an MTEB-compatible evaluator for the model."""

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    match model_config.model_framework:
        case "isaacus":
            match model_config.model_type:
                case "embedder":
                    return MTEBEmbedderForIsaacus(model_config)

        case "langchain":
            match model_config.model_type:
                case "embedder":
                    return MTEBEmbedderForLangchain(model_config)

                case _:
                    raise ValueError(f"Unsupported model type for langchain framework: {model_config.model_type}")

        case "sentence-transformer":
            match model_config.model_type:
                case "embedder":
                    model = SentenceTransformer(
                        model_config.id, trust_remote_code=model_config.trust_remote_code
                    ).eval()

                    match model_config.dtype:
                        case "float32":
                            model = model.float()

                        case "float16":
                            model = model.half()

                        case "bfloat16":
                            model = model.bfloat16()

                        case None:
                            pass

                        case _:
                            raise ValueError(f"Unsupported dtype: {model_config.dtype}")

                    original_encode = model.encode
                    autocast_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, False: None}[
                        model_config.amp_dtype
                    ]

                    if autocast_dtype:
                        autocast = torch.autocast(device_type="cuda", dtype=autocast_dtype)

                    else:
                        autocast = nullcontext()

                    def _encode(
                        self,
                        *args,
                        **kwargs,
                    ) -> np.ndarray | torch.Tensor:
                        if model_config.batch_size:
                            kwargs["batch_size"] = model_config.batch_size

                        if "show_progress_bar" in kwargs:
                            kwargs["show_progress_bar"] = False

                        with torch.inference_mode(), autocast:
                            emb: np.ndarray | torch.Tensor = original_encode(*args, **kwargs)

                        if isinstance(emb, np.ndarray):
                            emb = emb.astype(np.float32)

                        elif isinstance(emb, torch.Tensor):
                            emb = emb.to(torch.float32)

                        return emb

                    model.encode = _encode.__get__(model, SentenceTransformer)

                    return model

                case _:
                    raise ValueError(
                        f"Unsupported model type for sentence-transformer framework: {model_config.model_type}"
                    )

        case _:
            raise ValueError(f"Unsupported model framework: {model_config.model_framework}")


def evaluate_model(
    model_config: MLEBEvaluationModelConfig,
    dataset_configs: list[MLEBEvaluationDatasetConfig] = EVALUATION_DATASET_CONFIGS,
    output_dir: str | None = "results",
    progress: bool = True,
) -> dict[str, dict[str, float]]:
    """Evaluate a model on the MLEB evaluation datasets.

    Args:
        model_config: The model to evaluate.
        dataset_configs: The datasets to evaluate on. Defaults to all MLEB evaluation datasets.
        output_dir: The directory to save the results to. If None, results are not saved. Defaults to 'results'.

    Returns:
        A dictionary mapping dataset names to their evaluation results.
    """

    # Load the model.
    model = _get_mteb_evaluator(model_config)

    # Evaluate on each dataset.
    results = {}

    for dataset_config in tqdm(dataset_configs, desc="Evaluating MLEB datasets", unit=" dataset", disable=not progress):
        logger.info(f"Evaluating on dataset: {dataset_config.name}...")

        task = _get_mteb_task(dataset_config)

        task_results = list(
            mteb.MTEB(tasks=[task])
            .run(model, output_folder=output_dir, verbosity=0, progress_bar=False)[0]
            .scores.values()
        )[0][0]

        set_scores = {
            k: float(v) for k, v in task_results.items() if isinstance(v, (int, float, np.integer, np.floating))
        }

        results[dataset_config.name] = set_scores

    return results


if __name__ == "__main__":
    for model_id in MODEL_IDS:
        print(f"Evaluating model: {model_id}...")
        model_config = MODEL_CONFIGS[model_id]

        results = evaluate_model(model_config)

        gc.collect()
        torch.cuda.empty_cache()

        print(f"Results for {model_config.id}:")

        print(f"  Avg: {np.mean([dscores['main_score'] for dscores in results.values()]):.4f}\n")

        for dataset_name, dataset_scores in results.items():
            print(f"  {dataset_name}: {dataset_scores['main_score']:.4f}")

        print("")
