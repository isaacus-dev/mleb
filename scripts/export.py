from enum import Enum
from typing import Annotated

import orjson
import msgspec

from config import results_dir
from helpers import save_jsonl


class MLEBDatasetDocumentCategory(Enum):
    REGULATORY = "Regulation"
    JUDICIAL = "Caselaw"
    CONTRACTUAL = "Contracts"


class MLEBDataset(msgspec.Struct):
    id: Annotated[str, msgspec.Meta(examples=["singaporean-judicial-keywords"])]
    name: Annotated[str, msgspec.Meta(examples=["Singaporean Judicial Keywords"])]
    creator: Annotated[str, msgspec.Meta(examples=["Isaacus"])]
    category: Annotated[MLEBDatasetDocumentCategory, msgspec.Meta(examples=["Caselaw"])]
    link: Annotated[
        str, msgspec.Meta(examples=["https://huggingface.co/datasets/isaacus/singaporean-judicial-keywords"])
    ]


class MLEBModelProvider(msgspec.Struct):
    id: Annotated[str, msgspec.Meta(examples=["voyageai"])]
    name: Annotated[str, msgspec.Meta(examples=["Voyage AI"])]
    link: Annotated[str | None, msgspec.Meta(examples=["https://voyageai.com/"])] = None
    logo: Annotated[str | None, msgspec.Meta(examples=["https://www.voyageai.com/logo-v2.svg"])] = None


class MLEBModel(msgspec.Struct):
    id: Annotated[str, msgspec.Meta(examples=["voyage-3-large"])]
    name: Annotated[str, msgspec.Meta(examples=["Voyage 3 Large"])]
    provider: MLEBModelProvider
    open_source: Annotated[bool, msgspec.Meta(examples=[False])]
    embedding_dimensions: Annotated[int, msgspec.Meta(examples=[4096])]
    context_window: Annotated[int, msgspec.Meta(examples=[32768])]
    link: Annotated[str | None, msgspec.Meta(examples=["https://blog.voyageai.com/2025/01/07/voyage-3-large/"])] = None


class MLEBResult(msgspec.Struct):
    dataset: MLEBDataset
    score: Annotated[float, msgspec.Meta(examples=[0.6131], description="NDCG@10 score")]
    time_taken: Annotated[
        float,
        msgspec.Meta(examples=[123.45], description="Time taken in seconds to evaluate the model on this dataset."),
    ]


class MLEBModelReport(msgspec.Struct):
    model: MLEBModel
    results: list[MLEBResult]


MLEB_DATASETS = [
    MLEBDataset(
        id="bar-exam-qa",
        name="Bar Exam QA",
        creator="Stanford University",
        category=MLEBDatasetDocumentCategory.JUDICIAL,
        link="https://huggingface.co/datasets/isaacus/mteb-barexam-qa",
    ),
    MLEBDataset(
        id="scalr",
        name="SCALR",
        creator="Faiz Surani and Varun Iyer",
        category=MLEBDatasetDocumentCategory.JUDICIAL,
        link="https://huggingface.co/datasets/isaacus/mleb-scalr",
    ),
    MLEBDataset(
        id="singaporean-judicial-keywords",
        name="Singaporean Judicial Keywords",
        creator="Isaacus",
        category=MLEBDatasetDocumentCategory.JUDICIAL,
        link="https://huggingface.co/datasets/isaacus/singaporean-judicial-keywords",
    ),
    MLEBDataset(
        id="gdpr-holdings-retrieval",
        name="GDPR Holdings Retrieval",
        creator="Isaacus",
        category=MLEBDatasetDocumentCategory.JUDICIAL,
        link="https://huggingface.co/datasets/isaacus/gdpr-holdings-retrieval",
    ),
    MLEBDataset(
        id="australian-tax-guidance-retrieval",
        name="Australian Tax Guidance Retrieval",
        creator="Isaacus",
        category=MLEBDatasetDocumentCategory.REGULATORY,
        link="https://huggingface.co/datasets/isaacus/australian-tax-guidance-retrieval",
    ),
    MLEBDataset(
        id="irish-legislative-summaries",
        name="Irish Legislative Summaries",
        creator="Isaacus",
        category=MLEBDatasetDocumentCategory.REGULATORY,
        link="https://huggingface.co/datasets/isaacus/irish-legislative-summaries",
    ),
    MLEBDataset(
        id="uk-legislative-long-titles",
        name="UK Legislative Long Titles",
        creator="Isaacus",
        category=MLEBDatasetDocumentCategory.REGULATORY,
        link="https://huggingface.co/datasets/isaacus/uk-legislative-long-titles",
    ),
    MLEBDataset(
        id="contractual-clause-retrieval",
        name="Contractual Clause Retrieval",
        creator="Isaacus",
        category=MLEBDatasetDocumentCategory.CONTRACTUAL,
        link="https://huggingface.co/datasets/isaacus/contractual-clause-retrieval",
    ),
    MLEBDataset(
        id="license-tldr-retrieval",
        name="License TL;DR Retrieval",
        creator="Isaacus",
        category=MLEBDatasetDocumentCategory.CONTRACTUAL,
        link="https://huggingface.co/datasets/isaacus/license-tldr-retrieval",
    ),
    MLEBDataset(
        id="consumer-contracts-qa",
        name="Consumer Contracts QA",
        creator="Noam Kolt",
        category=MLEBDatasetDocumentCategory.CONTRACTUAL,
        link="https://huggingface.co/datasets/isaacus/mleb-consumer-contracts-qa",
    ),
]

MLEB_MODEL_PROVIDERS = [
    MLEBModelProvider(
        id="isaacus",
        name="Isaacus",
        link="https://isaacus.com/",
        logo="https://media.isaacus.com/third-parties/icons/isaacus.png",
    ),
    MLEBModelProvider(
        id="microsoft",
        name="Microsoft",
        link="https://www.microsoft.com/en-us/research/",
        logo="https://media.isaacus.com/third-parties/icons/microsoft.png",
    ),
    MLEBModelProvider(
        id="openai",
        name="OpenAI",
        link="https://openai.com/",
        logo="https://media.isaacus.com/third-parties/icons/openai.png",
    ),
    MLEBModelProvider(
        id="google",
        name="Google",
        link="https://ai.google.dev/",
        logo="https://media.isaacus.com/third-parties/icons/google.png",
    ),
    MLEBModelProvider(
        id="ibm",
        name="IBM",
        link="https://www.ibm.com/artificial-intelligence",
        logo="https://media.isaacus.com/third-parties/icons/ibm.png",
    ),
    MLEBModelProvider(
        id="snowflake",
        name="Snowflake",
        link="https://www.snowflake.com/en/",
        logo="https://media.isaacus.com/third-parties/icons/snowflake.png",
    ),
    MLEBModelProvider(
        id="qwen",
        name="Qwen",
        link="https://qwen.ai/",
        logo="https://media.isaacus.com/third-parties/icons/qwen.png",
    ),
    MLEBModelProvider(
        id="baai",
        name="BAAI",
        link="https://www.baai.ac.cn/en/",
        logo="https://media.isaacus.com/third-parties/icons/baai.png",
    ),
    MLEBModelProvider(
        id="voyageai",
        name="Voyage",
        link="https://voyageai.com/",
        logo="https://media.isaacus.com/third-parties/icons/voyage.png",
    ),
    MLEBModelProvider(
        id="mixedbread-ai",
        name="Mixedbread",
        link="https://www.mixedbread.com/",
        logo="https://media.isaacus.com/third-parties/icons/mixedbread.png",
    ),
    MLEBModelProvider(
        id="jinaai",
        name="Jina",
        link="http://jina.ai/",
        logo="https://media.isaacus.com/third-parties/icons/jina.png",
    ),
]

MLEB_MODEL_PROVIDERS = {p.id: p for p in MLEB_MODEL_PROVIDERS}

MLEB_MODELS = [
    # Embedders
    # | Isaacus
    MLEBModel(
        id="isaacus/kanon-2-embedder",
        name="Kanon 2 Embedder",
        provider=MLEB_MODEL_PROVIDERS["isaacus"],
        open_source=False,
        embedding_dimensions=1792,
        context_window=16384,
        link="https://docs.isaacus.com/models/introduction#embedding",
    ),
    # | Voyage
    MLEBModel(
        id="voyageai/voyage-3-large",
        name="Voyage 3 Large",
        provider=MLEB_MODEL_PROVIDERS["voyageai"],
        open_source=False,
        embedding_dimensions=1024,
        context_window=32_000,
        link="https://blog.voyageai.com/2025/01/07/voyage-3-large/",
    ),
    MLEBModel(
        id="voyageai/voyage-3.5",
        name="Voyage 3.5",
        provider=MLEB_MODEL_PROVIDERS["voyageai"],
        open_source=False,
        embedding_dimensions=1024,
        context_window=32_000,
        link="https://blog.voyageai.com/2025/05/20/voyage-3-5/",
    ),
    MLEBModel(
        id="voyageai/voyage-3.5-lite",
        name="Voyage 3.5 Lite",
        provider=MLEB_MODEL_PROVIDERS["voyageai"],
        open_source=False,
        embedding_dimensions=1024,
        context_window=32_000,
        link="https://blog.voyageai.com/2025/05/20/voyage-3-5/",
    ),
    MLEBModel(
        id="voyageai/voyage-law-2",
        name="Voyage Law 2",
        provider=MLEB_MODEL_PROVIDERS["voyageai"],
        open_source=False,
        embedding_dimensions=1024,
        context_window=16_000,
        link="https://blog.voyageai.com/2024/04/15/domain-specific-embeddings-and-retrieval-legal-edition-voyage-law-2/",
    ),
    # | Qwen
    MLEBModel(
        id="Qwen/Qwen3-Embedding-0.6B",
        name="Qwen3 Embedding 0.6B",
        provider=MLEB_MODEL_PROVIDERS["qwen"],
        open_source=True,
        embedding_dimensions=1024,
        context_window=32_000,
        link="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
    ),
    MLEBModel(
        id="Qwen/Qwen3-Embedding-4B",
        name="Qwen3 Embedding 4B",
        provider=MLEB_MODEL_PROVIDERS["qwen"],
        open_source=True,
        embedding_dimensions=2560,
        context_window=32_000,
        link="https://huggingface.co/Qwen/Qwen3-Embedding-4B",
    ),
    MLEBModel(
        id="Qwen/Qwen3-Embedding-8B",
        name="Qwen3 Embedding 8B",
        provider=MLEB_MODEL_PROVIDERS["qwen"],
        open_source=True,
        embedding_dimensions=4096,
        context_window=32_000,
        link="https://huggingface.co/Qwen/Qwen3-Embedding-8B",
    ),
    # | BAAI
    MLEBModel(
        id="BAAI/bge-m3",
        name="BGE M3",
        provider=MLEB_MODEL_PROVIDERS["baai"],
        open_source=True,
        embedding_dimensions=1024,
        context_window=8_192,
        link="https://huggingface.co/BAAI/bge-m3",
    ),
    # | Microsoft
    MLEBModel(
        id="intfloat/multilingual-e5-large-instruct",
        name="E5 Large Instruct",
        provider=MLEB_MODEL_PROVIDERS["microsoft"],
        open_source=True,
        embedding_dimensions=1024,
        context_window=512,
        link="https://huggingface.co/intfloat/multilingual-e5-large-instruct",
    ),
    # | Mixedbread
    MLEBModel(
        id="mixedbread-ai/mxbai-embed-large-v1",
        name="Mxbai Embed Large v1",
        provider=MLEB_MODEL_PROVIDERS["mixedbread-ai"],
        open_source=True,
        embedding_dimensions=1024,
        context_window=512,
        link="https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1",
    ),
    # | Google
    MLEBModel(
        id="google/gemini-embedding-001",
        name="Gemini Embedding",
        provider=MLEB_MODEL_PROVIDERS["google"],
        open_source=False,
        embedding_dimensions=3072,
        context_window=8_192,
        link="https://developers.googleblog.com/en/gemini-embedding-available-gemini-api/",
    ),
    MLEBModel(
        id="google/embeddinggemma-300m",
        name="EmbeddingGemma",
        provider=MLEB_MODEL_PROVIDERS["google"],
        open_source=True,
        embedding_dimensions=768,
        context_window=2_048,
        link="https://huggingface.co/google/embeddinggemma-300m",
    ),
    # | Snowflake
    MLEBModel(
        id="Snowflake/snowflake-arctic-embed-l-v2.0",
        name="Snowflake Arctic Embed L v2.0",
        provider=MLEB_MODEL_PROVIDERS["snowflake"],
        open_source=True,
        embedding_dimensions=1024,
        context_window=8_192,
        link="https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0",
    ),
    MLEBModel(
        id="Snowflake/snowflake-arctic-embed-m-v2.0",
        name="Snowflake Arctic Embed M v2.0",
        provider=MLEB_MODEL_PROVIDERS["snowflake"],
        open_source=True,
        embedding_dimensions=768,
        context_window=8_192,
        link="https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0",
    ),
    # | OpenAI
    MLEBModel(
        id="openai/text-embedding-3-large",
        name="Text Embedding 3 Large",
        provider=MLEB_MODEL_PROVIDERS["openai"],
        open_source=False,
        embedding_dimensions=3072,
        context_window=8_192,
        link="https://platform.openai.com/docs/models/text-embedding-3-large",
    ),
    MLEBModel(
        id="openai/text-embedding-3-small",
        name="Text Embedding 3 Small",
        provider=MLEB_MODEL_PROVIDERS["openai"],
        open_source=False,
        embedding_dimensions=1536,
        context_window=8_192,
        link="https://platform.openai.com/docs/models/text-embedding-3-small",
    ),
    MLEBModel(
        id="openai/text-embedding-ada-002",
        name="Text Embedding Ada 002",
        provider=MLEB_MODEL_PROVIDERS["openai"],
        open_source=False,
        embedding_dimensions=1536,
        context_window=8_192,
        link="https://platform.openai.com/docs/models/text-embedding-ada-002",
    ),
    # | IBM
    MLEBModel(
        id="ibm-granite/granite-embedding-english-r2",
        name="Granite Embedding English R2",
        provider=MLEB_MODEL_PROVIDERS["ibm"],
        open_source=True,
        embedding_dimensions=768,
        context_window=8_192,
        link="https://huggingface.co/ibm-granite/granite-embedding-english-r2",
    ),
    MLEBModel(
        id="ibm-granite/granite-embedding-small-english-r2",
        name="Granite Embedding Small English R2",
        provider=MLEB_MODEL_PROVIDERS["ibm"],
        open_source=True,
        embedding_dimensions=384,
        context_window=8_192,
        link="https://huggingface.co/ibm-granite/granite-embedding-small-english-r2",
    ),
    # | Jina
    MLEBModel(
        id="jinaai/jina-embeddings-v4",
        name="Jina Embeddings v4",
        provider=MLEB_MODEL_PROVIDERS["jinaai"],
        open_source=True,
        embedding_dimensions=2048,
        context_window=32_768,
        link="https://huggingface.co/jinaai/jina-embeddings-v4",
    ),
]


def export() -> None:
    reports: list[MLEBModelReport] = []

    for model in MLEB_MODELS:
        model_results_dir = results_dir / model.id.replace("/", "__")

        if not model_results_dir.exists():
            assert FileNotFoundError(f'No results found for model "{model.id}" in "{model_results_dir}".')

        model_version_dirs = list(model_results_dir.iterdir())

        if not model_version_dirs:
            raise FileNotFoundError(
                f'No results found for model "{model.id}" in "{model_results_dir}". The directory exists but there is no subdirectory.'
            )

        if len(model_version_dirs) > 1:
            raise ValueError(
                f'Multiple version directories found for model "{model.id}" in "{model_results_dir}". Expected only one subdirectory.'
            )

        model_results_dir = model_version_dirs[0]

        results: list[MLEBResult] = []

        for dataset in MLEB_DATASETS:
            dataset_results_file = model_results_dir / f"{dataset.id}.json"

            if not dataset_results_file.exists():
                raise FileNotFoundError(f'No results found for dataset "{dataset.id}" in "{dataset_results_file}".')

            dataset_results = orjson.loads(dataset_results_file.read_bytes().replace(b"NaN", b"null"))

            dataset_result = MLEBResult(
                dataset=dataset,
                score=dataset_results["scores"]["test"][0]["main_score"],
                time_taken=dataset_results["evaluation_time"],
            )
            results.append(dataset_result)

        report = MLEBModelReport(
            model=model,
            results=results,
        )

        reports.append(report)

    save_jsonl(results_dir / "results.jsonl", reports)


if __name__ == "__main__":
    export()
