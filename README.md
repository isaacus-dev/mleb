# Massive Legal Embedding Benchmark (MLEB)
The [**Massive Legal Embedding Benchmark (MLEB)**](https://isaacus.com/mleb) by [Isaacus](https://isaacus.com/) is the largest, most diverse, and most comprehensive benchmark for legal text embedding models. It contains 10 datasets spanning multiple document types, jurisdictions, areas of law, and task types. To do well on MLEB, embedding models must demonstrate strong legal domain knowledge and legal reasoning skills.

This repository contains the code used to evaluate embedding models on MLEB (available in the [`scripts`](./scripts/) directory), as well as the full results of evaluated models (available in the [`results`](./results/) directory).

## Setup
We recommend setting up a virtual environment for this project and installing necessary dependencies using [`uv`](https://docs.astral.sh/uv/) like so:

```bash
git clone https://github.com/isaacus-dev/mleb.git
cd mleb
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

That will download this repository, create a Python 3.12 virtual environment using [`uv`](https://docs.astral.sh/uv/), activate the virtual environment, and install all necessary dependencies.

Alternatively, you may manually install the necessary dependencies listed in our [`pyproject.toml`](./pyproject.toml) file.

After installing the necessary dependencies, we recommend creating a `.env` file in the root directory of this repository to store your API keys for various embedding model providers. You can use the provided [`.env.example`](./.env.example) file as a template:

```env
# Isaacus
ISAACUS_API_KEY=...

# OpenAI
OPENAI_API_KEY=...

# Google
GOOGLE_API_KEY=...

# Voyage AI
VOYAGE_API_KEY=...
```

Make sure to replace the `...` with your actual API keys. You may omit any keys for providers you won't be using.

## Usage
To evaluate embedding models on MLEB, you can simply run the [`scripts/mleb.py`](./scripts/mleb.py) script, like so:

```bash
python scripts/mleb.py
```

Inside the script, you can specify which specific models you want to evaluate by modifying the `MODEL_IDS` list near the top of the file. Model IDs correspond to `id`s of models defined in the `MODEL_CONFIGS` list in the [`scripts/models.py`](./scripts/models.py) file.

New models may be added by adding new `MLEBEvaluationModelConfig` instances (defined in [`scripts/structs.py`](./scripts/structs.py)) to the `MODEL_CONFIGS` list.

Results are written in the [`mteb`](https://github.com/embeddings-benchmark/mteb) format to the [`results`](./results) directory.

[`scripts/export.py`](./scripts/export.py) may be run to pack all results into a single JSONL file available at [`results/results.jsonl`](./results/results.jsonl). That file is used to dynamically present the latest benchmark results on the [MLEB website](https://isaacus.com/mleb).

## License
This project is licensed under the [MIT License](./LICENSE).