# PerspectiveBench
This project is a sophisticated pipeline designed to transform raw text from various subjects (e.g., Biology, History, Physics) into a structured knowledge graph. From this graph, it generates multi-hop learning paths, synthesizes information, and creates complex questions, effectively creating a dynamic learning and evaluation framework.

## 🚀 Features

- **Fact Extraction**: Ingests raw text and extracts distinct facts and entities using LLMs.
- **Topical Clustering**: Groups related facts into coherent topics using `BERTopic` and semantic embeddings.
- **Cluster Summarization**: Generates concise summaries for each fact cluster to represent core concepts.
- **Relation Inference**: Identifies and establishes relationships between different topics, forming a knowledge graph.
- **Path Generation**: Creates meaningful 1-hop, 2-hop, and 3-hop paths through the knowledge graph to represent learning trajectories.
- **Question Generation**: Synthesizes information along these paths to generate complex, multi-hop questions.
- **Evidence Gathering**: Focuses on relevant evidence from the source documents to support the generated questions and paths.

## ⚙️ Project Structure
```
PerspectiveBench/
├── config/                  # Configuration files (env, yaml)
├── dataset/                 # Raw and processed data
├── llm_clients/             # Clients for interacting with LLMs (OpenAI)
├── utils/                   # Utility scripts (I/O, logging, etc.)
├── step_1_extract_fact.py
├── step_2_1_clustering.py
├── step_2_2_summary.py
├── step_3_1_filtering.py
├── step_3_2_relation.py
├── step_4_1_path_generation.py
├── step_4_2_question_generation.py
├── step_4_3_focus_evidence.py
├── pyproject.toml           # Project dependencies
├── README.md
└── env.sample               # Environment variable template
```

##  workflow

The pipeline is organized into sequential steps, with each script performing a distinct phase of the data processing and generation workflow.

1.  **`step_1_extract_fact.py`**: Reads raw text chunks from the `dataset/raw/` directory and uses an LLM to extract individual facts and associated entities.
2.  **`step_2_1_clustering.py`**: Takes the extracted facts, generates sentence embeddings, and performs clustering using `BERTopic` to group them into topics.
3.  **`step_2_2_summary.py`**: Summarizes the facts within each cluster to create a high-level representation of the topic.
4.  **`step_3_1_filtering.py`**: Analyzes the clusters to identify candidate pairs that are likely to be related, based on shared entities, document proximity, and topic model probabilities.
5.  **`step_3_2_relation.py`**: Uses an LLM to analyze the candidate pairs and generate hypotheses about the specific relationship between them (e.g., "Cause," "Effect," "Sub-concept").
6.  **`step_4_1_path_generation.py`**: Builds a directed graph from the topics and their inferred relations. It then traverses this graph to generate multi-hop paths.
7.  **`step_4_2_question_generation.py`**: Takes the multi-hop paths and the associated topic summaries to generate complex, controllable questions that require synthesizing information across multiple topics.
8.  **`step_4_3_focus_evidence.py`**: Gathers the specific factual sentences that serve as evidence for answering the generated questions.

## Setup

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (for environment and package management)

### Installation
1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd www
    ```

2.  **Create a virtual environment and install dependencies:**
    This project uses `uv` to manage dependencies as specified in `pyproject.toml`.
    ```bash
    uv sync
    ```

3.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```

4.  **Set up environment variables:**
    Create a `.env` file by copying the sample file and fill in the required values.
    ```bash
    cp env.sample .env
    ```
    You will need to add your API keys for OpenAI and/or OpenRouter.

    **.env**
    ```
    LOG_LEVEL=INFO
    PROJECT_ROOT=/path/to/your/project/www
    OPENAI_API_KEY=your_openai_api_key
    OPENROUTER_API_KEY=your_openrouter_api_key
    ```

## Usage

To run the full pipeline, execute the Python scripts in sequential order. Ensure your `.env` file is correctly configured and the virtual environment is active.

```bash
python step_1_extract_fact.py
python step_2_1_clustering.py
python step_2_2_summary.py
python step_3_1_filtering.py
python step_3_2_relation.py
python step_4_1_path_generation.py
python step_4_2_question_generation.py
python step_4_3_focus_evidence.py
```

Each script will generate output in the `dataset/` directory, which will be used as input for the subsequent script.

## Configuration

- **`config/local.yaml`**: Main configuration file for specifying input/output paths, model parameters, and other settings for each step.
- **`config/prompts.yaml`**: Contains the system prompts used for interacting with the LLMs at various stages.
- **`.env`**: Stores secret keys and environment-specific paths.
