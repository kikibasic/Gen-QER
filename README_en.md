# Gen-QER: Generative Query Expansion & Reranking

[English](#english) | [Japanese](#japanese)

---

<a name="english"></a>
## 🇬🇧 English

**Gen-QER** is an experimental implementation for Information Retrieval (IR) tasks, focusing on Query Expansion and Dense Reranking using Large Language Models (LLMs).

This repository provides a pipeline to:
1. Perform sparse retrieval (BM25).
2. Generate pseudo-documents using LLMs (GPT-4o, etc.).
3. Rerank search results using dense retrievers with **Contextualized Pooling** strategies.

---

### 🛠️ Installation

**Requirements:** Python 3.10+, Java 11+ (Required for Pyserini)

```bash
# 1. Create a virtual environment (Conda is recommended)
conda create -n gen-qer python=3.10 openjdk=11 -c conda-forge -y
conda activate gen-qer

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download BM25 indices and benchmark data
chmod +x scripts/download_data.sh
./scripts/download_data.sh
```

---

### 🏃‍♂️ Usage

Set your OpenAI API key if you intend to use GPT models:

```bash
export OPENAI_KEY="your-api-key-here"
```

Run the main pipeline:

```bash
# Example: Run pipeline on DL19 dataset
# - llm: Model for generation (gpt-4o, gpt-3.5-turbo, or local path)
# - doc_gen: Number of pseudo-documents to generate (e.g., 2)
# - mode: Strategy for combining query and documents (use 'contex-pool')
# - rank_model: HuggingFace model ID for the dense retriever

python main.py \
  --irmode mugipipeline \
  --llm gpt-4o \
  --doc_gen 2 \
  --mode contex-pool \
  --rank_model BAAI/bge-large-en-v1.5
```

---

### 📂 Project Structure

- **main.py**: The main entry point for the pipeline. Handles the workflow from retrieval to evaluation.
- **config.py**: Handles command-line arguments and default configurations.
- **src/**: Source code modules.
  - **prompts.py**: Defines prompt templates for LLM generation.
  - **retriever.py**: Implements the dense reranking logic (including Contextualized Pooling).
  - **generator.py**: Wrapper class for OpenAI and HuggingFace models.
  - **searcher.py**: Handles sparse retrieval (Pyserini/BM25) and query expansion loops.
  - **evaluation.py**: Utilities for running trec_eval and calculating metrics.
- **exp/**: Stores intermediate results (JSON files containing search hits and generated text).
- **results/**: Stores final evaluation outputs (TREC run files) and summary logs (mugipipeline.json).

---

### 📚 Reference

This codebase is based on the implementation of MuGI.

```bibtex
@inproceedings{zhang-etal-2024-exploring-best,
    title = "Exploring the Best Practices of Query Expansion with Large Language Models",
    author = "Zhang, Le and Wu, Yihong and Yang, Qian and Nie, Jian-Yun",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    year = "2024"
}
```
