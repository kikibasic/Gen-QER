# Gen-QER: 生成的クエリ拡張とリランキング

[English](#english) | [日本語](#japanese)

---

<a name="japanese"></a>
## 🇯🇵 日本語

**Gen-QER**は、大規模言語モデル(LLM)を用いたクエリ拡張と密ベクトルリランキングに焦点を当てた、情報検索(IR)タスクの実験的実装です。

このリポジトリは以下のパイプラインを提供します:
1. スパース検索(BM25)の実行
2. LLM(GPT-4o等)を用いた疑似文書の生成
3. **コンテキスト化プーリング**戦略を用いた密ベクトル検索器によるリランキング

---

### 🛠️ インストール

**必要環境:** Python 3.10以上、Java 11以上 (Pyseriniに必要)

```bash
# 1. 仮想環境の作成 (Condaを推奨)
conda create -n gen-qer python=3.10 openjdk=11 -c conda-forge -y
conda activate gen-qer

# 2. Python依存パッケージのインストール
pip install -r requirements.txt

# 3. BM25インデックスとベンチマークデータのダウンロード
chmod +x scripts/download_data.sh
./scripts/download_data.sh
```

---

### 🏃‍♂️ 使用方法

GPTモデルを使用する場合は、OpenAI APIキーを設定してください:

```bash
export OPENAI_KEY="your-api-key-here"
```

メインパイプラインの実行:

```bash
# 例: DL19データセットでパイプラインを実行
# - llm: 生成に使用するモデル (gpt-4o, gpt-3.5-turbo, またはローカルパス)
# - doc_gen: 生成する疑似文書の数 (例: 2)
# - mode: クエリと文書を結合する戦略 ('contex-pool'を使用)
# - rank_model: 密ベクトル検索器のHuggingFaceモデルID

python main.py \
  --irmode mugipipeline \
  --llm gpt-4o \
  --doc_gen 2 \
  --mode contex-pool \
  --rank_model BAAI/bge-large-en-v1.5
```

---

### 📂 プロジェクト構造

- **main.py**: パイプラインのメインエントリーポイント。検索から評価までのワークフローを処理
- **config.py**: コマンドライン引数とデフォルト設定を処理
- **src/**: ソースコードモジュール
  - **prompts.py**: LLM生成用のプロンプトテンプレートを定義
  - **retriever.py**: 密ベクトルリランキングロジックを実装 (コンテキスト化プーリングを含む)
  - **generator.py**: OpenAIとHuggingFaceモデルのラッパークラス
  - **searcher.py**: スパース検索 (Pyserini/BM25) とクエリ拡張ループを処理
  - **evaluation.py**: trec_evalの実行とメトリクス計算のユーティリティ
- **exp/**: 中間結果を保存 (検索結果と生成テキストを含むJSONファイル)
- **results/**: 最終評価結果 (TREC実行ファイル) とサマリーログ (mugipipeline.json) を保存

---

### 📚 参考文献

このコードベースはMuGIの実装に基づいています。

```bibtex
@inproceedings{zhang-etal-2024-exploring-best,
    title = "Exploring the Best Practices of Query Expansion with Large Language Models",
    author = "Zhang, Le and Wu, Yihong and Yang, Qian and Nie, Jian-Yun",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    year = "2024"
}
```

---

<a name="english"></a>
## 🇬🇧 English

**Gen-QER** is an experimental implementation for Information Retrieval (IR) tasks, focusing on Query Expansion and Dense Reranking using Large Language Models (LLMs).

This repository provides a pipeline to:
1. Perform sparse retrieval (BM25).
2. Generate pseudo-documents using LLMs (GPT-4o, etc.).
3. Rerank search results using dense retrievers with **Contextualized Pooling** strategies.

For detailed documentation in English, please refer to [README_en.md](README_en.md).