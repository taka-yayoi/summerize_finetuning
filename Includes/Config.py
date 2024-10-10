# Databricks notebook source
# 事前にカタログとスキーマを作成し、以下で設定してください
CATALOG = "takaakiyayoi_catalog"
USER_SCHEMA = "finetune_summarizer"

# モデルサービングエンドポイント名。必要に応じて変更ください。
MODEL_ENDPOINT_NAME = "ft-summarize-endpoint"

# ファインチューニングするモデル
BASE_MODEL =  "meta-llama/Meta-Llama-3-8B-Instruct" # ファインチューニングする基盤モデル

# ファインチューニングしたモデルの名前
UC_MODEL_NAME = "blog_summarization_llm"

# 推論テーブル名のプレフィクス
INFERENCE_TABLE_PREFIX = "ift_request_response"

# COMMAND ----------

print(f"このデモでは {CATALOG}.{USER_SCHEMA} を使用します。データを書き込むために必要なカタログやスキーマを設定するには、Configノートブックで `CATALOG` や `SHARED_SCHEMA` 変数を指定してください。")

# COMMAND ----------

# UC指示データセット
INPUT_TABLE = "blogs_bronze"
OUTPUT_VOLUME = "blog_ift_data"
OUTPUT_TRAIN_TABLE = "blog_title_generation_train_ift_data"
OUTPUT_EVAL_TABLE = "blog_title_generation_eval_ift_data"

# デモで使用するブログデータのjsonlファイル名
RAW_JSONL_NAME = "blog.jsonl"

# COMMAND ----------

print(f"このデモでは、ボリューム {OUTPUT_VOLUME} に配置される {RAW_JSONL_NAME} から記事データを読み込み、テーブル {INPUT_TABLE} を作成します。このテーブルからトレーニングデータ {OUTPUT_TRAIN_TABLE} と評価用データ {OUTPUT_EVAL_TABLE} を作成します。")

# COMMAND ----------

print(f"このデモでは、基盤モデル {BASE_MODEL} をファインチューニングし、モデル {UC_MODEL_NAME} を作成します。ファインチューンしたモデルは、モデルサービングエンドポイント {MODEL_ENDPOINT_NAME} にデプロイされます。")
