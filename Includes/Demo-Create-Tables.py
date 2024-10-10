# Databricks notebook source
import os

# bronzeテーブル作成に使用するjsonlファイルのパス
source_file_path = f"{os.getcwd()}/{RAW_JSONL_NAME}"

# COMMAND ----------

import shutil

print("スキーマの初期化...")
spark.sql(f"DROP SCHEMA IF EXISTS {CATALOG}.{USER_SCHEMA} CASCADE")
spark.sql(f"CREATE SCHEMA {CATALOG}.{USER_SCHEMA}")
print(f"スキーマ {CATALOG}.{USER_SCHEMA} を作成しました。")
print()

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{USER_SCHEMA}.{OUTPUT_VOLUME}")
print(f"ボリューム {OUTPUT_VOLUME} を作成しました")
print()

# ワークスペースからボリュームにファイルをコピー
ret = shutil.copy(source_file_path, f"/Volumes/{CATALOG}/{USER_SCHEMA}/{OUTPUT_VOLUME}")
volume_path = f"/Volumes/{CATALOG}/{USER_SCHEMA}/{OUTPUT_VOLUME}/{RAW_JSONL_NAME}"

table_name = f"{CATALOG}.{USER_SCHEMA}.{INPUT_TABLE}"

df = spark.read.format("json").load(volume_path)
spark.sql(f"DROP TABLE IF EXISTS {table_name}")
df.write.saveAsTable(table_name)

print(f"テーブル {table_name} を作成しました")



print()
print("データの初期化を完了しました。")
