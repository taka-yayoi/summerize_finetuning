# Databricks notebook source
# MAGIC %md
# MAGIC # 1: 指示データセットの作成

# COMMAND ----------

# MAGIC %md
# MAGIC このノートブックでは、指示のファインチューニングに使用する指示データセットを作成します。解決しようとするユースケースは、過去のDatabricksブログ投稿のスタイルでブログ投稿のタイトルを生成することです。このため、過去のブログ投稿とそのタイトルを指示データセットに準備します。
# MAGIC
# MAGIC **手順:**
# MAGIC 1. 生のDatabricks記事データを読み込む
# MAGIC 2. [Unity Catalog Volume](https://docs.databricks.com/ja/connect/unity-catalog/volumes.html)にデータを書き込み、そこからテーブルを作成する。
# MAGIC 3. 空の行をフィルタリングし、データの重複を排除する
# MAGIC 4. ブログテキストをプロンプトに構造化する
# MAGIC 5. `prompt`、`response`の列を持つテーブルを作成する。ここで、`response`列はブログ投稿のタイトルです

# COMMAND ----------

# サンプルデータの準備
"""
blog_df = spark.sql("SELECT title, body AS text FROM takaakiyayoi_catalog.qiita_2023.taka_qiita_2023 LIMIT 100;")
blog_dict = blog_df.toPandas().to_dict(orient="records")
print(blog_dict)

import json

with open('blog.jsonl', mode='w', encoding='utf-8') as fout:
    for obj in blog_dict:
        json.dump(obj, fout, ensure_ascii=False)
        fout.write('\n')
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ### インポート

# COMMAND ----------

# MAGIC %md 環境を設定して、必要な変数とデータセットを読み込みます。

# COMMAND ----------

# MAGIC %run ../Includes/Config

# COMMAND ----------

# MAGIC %run ../Includes/Demo-Create-Tables

# COMMAND ----------

from typing import Iterator, List
import pandas as pd
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import StringType

# COMMAND ----------

# MAGIC %md
# MAGIC ### データの読み込みとフィルタリング
# MAGIC
# MAGIC 指定されたUnity Catalogテーブルから生のブログデータを読み込み、'text'または'title'列にnullまたは空の値がある行をフィルタリングします。

# COMMAND ----------

def load_and_filter(table_name: str, response_col: str = "title") -> DataFrame:
    """
    テーブルをロードし、'text'または`response_col`でnullまたは空の文字列をフィルタリングします。

    引数:
        table_name: ロードするテーブルの名前。
        response_col: nullまたは空の文字列をフィルタリングする列。

    戻り値:
        フィルタリングされたDataFrame。
    """
    print(f"テーブルをロード中: {table_name}")
    df = spark.table(table_name)
    original_count = df.count()
    print(f"行数: {original_count}")

    print(f"\n'text'または'{response_col}'でnullまたは空の文字列をフィルタリング")
    filtered_df = filter_null_or_empty(df, ["text", response_col])
    filtered_count = filtered_df.count()
    print(f"削除された行数: {original_count - filtered_count}")
    print(f"フィルタリングされた数: {filtered_count}")

    return filtered_df
  

def filter_null_or_empty(df: DataFrame, columns: List[str]) -> DataFrame:
    """
    指定された列のいずれかがnullまたは空である行をフィルタリングします。

    引数:
        df: フィルタリングするDataFrame。
        columns: nullまたは空の値をチェックする列のリスト。

    戻り値:
        フィルタリングされたDataFrame。
    """
    print("指定された列のいずれかがnullまたは空である行をフィルタリング中...")
    for col in columns:
        print(f"\t列: {col}")
        df = df.filter((F.col(col).isNotNull()) & (F.col(col) != ""))
    return df

# COMMAND ----------

filtered_df = load_and_filter(table_name=f"{CATALOG}.{USER_SCHEMA}.{INPUT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 重複排除
# MAGIC
# MAGIC `text`および`title`列に基づいてフィルタリングされたデータセットの重複を排除し、ユニークなブログ投稿を保証します。

# COMMAND ----------

filtered_deduped_df = filtered_df.drop_duplicates(subset=["text", "title"])
filtered_deduped_count = filtered_deduped_df.count()
print(f"重複排除後の件数: {filtered_deduped_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## プロンプト列の追加
# MAGIC
# MAGIC
# MAGIC `PromptTemplate` クラスが以下を返すようにします:
# MAGIC - 指示
# MAGIC - ブログキー
# MAGIC - ブログテキスト
# MAGIC - 応答

# COMMAND ----------

class PromptTemplate:
    """クラスを使用して、インストラクションデータセットの生成のためのプロンプトテンプレートを表すクラス。"""

    def __init__(self, instruction: str, blog_key: str, response_key: str) -> None:
        self.instruction = instruction
        self.blog_key = blog_key
        self.response_key = response_key

    def generate_prompt(self, blog_text: str) -> str:
        """
        テンプレートと与えられたブログテキストを使用してプロンプトを生成します。

        Args:
            blog_text: ブログのテキスト。

        Returns:
            プロンプトテンプレート。
        """
        return f"""{self.instruction}
{self.blog_key}
{blog_text}
{self.response_key}
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## プロンプトテンプレートの作成
# MAGIC
# MAGIC - `instruction` には、提供されたブログ記事に基づいてタイトルを生成するためのLLMへのポインタを含める必要があります

# COMMAND ----------

blog_title_generation_template = PromptTemplate(
    instruction="以下はDatabricksに関するブログ記事のテキストです。指定されたブログ記事の要約を作成してください。",
    blog_key="### ブログ記事:",
    response_key="### 要約:"
)  

# COMMAND ----------

def add_instruction_prompt_column(df: DataFrame, prompt_template: PromptTemplate) -> DataFrame:
    """
    DataFrameに指定されたテンプレートを使用して 'prompt' 列を追加します。

    Args:
        df: 入力のDataFrame。
        prompt_template: プロンプトを生成するために使用するテンプレート。

    Returns:
        'prompt' 列を持つDataFrame。
    """
    @F.pandas_udf(StringType())
    def generate_prompt(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
        for texts in batch_iter:
            prompts = texts.apply(prompt_template.generate_prompt)
            yield prompts

    return df.withColumn("prompt", generate_prompt(df["text"]))

# COMMAND ----------

# プロンプト列を追加
instruction_df = add_instruction_prompt_column(filtered_deduped_df, blog_title_generation_template)

# プロンプト列とタイトル列を選択し、タイトル列をレスポンス列に名前変更
instruction_df = instruction_df.selectExpr("prompt", "title as response")
display(instruction_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### プロンプトの例

# COMMAND ----------

print(instruction_df.select("prompt").limit(1).collect()[0]["prompt"])

# COMMAND ----------

# MAGIC %md
# MAGIC データをランダムにトレーニングデータとテストデータに分割します

# COMMAND ----------

train_df, eval_df = instruction_df.randomSplit([0.9,0.1], seed=42)
print(train_df.count(), eval_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## トレーニングデータと評価データを別々のテーブルに書き込む

# COMMAND ----------

train_data_path = f"{CATALOG}.{USER_SCHEMA}.{OUTPUT_TRAIN_TABLE}"
eval_data_path = f"{CATALOG}.{USER_SCHEMA}.{OUTPUT_EVAL_TABLE}"

train_df.write.mode("overwrite").saveAsTable(train_data_path)
eval_df.write.mode("overwrite").saveAsTable(eval_data_path)

# COMMAND ----------

# MAGIC %md
# MAGIC 以下のコマンドを実行して表示される[カタログエクスプローラ](https://docs.databricks.com/ja/catalog-explorer/index.html)でテーブルを確認してみてください。**依存関係**タブでテーブル間のリネージを確認することもできます。

# COMMAND ----------

displayHTML(f"<a href='/explore/data/{CATALOG}/{USER_SCHEMA}/'>カタログエクスプローラ</a>")

# COMMAND ----------

# MAGIC %md
# MAGIC [次のノートブック]($./02_Instruction_Fine_Tuning)に進みましょう。
