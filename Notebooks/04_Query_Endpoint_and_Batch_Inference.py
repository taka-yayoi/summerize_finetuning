# Databricks notebook source
# MAGIC %md
# MAGIC # 4: エンドポイントのクエリーとバッチ推論
# MAGIC
# MAGIC このノートブックでは、以下の操作を行います:
# MAGIC 1. 単一のリクエストでモデルサービングエンドポイントをクエリします

# COMMAND ----------

# MAGIC %md 環境をセットアップして、必要な変数とデータセットをロードします。

# COMMAND ----------

# MAGIC %run ../Includes/Config

# COMMAND ----------

from bs4 import BeautifulSoup
import json
import requests

import mlflow.deployments
from mlflow.deployments import get_deploy_client
from mlflow.exceptions import MlflowException

# COMMAND ----------

# MAGIC %md
# MAGIC ## ブログ記事のURLを提供してください
# MAGIC まず、ブログのURLを指定して、単一のブログ記事のテキストを取得しましょう。https://qiita.com/taka_yayoi にアクセスし、いずれかのブログ投稿のURLを選択してください。

# COMMAND ----------

def get_single_blog_post(url: str) -> str:
    """
    ブログのURLを指定して、単一のブログ記事のテキストを取得します。

    Args:
        url (str): ブログ記事のURL。

    Returns:
        str: クリーンされたブログ記事のテキスト。
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # ブログ記事のテキストコンテナを見つける(サイトによって構成が異なります)
    blog_text_container = soup.find("div", class_="p-items_main")
    
    if blog_text_container:
        # HTMLタグを削除してテキストを抽出する
        blog_text = " ".join(blog_text_container.stripped_strings)
        
        # テキストをクリーンアップする
        blog_text = blog_text.replace("\\'", "'")
        blog_text = blog_text.replace(" ,", ",")
        blog_text = blog_text.replace(" .", ".")
        
        return blog_text
    else:
        print(f"URL {url} のブログ記事が見つかりませんでした。")
        return ""

url = "https://qiita.com/taka_yayoi/items/9ee9ba97ebdb82704244"
blog_post_text = get_single_blog_post(url)

blog_post_text

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1つ目のノートブックで取り組んだテンプレートを用いてプロンプトを構成

# COMMAND ----------

class PromptTemplate:
    """Promptテンプレートを表すクラス。データセット生成用。"""

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

blog_title_generation_template = PromptTemplate(
    instruction="以下はDatabricksのブログ記事のテキストです。提供されたブログ記事にタイトルを作成してください。",
    blog_key="### ブログ:",
    response_key="### タイトル:"
)

prompt = blog_title_generation_template.generate_prompt(blog_post_text)
print(prompt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## エンドポイントへのクエリー

# COMMAND ----------

from mlflow.utils.databricks_utils import get_databricks_env_vars

mlflow_db_creds = get_databricks_env_vars("databricks")
API_TOKEN = mlflow_db_creds["DATABRICKS_TOKEN"]
WORKSPACE_URL = mlflow_db_creds["_DATABRICKS_WORKSPACE_HOST"]

# モデルのパラメーター
max_tokens = 128
temperature = 0.9

payload = {
    "prompt": [prompt], "max_tokens": max_tokens, "temperature": temperature
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_TOKEN}"
}

response = requests.post(
    url=f"{WORKSPACE_URL}/serving-endpoints/{MODEL_ENDPOINT_NAME}/invocations",
    json=payload,
    headers=headers
)

predictions = response.json().get("choices")
print(predictions[0]["text"])

# COMMAND ----------

# MAGIC %md
# MAGIC モデルからレスポンスが得られたことを確認して、[最後のノートブック]($./05_Offline_Evaluation)に進みましょう。
