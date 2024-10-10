# Databricks notebook source
# MAGIC %md
# MAGIC # 3: プロビジョニングされたスループット サービング エンドポイントの作成
# MAGIC
# MAGIC このノートブックでは、[プロビジョニングされたスループット ファウンデーション モデル API](https://docs.databricks.com/ja/machine-learning/foundation-models/deploy-prov-throughput-foundation-model-apis.html#provisioned-throughput-foundation-model-apis) を作成します。プロビジョニングされたスループットは、本番ワークロードのパフォーマンス保証を備えたファウンデーション モデルの最適化された推論を提供します。
# MAGIC
# MAGIC サポートされているモデルアーキテクチャのリストについては、[プロビジョニングされたスループット ファウンデーション モデル API](https://docs.databricks.com/ja/machine-learning/foundation-models/index.html#provisioned-throughput-foundation-model-apis) を参照してください。
# MAGIC
# MAGIC このノートブックでは以下を行います:
# MAGIC
# MAGIC 1. デプロイするモデルを定義します。これは、Unity Catalogに登録されたファインチューニングされたモデルになります
# MAGIC 1. 登録されたモデルの最適化情報を取得します
# MAGIC 1. エンドポイントを設定して作成します
# MAGIC 1. エンドポイントをクエリします

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.31.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 環境を設定して、必要な変数とデータセットを読み込みます。

# COMMAND ----------

# MAGIC %run ../Includes/Config

# COMMAND ----------

import json
import requests

import mlflow
import mlflow.deployments
from mlflow.deployments import get_deploy_client
from mlflow.exceptions import MlflowException
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    ServedEntityInput,
    EndpointCoreConfigInput,
    AutoCaptureConfigInput,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 設定
# MAGIC
# MAGIC Unity Catalogの設定変数をセットアップします。登録されたモデル名とデプロイするモデルバージョンを定義します。また、エンドポイントの名前と、[推論テーブル](https://docs.databricks.com/ja/machine-learning/model-serving/inference-tables.html)の名前も定義します。この推論テーブルは、エンドポイントのリクエストとレスポンスのペイロードを記録するために作成されます。

# COMMAND ----------

# MAGIC %md
# MAGIC # UCモデルのバージョン

# COMMAND ----------

# 前のノートブックでUCに登録したモデルのバージョンを指定します
MODEL_VERSION = 1

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルの最適化情報を取得する
# MAGIC
# MAGIC モデル名とモデルバージョンを指定することで、モデルの最適化情報を取得することができます。これは、特定のモデルに対して1つのスループットユニットに対応するトークン/秒の数です。

# COMMAND ----------

API_ROOT = mlflow.utils.databricks_utils.get_databricks_host_creds().host
API_TOKEN = mlflow.utils.databricks_utils.get_databricks_host_creds().token

def get_model_optimization_info(full_model_name: str, model_version: int):
    """指定された登録済みモデルとバージョンのモデル最適化情報を取得します。"""
    headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}
    url = f"{API_ROOT}/api/2.0/serving-endpoints/get-model-optimization-info/{full_model_name}/{model_version}"
    response = requests.get(url=url, headers=headers)
    return response.json()


# 登録済みモデルを指定して最適化情報を取得する
model_optimization_info = get_model_optimization_info(
    full_model_name=f"{CATALOG}.{USER_SCHEMA}.{UC_MODEL_NAME}", model_version=MODEL_VERSION
)
print("model_optimization_info: ", model_optimization_info)
min_provisioned_throughput = model_optimization_info["throughput_chunk_size"]
# コスト削減のために最小値と同じ値を最大値に設定します。予想されるリクエストの負荷に基づいてより高い数値を選択することもできます。
max_provisioned_throughput = model_optimization_info["throughput_chunk_size"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## GPUモデルサービングエンドポイントの設定と作成
# MAGIC
# MAGIC エンドポイントの作成APIを呼び出した後、ログされたモデルは自動的に最適化されたLLMサービングで展開されます。

# COMMAND ----------

w = WorkspaceClient()

_ = spark.sql(f"DROP TABLE IF EXISTS {CATALOG}.{USER_SCHEMA}.{INFERENCE_TABLE_PREFIX}_payload")

print("エンドポイントの作成中..")
w.serving_endpoints.create_and_wait(
    name=MODEL_ENDPOINT_NAME,
    config=EndpointCoreConfigInput(
        name=MODEL_ENDPOINT_NAME,
        served_entities=[
            ServedEntityInput(
                entity_name=f"{CATALOG}.{USER_SCHEMA}.{UC_MODEL_NAME}", # サービングエンドポイントにデプロイされるUC登録モデル
                entity_version=MODEL_VERSION, # モデルのバージョン
                max_provisioned_throughput=max_provisioned_throughput, # スループットの最大値
                min_provisioned_throughput=0, # スループットの最小値
                scale_to_zero_enabled=True # ゼロノードへのスケーリングを有効にする
            )
        ],
        # エンドポイントの入出力をキャプチャする推論テーブル
        auto_capture_config=AutoCaptureConfigInput(
            catalog_name=CATALOG,
            schema_name=USER_SCHEMA,
            enabled=True,
            table_name_prefix=INFERENCE_TABLE_PREFIX,
        ),
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC 上を実行した際、`TimeoutError: timed out after 0:20:00: current status: EndpointStateConfigUpdate.IN_PROGRESS`というエラーが出たとしても、以下のセルを実行して表示されるリンク先で処理が進んでいれば問題ありません。

# COMMAND ----------

displayHTML(f"<a href='/ml/endpoints/{MODEL_ENDPOINT_NAME}' target='_blank'>エンドポイント{MODEL_ENDPOINT_NAME}の詳細</a>")

# COMMAND ----------

# MAGIC %md
# MAGIC ## エンドポイントを表示
# MAGIC エンドポイントに関する詳細情報を表示するには、左側のナビゲーションバーで**Serving**を選択し、エンドポイント名を検索してください。
# MAGIC
# MAGIC モデルのサイズや複雑さによっては、エンドポイントが準備完了するまでに30分以上かかることがあります。
# MAGIC
# MAGIC エンドポイントが稼働していることを確認して、[次のノートブック]($./04_Query_Endpoint_and_Batch_Inference)に進みましょう。
# MAGIC
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/ft_serving_endpoint.png)
