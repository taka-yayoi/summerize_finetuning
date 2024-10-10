# Databricks notebook source
# MAGIC %md
# MAGIC # 5: オフライン評価
# MAGIC
# MAGIC このノートブックでは:
# MAGIC 1. 単一のリクエストでモデル提供エンドポイントをクエリします
# MAGIC 1. 一連のリクエストでエンドポイントをクエリします
# MAGIC     - デモでは、pandas UDFを使用して一連のリクエストを送信しました。このラボでは、リクエスト構造を単純に変更して、モデル提供エンドポイントに対して送信できるプロンプトのリストを受け入れるようにします

# COMMAND ----------

# MAGIC %pip install databricks-sdk==0.31.1 textstat==0.7.3
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 環境をセットアップして、必要な変数とデータセットをロードします。

# COMMAND ----------

# MAGIC %run ../Includes/Config

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import json
import mlflow
from mlflow import MlflowClient
from mlflow.metrics.genai.metric_definitions import answer_similarity
import pandas as pd
from typing import Iterator
import pyspark.sql.functions as F
import requests

# COMMAND ----------

# 評価データ
table_name = f"{CATALOG}.{USER_SCHEMA}.{OUTPUT_EVAL_TABLE}"

API_TOKEN = mlflow.utils.databricks_utils.get_databricks_host_creds().host
WORKSPACE_URL = mlflow.utils.databricks_utils.get_databricks_host_creds().token

# モデルのパラメーター
max_tokens = 128
temperature = 0.9

# COMMAND ----------

# MAGIC %md
# MAGIC ## データフレームを適切な形式に整形する
# MAGIC
# MAGIC `mlflow.evaluate` では、評価用のデータフレームを "inputs" と "ground_truth" のカラムを持つ pandas のデータフレームとして整形する必要があります。

# COMMAND ----------

eval_df = (spark.table(table_name)
           .select("prompt", "response")
           .withColumnRenamed("prompt", "inputs")
           .withColumnRenamed("response", "ground_truth") 
           )
eval_pdf = eval_df.toPandas()
    
print(eval_pdf.count())
display(eval_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## エンドポイントに対してプロンプトのバッチを送信する
# MAGIC
# MAGIC ここでは、Databricks Python SDKやpandas UDFの代わりに、`requests`ライブラリを使用してエンドポイントに対してプロンプトのバッチを送信します。最初に、単一のプロンプトでテストを行います。

# COMMAND ----------

def get_predictions(prompts, max_tokens, temperature, model_serving_endpoint):
    from mlflow.utils.databricks_utils import get_databricks_env_vars
    import requests

    # Databricksの環境変数から認証情報を取得
    mlflow_db_creds = get_databricks_env_vars("databricks")
    API_TOKEN = mlflow_db_creds["DATABRICKS_TOKEN"]
    WORKSPACE_URL = mlflow_db_creds["_DATABRICKS_WORKSPACE_HOST"]

    # モデルに送るデータの準備
    payload = {"prompt": prompts, "max_tokens": max_tokens, "temperature": temperature}
    
    # 認証情報をヘッダーに設定
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {API_TOKEN}"
               }
    
    # モデルサービングエンドポイントにリクエストを送信
    response = requests.post(url=f"{WORKSPACE_URL}/serving-endpoints/{model_serving_endpoint}/invocations",
                             json=payload,
                             headers=headers
                             )
    # レスポンスから予測結果を取得
    predictions = response.json().get("choices")
    return predictions

# COMMAND ----------

def make_prediction_udf(model_serving_endpoint):
    @F.pandas_udf("string")
    def get_prediction_udf(batch_prompt: Iterator[pd.Series]) -> Iterator[pd.Series]:

        import mlflow

        max_tokens = 100  # 最大トークン数を設定
        temperature = 1.0  # 温度パラメータを設定
        api_root = mlflow.utils.databricks_utils.get_databricks_host_creds().host  # APIのルートURLを取得
        api_token = mlflow.utils.databricks_utils.get_databricks_host_creds().token  # APIトークンを取得

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_token}"  # 認証ヘッダーを設定
        }
        
        for batch in batch_prompt:
            
            result = []  # 結果を格納するリスト
            for prompt, max_tokens, temperature in batch[["prompt", "max_tokens", "temperature"]].itertuples(index=False):  
                data = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}  # リクエストデータを準備
                response = requests.post(
                    url=f"{api_root}/serving-endpoints/{model_serving_endpoint}/invocations",
                    json=data,
                    headers=headers  # リクエストを送信
                )
                if response.status_code == 200:  # レスポンスが成功した場合
                    endpoint_output = json.dumps(response.json())  # レスポンスデータをJSON文字列に変換
                    data = json.loads(endpoint_output)  # JSON文字列を辞書に変換
                    prediction = data.get("choices")  # 予測結果を取得
                    try:
                        predicted_docs = prediction[0]["text"]  # 予測テキストを取得
                        result.append(predicted_docs)  # 結果リストに追加
                    except IndexError as e:  # 予測テキストが存在しない場合
                        result.append("null")  # nullを追加
                else:  # レスポンスが失敗した場合
                    result.append(str(response.raise_for_status()))  # エラーメッセージを追加

        yield pd.Series(result)  # 結果のシリーズを生成
    return get_prediction_udf  # UDFを返す

get_prediction_udf = make_prediction_udf(MODEL_ENDPOINT_NAME)  # UDFを作成

# COMMAND ----------

predictions = get_predictions(prompts=eval_pdf["inputs"][0], 
                max_tokens=max_tokens, 
                temperature=temperature,
                model_serving_endpoint=MODEL_ENDPOINT_NAME)

print(predictions[0]["text"])

# COMMAND ----------

# MAGIC %md
# MAGIC 上記のコードでは、単一のプロンプトでエンドポイントをクエリできることがわかります。今度は、`eval_df`のバッチのプロンプトを受け入れるために、プロンプトの構造を変更する番です。`eval_df`のレコード数が多くなるほど処理に時間を要することになります。

# COMMAND ----------

predictions_df = eval_df.withColumn(
    "generated_title",
    get_prediction_udf(
        F.struct(
            F.col("inputs").alias("prompt"),  # 入力プロンプト
            F.lit(max_tokens).alias("max_tokens"),  # 最大トークン数
            F.lit(temperature).alias("temperature"),  # 温度パラメータ
        )
    ),
)

# COMMAND ----------

# 評価データフレームを保存する
predictions_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{USER_SCHEMA}.eval_blog_df")
display(spark.table(f"{CATALOG}.{USER_SCHEMA}.eval_blog_df"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLMで生成されたタイトルを評価する
# MAGIC
# MAGIC LLMで生成されたタイトルの品質を評価しましょう！

# COMMAND ----------

eval_pdf = spark.table(f"{CATALOG}.{USER_SCHEMA}.eval_blog_df").toPandas()

# COMMAND ----------

w = WorkspaceClient()
model_name = w.serving_endpoints.get(name=MODEL_ENDPOINT_NAME).config.served_entities[0].entity_name
model_version = 1
mlflow_client = MlflowClient(registry_uri="databricks-uc")

# 登録済みモデルのモデルバージョンオブジェクトを取得する
# 注意：UCの適切な権限がない場合は失敗します
mv = mlflow_client.get_model_version(name=model_name, version=model_version)
training_run_id = mv.run_id

# COMMAND ----------

with mlflow.start_run(run_id=training_run_id) as run: 
    # MLflowを使用してモデルの評価を実行
    results = mlflow.evaluate(data=eval_pdf, 
                              targets="ground_truth",
                              predictions="generated_title",
                              model_type="text",
                             )
    
    # 評価結果のメトリックをJSON形式で出力
    print(json.dumps(results.metrics, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLMをジャッジとして使用する
# MAGIC
# MAGIC 上記のメトリクスに加えて、LLMをジャッジとして使用してさらにメトリクスを生成しましょう。既にデフォルトのメトリクスを生成しているため、`model_type`引数を削除して、LLM判定のメトリクスのみを生成します。

# COMMAND ----------

llm_judge = "endpoints:/databricks-dbrx-instruct"
# モデルの回答類似度メトリックを定義
answer_similarity_metric = answer_similarity(model=llm_judge)

with mlflow.start_run(run_id=training_run_id) as run: 
    # MLflowを使用してモデルの評価を実行し、追加メトリックを含める
    results = mlflow.evaluate(data=eval_pdf, 
                              targets="ground_truth",
                              predictions="generated_title",
                              extra_metrics=[answer_similarity_metric]
                             )
    # 評価結果のメトリックをJSON形式で出力
    print(json.dumps(results.metrics, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC エクスペリメント画面の**Evaluation**タブで正解データと生成された結果や評価指標を比較することができます。
# MAGIC
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/ft_evaluation.png)

# COMMAND ----------

displayHTML(f"<a href='/ml/experiments/{run.info.experiment_id}'>エクスペリメント</a>")

# COMMAND ----------

# MAGIC %md
# MAGIC お疲れ様でした！色々なデータやプロンプトでトライしてみてください！
# MAGIC
# MAGIC **モデルサービングエンドポイントを使わない場合には停止しておきましょう。**
# MAGIC
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/ft_stop_endpoint.png)
# MAGIC
# MAGIC **関連資料**
# MAGIC
# MAGIC - [Mosaic AI Model TrainingによるLLMのファインチューニング](https://qiita.com/taka_yayoi/items/91f35205855127463bbf)
# MAGIC - [mlflow\.evaluateを用いた大規模言語モデルの評価](https://qiita.com/taka_yayoi/items/4821215bfc133043416d)
