# Databricks notebook source
# MAGIC %md
# MAGIC # 2: インストラクションのファインチューニング
# MAGIC
# MAGIC このノートブックでは、事前訓練された言語モデルに対してインストラクションのファインチューニング（IFT）を実行する方法を示します。
# MAGIC
# MAGIC 目標:
# MAGIC
# MAGIC 1. 指定されたハイパーパラメータで単一のIFT実行をトリガーする

# COMMAND ----------

# MAGIC %pip install databricks-genai==1.0.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 環境をセットアップして、必要な変数とデータセットをロードします。

# COMMAND ----------

# MAGIC %run ../Includes/Config

# COMMAND ----------

# MAGIC %md
# MAGIC ## ファインチューニングのランの作成

# COMMAND ----------

from databricks.model_training import foundation_model as fm

register_to = f"{CATALOG}.{USER_SCHEMA}.{UC_MODEL_NAME}" # ファインチューニングしたモデルの登録先
training_duration = "3ep" # エポック数
learning_rate = "3e-06" # 学習率
data_prep_cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId") # データ準備に使用するクラスターID

run = fm.create(
  model=BASE_MODEL,
  train_data_path=f"{CATALOG}.{USER_SCHEMA}.{OUTPUT_TRAIN_TABLE}",
  eval_data_path=f"{CATALOG}.{USER_SCHEMA}.{OUTPUT_EVAL_TABLE}",
  data_prep_cluster_id=data_prep_cluster_id,
  register_to=register_to,
  training_duration=training_duration,
  learning_rate=learning_rate,
)
run

# COMMAND ----------

# 進捗の確認
run.get_events()

# COMMAND ----------

# MAGIC %md
# MAGIC ファインチューニングされたモデルは、以下を実行して表示される[MLflowエクスペリメント](https://docs.databricks.com/ja/mlflow/experiments.html)で管理されます。エクスペリメントでトレーニングの進捗を確認することもできます。
# MAGIC
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/ft_metrics.png)

# COMMAND ----------

displayHTML(f"<a href='/ml/experiments/{run.experiment_id}'>エクスペリメント</a>")

# COMMAND ----------

# MAGIC %md
# MAGIC **Status**が**Completed**になったことを確認して[次のノートブック]($./03_Create_a_Provisioned_Throughput_Serving_Endpoint)に進んでください。
# MAGIC
# MAGIC ![](https://sajpstorage.blob.core.windows.net/yayoi/ft_complete.png)
