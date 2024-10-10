# Databricksにおける要約生成モデルのファインチューニング

## 要件
- [Mosaic AI Model Training API](https://docs.databricks.com/ja/large-language-models/foundation-model-training/index.html)を使用するため、サポートしているリージョンである必要があります。
- ワークスペースでUnity Catalogが有効化されている必要があります。
- `15.4 LTS ML`シングルノードクラスターで動作確認しています。GPUクラスターを使用する必要はありません。

## ディレクトリ構成

以下の`Notebooks`を番号順に実行していきます。

- Includes
  - Config: カタログ、データベース名などを設定します。実行する環境に応じて変更してください。
  - Demo-Create-Tables: デモ用のデータを格納するテーブルを作成します。  
- Notebooks
  - 01_Create_Instruction_Dataset: 1: 指示データセットの作成
  - 02_Instruction_Fine_Tuning: 2: インストラクションのファインチューニング
  - 03_Create_a_Provisioned_Throughput_Serving_Endpoint: 3: プロビジョニングされたスループット サービング エンドポイントの作成
  - 04_Query_Endpoint_and_Batch_Inference: 4: エンドポイントのクエリーとバッチ推論
  - 05_Offline_Evaluation: 5: オフライン評価
  - blog.jsonl: デモデータを格納するjsonl
