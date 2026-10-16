# LLNA EC3 皮膚感作性予測アプリ

XGBoost の各Foldモデルを保存し、単一物質・CSV一括予測、k近傍 Applicability Domain (AD)、性能評価、Fold別 SHAP を提供する Streamlit アプリです。NAMs（動物実験代替手法）の活用を目指す研究用ツールです。

**本番のNotebook条件（1000 trials × 5 Fold）で学習し、原本クラスを独立に同条件で実行した112項目との完全一致を確認しました。** 通常の起動先にはこの本番モデルを設定しています。軽量モデルは別ファイルに保持し、画面にも実際の学習条件を表示します。原本Notebookの最終学習・予測セルには保存済み結果がないため、当時の未保存モデルとの一致とは区別します。外部72件のEC3 R²は **-0.711566** であり、計算の再現は確認できても実用上十分な予測精度を示す結果ではありません。[Notebook監査](docs/notebook_audit.md) と [検証記録](docs/validation.md) を参照してください。

## 環境構築と起動

Python **3.12** を使用してください。検証した依存のバージョンを `requirements.txt` に固定しています。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

GitHub Codespaces では、このリポジトリから Codespace を作成すると `.devcontainer/devcontainer.json` の設定で依存を導入し、画面を起動します。Ports の 8501 からアクセスしてください。CORS / XSRF 保護を無効化する必要はありません。

クラウド作業環境では既に `/workspace/.venvs/ec3` を用意しています。

```bash
cd /workspace/ec3-prediction-app
source /workspace/.venvs/ec3/bin/activate
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

モデルがない場合、ホーム画面で学習方法を案内します。画面の起動時・再読み込み時に自動で学習は行いません。別の信頼できる自作モデルを使う場合は、`EC3_ARTIFACT_PATH` 環境変数でパスを指定できます。joblib形式のため、出所不明のモデルを読み込まないでください。

## データと学習

原本3ファイルの内容は変更していません。既存の学習CSVは添付原本と同一です。追加された外部検証CSVとNotebookは、無断で公開しないよう `.gitignore` で除外しています。新しいチェックアウトでは、ご自身で管理する原本を次の場所に置いてください。

```text
data/DASS_join_data_opera_delete_LLNAnan.csv
data/newCE_join_data_opera_delete_LLNAnan_duplicate_plus_manual.csv
reference_private/★XGBoost_model_refact_tokunaga.ipynb
```

学習データ154件、外部検証74件（Notebookと同じく分子量欠損2件を除いて72件）。外部検証データは学習へ混合しません。データの照合用SHA256は [data_provenance.json](docs/data_provenance.json) を参照してください。

動作確認用の軽量学習を明示して実行する場合:

```bash
python train.py --mode smoke \
  --train-csv data/DASS_join_data_opera_delete_LLNAnan.csv \
  --exval-csv data/newCE_join_data_opera_delete_LLNAnan_duplicate_plus_manual.csv \
  --n-trials 2 --max-estimators 200 \
  --artifact-path artifacts/smoke_model_artifact.joblib \
  --output-dir outputs/smoke
EC3_ARTIFACT_PATH=artifacts/smoke_model_artifact.joblib \
  streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

原本の本番条件で実行する場合（長時間かかります）:

```bash
python train.py --mode notebook \
  --train-csv data/DASS_join_data_opera_delete_LLNAnan.csv \
  --exval-csv data/newCE_join_data_opera_delete_LLNAnan_duplicate_plus_manual.csv \
  --artifact-path artifacts/notebook_model_artifact.joblib \
  --output-dir outputs/notebook
EC3_ARTIFACT_PATH=artifacts/notebook_model_artifact.joblib \
  streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

原本再現モードは `random_state=1037`、5分割、Boruta (RandomForest深さ5、perc=100)、各FoldでOptuna 1000 trials、内部5分割、上限100000木とearly stoppingを使用します。探索範囲・目的関数・最終木数の決定規則を原本に合わせます。軽量モードでもBorutaと実際のOptuna探索を実行し、探索回数と木数上限だけを明示的に減らします。Optunaを無効にして別のモデルを作りません。

ADのcoverageは原本で「**AD外にする割合**」です。`--ad-coverage 0.1` なら約10%を除外する閾値を使います。原本には0.0・0.1・0.2の検討があり、デフォルト0.1を採用し、値をモデルと画面に保存します。k=5、自己距離を除いた学習距離の順序統計を用い、一般的な補間百分位へ置換しません。

原本は全学習データでの特徴量選択後に外側CVを行うため、CV評価に情報の共有があります。再現優先でこの処理を保持し、CV結果を独立な汎化性能として扱いません。別のリーク対策アルゴリズムに黙って置換していません。in vitro ADのうち `hCLAT.Call` の原本には添付外のファイルが必要なため、不明な判定を推測せず欠損として表示します。

## 画面と予測入力

1. **ホーム**: EC3、モデル、NAMsの説明と学習条件。
2. **単一物質の予測**: 分子量とdescriptorを入力。空欄は学習済みKNNImputerで補完します。分子量だけは有限で正の値が必要です。
3. **CSV一括予測**: テンプレートをダウンロードし、同じ列名で入力します。UTF-8/BOM・CP932・Shift-JISに対応します。識別情報を保ち、結果をCSVで取得できます。
4. **モデルの性能評価**: 学習・OOF CV・外部検証を区別したR²/RMSE/MAE、実測対予測、Fold評価。
5. **SHAPによる説明**: 対象Foldを指定した重要度、Summary / Beeswarm / Waterfall と個別予測の寄与。

Notebookは452個の全descriptorで補完・標準化してから特徴量を選ぶため、CSVには**選択後の特徴量だけでなく全descriptor列が必要**です。列内の空欄は許可します。列そのものの欠落を黙って補いません。列順は保存モデルの順に揃えます。実測の `LLNA_EC3` 列は予測の必須入力ではありません。

各Foldの逆変換は `EC3 = MolWeight / 10**y`。最終EC3は原本の出力処理に合わせ、行ごとの**最小EC3**を採用し、変換後yも同じFoldから取得します。AD内予測はAD内Foldだけから最小EC3を採用し、候補がなければ欠損を返します。全Foldの値も出力します。

SHAPは `y = log10(MolWeight / EC3)` の寄与です。同じ分子量ならyを大きくする寄与はEC3を小さくします。異なるFoldのSHAPを単純平均したり、EC3へ直接加算したりしません。

アップロードはメモリ上で処理し、アプリが化学物質CSVを自動保存する処理はありません。CLIで明示した出力と学習時に生成するモデル・評価結果は保存します。

## CLIと検証

```bash
# モデルの列構成だけを持つテンプレート
python original_runtime.py --template-csv /tmp/ec3-template.csv

# 既知のdescriptor CSVによる予測（目的変数列は不要）
python original_runtime.py --input-csv /path/to/descriptors.csv \
  --output-csv /tmp/predictions.csv

python -m compileall -q train.py original_runtime.py utils_io.py utils_ml.py streamlit_app.py app_original.py
python -m pytest -q
```

原本ソースとの比較は `scripts/verify_notebook.py --help` を参照してください。原本Notebookの計算クラスを独立に抽出して実行し、前処理、選択、Fold、予測、逆変換、ADを比較します。軽量条件と本番1000 trials条件の両方で112項目の差分0、さらにADの0.0・0.1・0.2設定で54項目の差分0を確認済みです。当時のPCでの未保存モデルとの一致や、未提供のhCLAT資料との一致を意味しません。

```bash
python scripts/verify_notebook.py --n-trials 1000 --max-estimators 100000 \
  --use-existing-artifact --artifact-path artifacts/notebook_model_artifact.joblib \
  --output outputs/notebook_comparison_full.json
```

比較対象の `artifacts/notebook_model_artifact.joblib` は生成時のまま保持しています。既定の `artifacts/original_model_artifact.joblib` は同じ学習結果に検証範囲・元ファイルSHA256などのメタデータを追記したコピーです。軽量版を試す場合は明示的に `EC3_ARTIFACT_PATH` を指定し、本番モデルを上書きしないでください。

`artifacts/`・`outputs/`・`validation_runs/` はGit管理しません。生成モデルのサイズを確認し、GitHubの通常ファイル100 MiB制限を超える場合は、権利を確認した上でGit LFSやアクセス制限付きストレージを使ってください。Streamlit Community CloudではPython 3.12を選び、権利確認済みのモデルを配備して `streamlit_app.py` を入口にしてください。

## 構成と互換性

`train.py` は学習・外部検証・保存、`original_runtime.py` は保存済みFoldの推論、`utils_io.py` はCSVと入力検査、`utils_ml.py` は原本の計算、`streamlit_app.py` は画面を担当します。`app_original.py` は画面への互換入口です。

旧 `app.py` / `app_research.py` / `predict.py` / `train_model*.py` と既存 `model*` は互換用に残しています。これらの固定23特徴量モデルはNotebook再現モデルではありません。[既存ファイルの調査記録](docs/repository_inventory.md) で役割と差分を説明しています。
