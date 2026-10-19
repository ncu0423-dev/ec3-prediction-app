# LLNA EC3 皮膚感作性予測アプリ

XGBoost の各Foldモデルを保存し、単一物質・CSV一括予測、k近傍 Applicability Domain (AD)、性能評価、Fold別 SHAP を提供する Streamlit アプリです。NAMs（動物実験代替手法）の活用を目指す研究用ツールです。

**現在の配備版は、指定された23項目だけを使う固定説明変数モデルです。** 欠損補完・標準化も23項目の中で行い、Borutaで増減しません。詳細と列順は [23項目版](docs/fixed23.md) を参照してください。以前の452項目/Boruta版は原本クラスとの112項目完全一致を確認して保持していますが、その一致結果・性能指標は現在の23項目版に流用しません。[Notebook監査](docs/notebook_audit.md) と [以前の検証記録](docs/validation.md) は旧Notebook再現版の記録です。

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
python train.py --mode smoke --feature-mode fixed23 \
  --train-csv data/DASS_join_data_opera_delete_LLNAnan.csv \
  --exval-csv data/newCE_join_data_opera_delete_LLNAnan_duplicate_plus_manual.csv \
  --n-trials 2 --max-estimators 200 \
  --artifact-path artifacts/fixed23_smoke_model_artifact.joblib \
  --output-dir outputs/fixed23_smoke
EC3_ARTIFACT_PATH=artifacts/fixed23_smoke_model_artifact.joblib \
  streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

23項目版を本番の探索回数で実行する場合（長時間かかります）:

```bash
python train.py --mode notebook --feature-mode fixed23 \
  --train-csv data/DASS_join_data_opera_delete_LLNAnan.csv \
  --exval-csv data/newCE_join_data_opera_delete_LLNAnan_duplicate_plus_manual.csv \
  --artifact-path artifacts/fixed23_model_artifact.joblib \
  --output-dir outputs/fixed23_notebook
EC3_ARTIFACT_PATH=artifacts/fixed23_model_artifact.joblib \
  streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

`--mode notebook` は本番探索予算（各Fold 1000 trials / 上限100000木）、`--mode smoke` は軽量予算を指定します。説明変数の方式は別の `--feature-mode` で明示します。`fixed23` は指定23項目、`boruta` は旧Notebookの452項目からの選択です。固定23版でも `random_state=1037`、外側5分割・内側5分割、Optuna探索範囲と目的関数、early stopping、EC3変換・集約・ADの規則を維持します。23項目版ではBorutaを実行しません。Optunaを無効にして別のモデルを作りません。

ADのcoverageは原本で「**AD外にする割合**」です。`--ad-coverage 0.1` なら約10%を除外する閾値を使います。原本には0.0・0.1・0.2の検討があり、デフォルト0.1を採用し、値をモデルと画面に保存します。k=5、自己距離を除いた学習距離の順序統計を用い、一般的な補間百分位へ置換しません。

旧NotebookのBoruta版は全学習データでの特徴量選択後に外側CVを行うため、CV評価に情報の共有があります。23項目版ではその特徴量選択を行いませんが、内側CV前の前処理など既存手順を維持します。in vitro ADのうち `hCLAT.Call` の原本には添付外のファイルが必要なため、不明な判定を推測せず欠損として表示します。

## 画面と予測入力

1. **ホーム**: EC3、モデル、NAMsの説明と学習条件。
2. **単一物質の予測**: 分子量とdescriptorを入力。空欄は学習済みKNNImputerで補完します。分子量だけは有限で正の値が必要です。
3. **CSV一括予測**: テンプレートをダウンロードし、同じ列名で入力します。UTF-8/BOM・CP932・Shift-JISに対応します。識別情報を保ち、結果をCSVで取得できます。
4. **モデルの性能評価**: 学習・OOF CV・外部検証を区別したR²/RMSE/MAE、実測対予測、Fold評価。
5. **SHAPによる説明**: 対象Foldを指定した重要度、Summary / Beeswarm / Waterfall と個別予測の寄与。

現在のCSV入力に必要なのは、指定された**23項目の列だけ**です。列内の空欄は許可します。MolWeightは有限かつ正の値が必要です。列そのものの欠落を黙って補いません。列順は保存モデルの順に揃えます。Name・CAS等の識別情報は任意で追加でき、実測の `LLNA_EC3` 列は予測の必須入力ではありません。旧452項目モデルを明示的に選ぶ場合だけ、そのモデル用の全入力列が必要です。

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

**以下の原本比較は旧452項目/Boruta版だけが対象です。** 原本ソースとの比較は `scripts/verify_notebook.py --help` を参照してください。原本Notebookの計算クラスを独立に抽出して実行し、前処理、選択、Fold、予測、逆変換、ADを比較します。旧版の軽量条件と本番1000 trials条件の両方で112項目の差分0、さらにADの0.0・0.1・0.2設定で54項目の差分0を確認済みです。当時のPCでの未保存モデルとの一致や、現在の23項目版との一致を意味しません。旧方式の学習は `python train.py --mode notebook --feature-mode boruta --artifact-path artifacts/notebook_model_artifact.joblib` で明示します。

```bash
python scripts/verify_notebook.py --n-trials 1000 --max-estimators 100000 \
  --use-existing-artifact --artifact-path artifacts/notebook_model_artifact.joblib \
  --output outputs/notebook_comparison_full.json
```

旧版の比較対象 `artifacts/notebook_model_artifact.joblib` は生成時のまま保持しています。既定の `artifacts/original_model_artifact.joblib` は23項目で再学習した別モデルのCommunity Cloud配備版です。評価表の物質名・CAS・SMILES・元行番号と、推論に不要な全体学習データを除いています。KNN補完・AD・SHAPに必要な数値参照データは残します。軽量版を試す場合は明示的に `EC3_ARTIFACT_PATH` を指定し、本番モデルを上書きしないでください。

非公開リポジトリでの配備用に `artifacts/original_model_artifact.joblib` だけをGit管理します。その他の `artifacts/`・`outputs/`・`validation_runs/` と原本Notebook・外部CSVは引き続き除外します。モデルは100 MiB未満で通常のGitで扱えます。Streamlit Community CloudではPython 3.12、入口 `streamlit_app.py`、ブランチ `codex/notebook-faithful-ec3-app` を指定してください。[配備手順](docs/community_cloud.md) を参照してください。

## 構成と互換性

`train.py` は学習・外部検証・保存、`original_runtime.py` は保存済みFoldの推論、`utils_io.py` はCSVと入力検査、`utils_ml.py` は原本の計算、`streamlit_app.py` は画面を担当します。`app_original.py` は画面への互換入口です。

旧 `app.py` / `app_research.py` / `predict.py` / `train_model*.py` と既存 `model*` は互換用に残しています。これらの固定23特徴量モデルはNotebook再現モデルではありません。[既存ファイルの調査記録](docs/repository_inventory.md) で役割と差分を説明しています。
