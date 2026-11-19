# LLNA EC3 皮膚感作性予測アプリ（Notebook再現版）

**通常の配備版を元Notebookの方式へ戻しました。** 学習・外部検証とも、452項目でKNN欠損補完と標準化を行い、その後にBoruta選択項目をモデルへ渡します。現在の選択結果は、利用者が提示した21項目と一致しています。21項目だけで前処理していた固定版とは別のモデルです。

予測・学習条件はNotebookセル41–42、最終CSVはセル265/270を基準にします。末尾のAD外割合0.2を既定にし、初期0.0・別枝0.1も明示的に指定できます。必要なhCLAT元資料2ファイルは未提供なので、追加in vitro ADと、それに依存する解析は未再現です。**全セル・過去PCの未保存モデルまで完全一致したという意味ではありません。** [復旧監査](docs/notebook_restore_audit.md) と [現在の復旧・検証記録](docs/notebook_restored.md) を参照してください。

## 起動

Python3.12を使用します。依存バージョンは `requirements.txt` に固定しています。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

このクラウド作業環境では `/workspace/.venvs/ec3/bin/activate` を使用できます。Codespacesでは `.devcontainer/devcontainer.json` が依存を導入して起動し、Portsの8501からアクセスできます。Community CloudではPython3.12、ブランチ `codex/notebook-faithful-ec3-app`、入口 `streamlit_app.py` を指定してください。更新後は再読み込み、必要ならReboot appを行います。[配備手順](docs/community_cloud.md)。

既定のモデルは `artifacts/original_model_artifact.joblib` です。非公開リポジトリへ、識別情報を除いた配備モデルを同梱しています。画面起動時に再学習しません。別モデルを選ぶ場合だけ `EC3_ARTIFACT_PATH` を明示してください。joblibは信頼できる自作モデルだけを読み込んでください。

## 前処理・選択・学習の順序

1. 元のCSVから先頭6識別列と最終目的変数列を除き、元順序で452項目を抽出。MolWeightも含みます。
2. 全学習データで `KNNImputer(n_neighbors=5, weights='uniform')` → `StandardScaler` をfitしてBoruta用の入力を作成。
3. RandomForest深さ5、Boruta perc100、random_state0、support_で選択。現在は21項目を選択します。固定リストで置換しません。
4. 外側5-foldの各学習部分で、452項目のKNN補完→標準化をfitし直す。その後に同じBorutaマスクを適用。
5. 外側の検証部分・外部検証・新規入力は、保存した**各Foldの学習済み452項目前処理**でtransformし、選択項目を抽出。
6. 内側5-foldによるOptuna探索後、各外側モデルをfit。分割seed1037、内側seed0、XGBoost seed0。各Fold1000 trials、上限100000木、early stopping15。探索範囲・最良木数の決め方は原本を保持。

KNNの近さは標準化の前に計算します。全学習データでのBoruta選択と、内側CV前の前処理という原本の情報共有はそのまま維持し、別のリーク対策方式へ勝手に変えません。

## 必要な入力

**CSVには452項目すべての列名が必要です。** MolWeightはその中に含まれるので、換算用の別列を追加して453列にする必要はありません。選択21項目だけのCSVは使えません。テンプレートを新しく取得してください。

セルの欠損は原本と同じKNNで補完しますが、未選択431項目を空欄で追加しても、既知の全項目がある元CSVと同じ変換になるとは限りません。全ての既知のdescriptor値を含むCSVを推奨します。MolWeightはEC3換算にも必要なので有限・正の値が必須です。Name・CAS等の識別列は任意で追加でき、行順とともに結果へ保持します。実測 `LLNA_EC3` は予測入力に不要です。

アップロードはセッションのメモリで扱い、CSVを自動保存しません。UTF-8/BOM・CP932・Shift-JISに対応します。CLIで明示した出力、学習で作成したモデル・評価結果は保存します。

## 予測と評価

- `y=log10(MolWeight / EC3)`、`EC3=MolWeight / 10**y`。原本と同じfloat32予測値の演算順を保持。
- 5 Foldの最小EC3を最終予測とし、yも同じFoldから取得。AD内予測はAD内Foldだけから最小EC3を選び、候補なしは欠損。
- kNN ADは選択・標準化後の空間、k=5。自己距離を除いた学習距離平均の順序統計で閾値を求めます。coverageはAD外割合で、最終枝の0.2が既定。全体ADは1つ以上のFoldがAD内かで表示します。
- 学習部分・OOF・外部72件を区別。R²/RMSE、厳密な1/5～5倍範囲、GHS/ECETOCの原本境界・予測行×実測列の分類表を表示。MAEは追加指標と明示。
- 外部の最小EC3と平均値を区別。yの平均とEC3の平均は別々に計算し、相互の逆変換とはしません。
- SHAPはFoldごとの全学習行、全選択項目を使用。原本のviolin Summary、Waterfall、重要度 `abs(sum(SHAP))` とその数値的なFold間平均を保持します。Beeswarmと新規入力の説明は追加機能として区別。SHAPはy空間の寄与で、EC3への直接加算ではありません。

外部全72件のEC3 R²は約-0.712です。計算の再現と、実用上十分な予測性能は別です。Notebook中の事後的に最高R²を選ぶ38件の部分集合と、外部全件評価を混同しません。

## 原本データと学習・復旧

原本Notebookと2つのCSVは変更していません。学習154件、外部74件はMolWeight欠損2件を除いて72件。外部データを学習へ混合しません。照合用SHA256は [data_provenance.json](docs/data_provenance.json) に記録しています。原本Notebookと外部CSVはGitから除外しています。

```text
data/DASS_join_data_opera_delete_LLNAnan.csv
data/newCE_join_data_opera_delete_LLNAnan_duplicate_plus_manual.csv
reference_private/★XGBoost_model_refact_tokunaga.ipynb
```

元モデルを再学習する場合（長時間）:

```bash
python train.py --mode notebook --feature-mode boruta --fold-workers 4 \
  --ad-coverage .2 --artifact-path artifacts/notebook_trained_model_artifact.joblib \
  --output-dir outputs/notebook_trained
```

軽量検証は `--mode smoke --n-trials 2 --max-estimators 200` を明示し、本番と混同しません。Borutaは軽量化せずOptunaも実際に実行します。以前の `fixed21` / `fixed23` モードは履歴再現用として残しますが、既定のアプリ・CLIでは使いません。

今回の復旧では、既存の1000 trials×5モデルと全原本ハッシュ・ライブラリ版・112項目の独立照合記録が一致したため、その重みを再利用しました。再fitせず、最終AD=.2の判定・評価を再計算しています。

```bash
python scripts/restore_notebook_model.py
python scripts/export_deployment_model.py \
  --source artifacts/notebook_restored_model_artifact.joblib \
  --output artifacts/original_model_artifact.joblib
```

比較済みの `artifacts/notebook_model_artifact.joblib` はバイト単位で変更していません。再学習時にも別の保存先を使ってください。

## 検証コマンド

```bash
EC3_TEST_ARTIFACT=artifacts/original_model_artifact.joblib python -m pytest -q
python original_runtime.py --template-csv /tmp/ec3-template.csv
python original_runtime.py --input-csv /path/to/full-descriptors.csv --output-csv /tmp/predictions.csv

# 現在コードを原本クラスと別々に軽量学習して照合
python scripts/verify_notebook.py --ad-coverage .2 \
  --artifact-path artifacts/notebook_current_smoke.joblib \
  --output outputs/notebook_restored/current_code_source_comparison.json

# 保存モデルで原本のAD0/.1/.2を照合（再学習なし）
python scripts/verify_notebook.py --ad-coverage-only \
  --artifact-path artifacts/notebook_restored_model_artifact.joblib \
  --output outputs/notebook_restored/ad_source_comparison.json
```

元PCの最終モデル・予測値はNotebookに保存されていないため、過去実行との一致は証明できません。現在の固定環境での原本クラスとの照合と区別します。

## 未提供情報と構成

追加in vitro ADには `Annex-2-in-vitro-in-silico-in-vivo-defined-approaches.csv` と `newCE_data.csv` のhCLAT実験callが必要です。`hCLAT.score`で推測代用せず、未取得と表示します。2000 seed探索・10000回の部分集合解析・矛盾する発表図は通常の学習・画面起動で自動実行しません。単一の「全セルをそのまま動かした仕様」とは呼びません。

`train.py` は学習、`original_runtime.py` は推論、`utils_io.py` は入力検査、`utils_ml.py` は原本計算、`notebook_metrics.py` は原本評価、`streamlit_app.py` は画面です。長い原本classを分離していますが、比較で数値を確認します。旧 `app.py` / `app_research.py` / `train_model*.py` / `predict.py` は旧固定モデルの互換用で、通常のNotebook再現アプリには使いません。[既存構成](docs/repository_inventory.md)、[旧21項目固定版](docs/fixed21.md)、[旧23項目固定版](docs/fixed23.md)。

Git管理する生成モデルは識別情報を除いた配備用 `artifacts/original_model_artifact.joblib` のみです。他の生成物と原本参照資料は除外します。数値の学習参照データはKNN/AD/SHAPに必要なため残り、完全匿名化ではありません。リポジトリとアプリの閲覧権限は別に管理してください。
