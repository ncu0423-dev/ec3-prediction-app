# LLNA EC3 皮膚感作性予測アプリ

**現在の入力CSVは、No.・Name・CAS Numberの3列＋指定21項目の計24列です。** 指定順の [CSVテンプレート](templates/csv21_input_template.csv) をアプリから取得し、アップロード後に各物質のMolWeight（分子量）を画面で入力します。全行の分子量が有効になってから予測できます。[現在の入力仕様と検証記録](docs/csv21_input.md)。

モデルは21項目だけで欠損補完・標準化・予測を行います。VdW surface DPSA2や他の431項目は要求しません。MolWeightはEC3換算専用で、説明変数・補完距離・AD・SHAPには使いません。**これは452項目前処理のNotebook再現版とは異なるモデルです。** Notebook版と固定21/23項目版は履歴として保持します。

## 使い方

1. 「CSV一括予測」でテンプレートをダウンロード。
2. No.・Name・CAS Numberと21項目を入力し、CSVを選択。
3. 画面の「物質ごとの分子量」表に各行のMolWeightを入力。
4. 「一括予測を実行」し、結果CSVを取得。

CSVにはMolWeight列を含める必要はありません。実測LLNA_EC3やSMILESも不要です。
既存CSVにMolWeightがある場合のみ任意の末尾列として受け取り、画面で確認できます。
識別列と行順を保ち、No.・CASが重複していても内部の入力行番号で分子量を対応付けます。
CSVが変われば以前の分子量や予測結果をそのまま流用しません。

モデルのターゲットは `y=log10(MolWeight / EC3)`、逆変換は `EC3=MolWeight / 10**y` です。
**CSVにMolWeightがないことと、EC3計算にMolWeightが不要なことは別です。** 分子量を一切使わずにEC3を出すにはターゲットを変更した別モデルの学習が必要で、今回は行いません。分子量を推測や固定値で埋めません。

各説明変数の空欄は、21項目でfitしたKNNImputerで補完します。全説明変数が欠損すると学習項目平均へフォールバックします。MolWeightは有限・正の値が必須です。非数値・無限大・列不足・重複列・指定外列はエラーにします。列順が違うCSVは列名で指定順に整列します。

## 起動・配備

Python3.12を使用します。依存バージョンはrequirements.txtで固定しています。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

この作業環境では `/workspace/.venvs/ec3/bin/activate` を使えます。
Codespacesではdevcontainer設定を利用し、Portsの8501からアクセスします。
Community CloudはPython3.12、ブランチ `codex/notebook-faithful-ec3-app`、入口 `streamlit_app.py`。
更新後に再読み込み、必要ならReboot appしてください。[配備手順](docs/community_cloud.md)。

既定の配備モデルは `artifacts/original_model_artifact.joblib`。
画面起動時に学習や元CSVの読み込みは行いません。モデル名は「指定21項目モデル」です。
`EC3_ARTIFACT_PATH` を設定している場合は古いNotebookモデル等を指していないか確認してください。
joblib形式は信頼できるモデルだけを読み込みます。

アップロードや追加入力した分子量はセッションメモリで扱い、自動でディスクへ保存しません。
モデルにはKNN/AD/SHAPに必要な数値参照データが残りますが、配備時の評価表から物質名・CAS・SMILES等を除きます。
GitHubのPrivate設定とCommunity Cloudの閲覧権限は別です。

## 学習条件・モデルの再利用

原本NotebookとDASS/newCE CSVは変更しません。学習154件、外部74件から分子量欠損2件を除いた72件は学習に混合しません。元データの照合用SHAは [data_provenance.json](docs/data_provenance.json)。

現在の固定21項目モデルは各Fold1000 trials（外側5-fold、内側5-fold）、random_state1037、上限100000木・early stopping15で学習済みのモデルを再利用しました。元モデルSHA、21列の補完器・XGBoost、1000試行履歴、9ライブラリ版と原本データSHAが検証記録に一致したため、重みを再学習しません。直前のサービスと同じAD外割合.2を維持し、閾値・AD関連評価のみ再計算しています。

```bash
# 検証済み固定21項目モデルから、現在の画面用プロファイルを再構成
python scripts/prepare_csv21_model.py
python scripts/export_deployment_model.py \
  --source artifacts/csv21_model_artifact.joblib \
  --output artifacts/original_model_artifact.joblib

# 明示的に新しい固定21項目モデルを学習する場合（長時間）
python train.py --mode notebook --feature-mode fixed21 --fold-workers 4 \
  --ad-coverage .2 --artifact-path artifacts/new_fixed21_model_artifact.joblib \
  --output-dir outputs/new_fixed21
```

欠損補完→標準化の順序、yの変換・逆変換、最小EC3採用、k=5のAD規則は従来を維持します。
AD全体は1つ以上のFoldが範囲内か、AD限定予測は対象Foldのみの最小EC3です。
MAEや分類表、SHAPは対象モデルの値で表示し、旧Notebookの一致記録や精度を流用しません。
外部72件のEC3 R²は約-0.680で、十分な予測性能を示していません。[現在の検証数値](docs/csv21_validation.json)。

## テストと構成

```bash
EC3_TEST_ARTIFACT=artifacts/original_model_artifact.joblib python -m pytest -q
```

Web用24列CSVと、バックエンド内部の換算用入力（21項目＋MolWeight）は別です。
CLIで直接予測する場合は、既知のMolWeight列を追加した入力を指定してください。
通常の利用者はアプリのCSV画面と分子量入力表を使います。

`csv_input_schema.py` は24列の順序・検証、`csv_batch_inputs.py` は分子量の行対応、
`streamlit_app.py` は画面、`original_runtime.py` は保存モデルの推論、`train.py` は学習です。
`utils_io.py` はデータ読み込み、`utils_ml.py` は共通モデル計算、`notebook_metrics.py` は評価計算です。

## 履歴

452項目の元Notebook再現版のモデル・記録は保持していますが、通常のアプリには使用しません。
その版の112/166項目一致の証拠は、現在の21項目だけの前処理へ適用できません。
[Notebook復旧記録](docs/notebook_restored.md)、[原本監査](docs/notebook_audit.md)、
[固定21項目版の元の学習](docs/fixed21.md)、[旧23項目版](docs/fixed23.md)。
元Notebookの追加in vitro ADに必要なhCLAT資料2ファイルは未提供です。
旧app.py/app_research.py/train_model*.py/predict.pyも互換用に残しますが、現在のアプリの入口はstreamlit_app.pyです。

Git管理する生成モデルは配備用original_model_artifact.joblibのみ。
原本Notebook・外部CSV・他の生成物はGitから除外しています。
