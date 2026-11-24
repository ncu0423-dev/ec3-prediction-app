# LLNA EC3 皮膚感作性予測アプリ

サイドバーで **「452項目版（元Notebook方式）」と「21項目版」** を切り替えて使えます。既定は452項目版です。入力・CSVテンプレート・予測・性能評価・SHAPは、選んだモデルに対応します。

| モード | 前処理と予測 | 入力CSV |
| --- | --- | --- |
| 452項目版（元Notebook方式） | 全452項目でKNN欠損補完→標準化し、Borutaで選択した21項目をXGBoostへ渡す | No.・Name・CAS Number＋452項目の計455列。MolWeightは452項目に含む |
| 21項目版 | 指定21項目だけで欠損補完→標準化→予測する別モデル | No.・Name・CAS Number＋21項目の計24列。MolWeightはアップロード後に画面で入力 |

同じ21項目を最終予測に使っていても、補完距離・学習済みモデルが異なるため、両モードの結果は一致するとは限りません。[方式と再現範囲](docs/dual_models.md)。

## 使い方

1. サイドバーで使用するモードを選びます。
2. 「CSV一括予測」で、そのモードのテンプレートをダウンロードします。
3. CSVをアップロードします。21項目版では物質ごとの分子量も入力します。
4. 「一括予測を実行」から結果CSVを取得します。

[21項目版テンプレート](templates/csv21_input_template.csv) / [452項目版テンプレート](templates/csv452_input_template.csv)

452項目版は、選択後の21項目だけのCSVを受け付けません。元の全項目入りCSVにある既知の値を含めてください。残りを空欄の列で追加しても、元の完全なデータと同じ補完結果は保証できません。数値の欠損セルは許可しますが、非数値・無限大・必要列の不足・重複列はエラーになります。列順はモデルの入力順にそろえます。

21項目版は指定24列順のCSVを使います。MolWeightは任意の末尾列として受け取り、画面で確認することもできます。No.・Name・CASと行順を保持し、識別情報が重複しても入力行番号で分子量を対応付けます。モデルを切り替えると入力編集・予測・SHAP等のモデルに依存する状態をリセットし、別モデルの結果を流用しません。

両モードの目的変数は `y=log10(MolWeight / EC3)`、逆変換は `EC3=MolWeight / 10**y` です。EC3算出には各物質の有限で正のMolWeightが必要です。21項目版では換算専用、452項目版では欠損補完の距離にも使います。

## DASSの検証結果と新規物質の予測

元NotebookのDASS結果は、各物質を学習から除いたFoldによる検証予測（OOF）です。「モデルの性能評価」→「交差検証（OOF）」で、そのモードに保存された検証結果を確認・ダウンロードできます。

単一物質・CSV一括予測は、新規入力として5 Foldで予測し、元Notebookの新規・外部予測と同様に最小EC3を採用します。DASSの学習物質を再入力しても、OOFとは使用するFoldが異なるため一致するとは限りません。元のDASS結果を比較するときは、CSV一括予測とOOFを混同しないでください。

## 起動・配備

Python 3.12を使用し、依存バージョンは `requirements.txt` で固定しています。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

この作業環境では `/workspace/.venvs/ec3/bin/activate` を使えます。CodespacesではPortsの8501からアクセスします。Community CloudはPython 3.12、入口は `streamlit_app.py` です。[配備手順](docs/community_cloud.md)。

両方の配備用モデルをリポジトリに含めています。

- 21項目版：`artifacts/original_model_artifact.joblib`
- 452項目版：`artifacts/notebook_deployment_artifact.joblib`

画面の起動・切り替えで再学習しません。独自の配備モデルに差し替える場合は、それぞれ `EC3_FIXED21_ARTIFACT_PATH`、`EC3_NOTEBOOK452_ARTIFACT_PATH` を設定します。異なる方式のモデルを指定した場合はエラーにし、他方へ自動で切り替えません。旧 `EC3_ARTIFACT_PATH` / `ARTIFACT_PATH` は指定モデルの方式を判定し、その方式の既定選択・パスだけを置き換える互換設定です。もう一方の方式は引き続き選べます。通常はこれらの設定は不要です。joblibは信頼できるモデルだけを読み込みます。

アップロードしたCSV・追加入力した分子量・予測結果はセッションメモリで扱い、自動でディスクへ保存しません。配備モデルにはKNN/AD/SHAPに必要な数値参照データが残りますが、評価表の物質名・CAS・SMILES等は除いています。GitHubのPrivate設定とCommunity Cloudの閲覧権限は別です。

## 元Notebookの再現範囲

452項目版は、元Notebookの実行可能なモデルクラスと現在の固定ライブラリ環境で独立照合した学習済みモデルを使用します。元データ・NotebookのSHA、9ライブラリ版、各Fold1000 trialsの履歴を再確認し、前処理・モデルの重みを再学習せずに再利用しました。以前の本番条件の独立照合112項目はすべて最大差0で一致しています。今回も配備コピーの前処理・モデルを照合し、OOF154件、通常予測154件と外部72件の数値が元モデルと最大差0で一致しました。[検証記録](docs/dual_models_validation.json)。

これは、元PCで作成された過去のDASS CSVの数値との完全一致を証明するものではありません。そのCSVを作った当時の保存モデルと環境は未提供です。また、hCLAT.Callの元資料2ファイルが未提供のため、in vitro ADおよびそれを使う複合ADは完全再現できません。未提供資料からの値を推測して補いません。kNN ADは利用できます。

21項目版は452項目前処理とは別モデルです。452項目版の一致記録を21項目版の再現性や性能の証拠として流用しません。外部データ72件は学習154件へ混合していません。両モデルの外部EC3 R²は負であり、再現性と予測性能は別に評価してください。

## テスト・構成

```bash
python -m pytest -q
```

`model_profiles.py` はモデルの選択と方式の検証、`streamlit_app.py` は画面、`original_runtime.py` は保存モデルの推論、`train.py` は学習です。`csv_input_schema.py` と `csv_batch_inputs.py` は21項目CSVと分子量の行対応、`utils_ml.py` は共通計算、`notebook_metrics.py` はNotebook方式の評価計算です。

配備モデルの準備は、信頼できる検証済みのローカル元モデルがある環境だけで実行します。通常のアプリ利用・起動には不要です。

```bash
python scripts/prepare_dual_models.py
```

原本Notebook・学習CSV・外部CSVと既存の検証済み元モデルは変更しません。原本Notebook、外部CSV、研究用の他の生成モデルはGitから除外しています。過去の記録：[原本監査](docs/notebook_audit.md)、[Notebook復旧記録](docs/notebook_restored.md)、[21項目版](docs/csv21_input.md)、[学習データ出典](docs/data_provenance.json)。旧 `app.py` / `app_research.py` は現在の入口ではありません。
