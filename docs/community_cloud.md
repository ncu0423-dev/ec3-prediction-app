# Streamlit Community Cloud（24列CSV・21項目モデル）

- Repository: ncu0423-dev/ec3-prediction-app（Private）
- Branch: codex/notebook-faithful-ec3-app
- Main file path: streamlit_app.py
- Python: 3.12

既定のモデルは artifacts/original_model_artifact.joblib。
更新後は再読み込みし、必要なら管理画面でReboot appします。
ホームのモデル名が「指定21項目モデル」、CSVテンプレートがNo.・Name・CAS Number＋21項目の24列なら更新済みです。
CSVアップロード後、画面の表で各行のMolWeightを入力し、予測ボタンを押します。
全行が有限・正の分子量になるまで予測は実行できません。CSVへMolWeight列を追加する必要はありません。

以前の452項目版を指すEC3_ARTIFACT_PATH/ARTIFACT_PATHをSecrets等に設定している場合は外してください。
通常はどちらも不要です。今回のモデルは21項目で前処理するため、元Notebookの452項目版と結果は異なります。
詳しくはcsv21_input.mdを参照してください。

リポジトリのPrivate設定とアプリの閲覧権限は別です。配備モデルの評価表には物質名・CAS・SMILES等を含めませんが、学習由来の数値情報は必要なため残ります。公開範囲はCommunity Cloud側でも管理してください。
