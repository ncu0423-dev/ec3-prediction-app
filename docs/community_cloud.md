# Streamlit Community Cloud（Notebook再現版）

既定の配備モデルを元Notebookの452項目前処理→Boruta選択→5 Fold予測へ戻しました。
学習済みモデルは同じ `artifacts/original_model_artifact.joblib` に同梱しています。
画面起動時に再学習や原本CSVの読み込みはしません。

- Repository: `ncu0423-dev/ec3-prediction-app`（Private）
- Branch: `codex/notebook-faithful-ec3-app`
- Main file path: `streamlit_app.py`
- Python: `3.12`

更新後、アプリを再読み込みし、必要なら管理画面で **Reboot app** を実行してください。
ホームに「Notebook再現条件」、前処理入力452項目・Boruta選択21項目が表示されます。
CSVテンプレートは新しく取得してください。旧21列/22列だけの入力は使えません。
MolWeightは452列の中に含まれ、前処理とEC3換算に使います。
未選択431列を単に空欄で足しても、値の揃った元データと同じ変換になるとは限りません。

`EC3_ARTIFACT_PATH` / `ARTIFACT_PATH` を以前Secrets等に設定している場合は、
古い固定21/23項目モデルを指していないか確認してください。通常は設定不要です。

配備用モデルは物質名・CAS・SMILES・元行番号等を評価表から除き、
前処理・AD・SHAPに必要な数値参照データを保持します。GitHubがPrivateでも
Community Cloudの閲覧権限は別なので、共有範囲はサービスの設定で管理してください。

追加in vitro ADに必要なhCLAT元資料は未提供です。画面に未取得と表示し、
全仕様が完全に再現済みとは表示しません。復旧内容・限界は `notebook_restored.md`、
原本との対応は `notebook_restore_audit.md` を参照してください。
