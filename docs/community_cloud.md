# Streamlit Community Cloud（21項目版・452項目版）

- Repository: ncu0423-dev/ec3-prediction-app（Private）
- Branch: codex/notebook-faithful-ec3-app
- Main file path: streamlit_app.py
- Python: 3.12

両方の学習済み配備モデルをリポジトリに含めています。画面起動時の学習は不要です。

| モード | 配備ファイル | 入力テンプレート |
| --- | --- | --- |
| 452項目版（元Notebook方式）・既定 | artifacts/notebook_deployment_artifact.joblib | No.・Name・CAS Number＋452項目＝455列。MolWeightを含む |
| 21項目版 | artifacts/original_model_artifact.joblib | No.・Name・CAS Number＋21項目＝24列。MolWeightはアップロード後に画面で入力 |

更新後は再読み込みし、必要なら管理画面でReboot appします。サイドバーで2方式を切り替えられ、それぞれのCSVテンプレートを取得できれば更新済みです。モデルを切り替えると、入力編集・予測・SHAPなどの状態をリセットします。

通常はモデルパスの環境変数を設定する必要はありません。独自モデルを使う場合は、`EC3_FIXED21_ARTIFACT_PATH` と `EC3_NOTEBOOK452_ARTIFACT_PATH` で方式ごとに設定します。旧 `EC3_ARTIFACT_PATH` / `ARTIFACT_PATH` が残っている場合は、そのモデルの方式が既定になります。新しい既定の452項目版を使用する場合は、旧設定を外してください。

452項目版は元Notebookと同じ452項目で欠損補完・標準化し、選択21項目で予測します。DASSの検証用予測（OOF）は「モデルの性能評価」から確認・ダウンロードします。単一・CSV一括予測は新規入力として5 Foldの最小EC3を採用するため、DASSのOOFとは区別してください。[方式と再現範囲](dual_models.md)。

リポジトリのPrivate設定とアプリの閲覧権限は別です。配備モデルの評価表には物質名・CAS・SMILES等を含めませんが、学習由来の数値情報は必要なため残ります。公開範囲はCommunity Cloud側でも管理してください。
