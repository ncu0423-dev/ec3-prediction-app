# 調査した既存ファイルと今回の役割

Git 履歴を保持し、既存の学習済みモデルと原本 CSV は上書きしません。

| ファイル | 調査時の状態・役割 | 対応 |
| --- | --- | --- |
| `streamlit_app.py` | Streamlit の雛形のみ | 新しい5画面の入口へ変更 |
| `requirements.txt` | streamlit / shap / matplotlib / pillow のみ、バージョン指定なし | 推論・Boruta・Optuna・学習に必要な依存を明記・固定 |
| `app.py` | 固定23特徴量・全件再学習モデルによる旧予測画面 | 互換用に維持、アップロード用一時ファイルの削除を追加 |
| `app_research.py` | 旧モデルの性能表示・保存済みFold SHAP画像 | 互換用に維持、SHAPの方向の説明を修正 |
| `predict.py` | 旧モデルの単一Pipeline推論・分類・AD | 互換用に維持、新しいNotebook再現モデルには使用しない |
| `train_model.py` | 固定23特徴量・random_state=0・150木・Optunaなし・CV後全件fit | 旧ワークフローとして維持 |
| `train_model_research.py` | 同上、各Foldのモデルと画像も保存 | 旧ワークフローとして維持 |
| `model/`, `model_research/` | 旧学習済みモデル・指標・テンプレート・SHAP画像 | 上書きせず維持 |
| `data/DASS_join_data_opera_delete_LLNAnan.csv` | 学習用154行・459列、添付とバイト単位で同一 | 内容を変更せず使用 |
| `.devcontainer/devcontainer.json` | Python 3.11、未固定依存導入、CSRF/CORSを無効化する起動設定 | 検証環境と同じPython 3.12へ、標準の保護設定で起動 |
| `.github/CODEOWNERS`, `LICENSE` | 所有者設定・Apache 2.0ライセンス | 維持 |
| `.gitignore` | Pythonの一般的な生成物等の除外 | 非公開参照資料・外部データ・新規生成モデルと出力を除外 |

新しい `train.py` / `original_runtime.py` / `utils_io.py` / `utils_ml.py` は、添付された途中コードの目的を引き継ぎます。重複したFold選択・入出力チェックなどを共通化し、原本Notebookの計算に合わせます。`app_original.py` は新画面への互換入口です。原本3ファイルは `docs/data_provenance.json` のSHA256で識別できます。

原本Notebookと外部CSV、添付された途中コードはローカルの無変更コピーを保持し、Gitへ追加しません。既存のGit管理対象の学習CSVの扱いは変更していません。

旧モデルはNotebook再現モデルではありません。新しい画面では旧モデルを自動読込して再現済みとして表示しません。Notebookとの差分と検証範囲は `notebook_audit.md`、実行した確認は `validation.md` を参照してください。
