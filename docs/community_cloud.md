# Streamlit Community Cloudへの配備

このブランチには、信頼できる環境で学習・検証した本番モデルを
`artifacts/original_model_artifact.joblib` に同梱しています。起動時に再学習せず、
外部CSVやNotebookが配備先になくても画面・予測・AD・SHAPが動きます。

## アプリ設定

- Repository: `ncu0423-dev/ec3-prediction-app`（Private）
- Branch: `codex/notebook-faithful-ec3-app`
- Main file path: `streamlit_app.py`
- Python: `3.12`

この設定で起動済みなら、GitHubの更新が反映された後にアプリを再読み込みしてください。
モデル未検出の表示が残る場合は、Community Cloudの管理画面で **Reboot app** を実行します。
モデルのダウンロードや、画面からのjoblibアップロードは不要です。

ホームと左メニューに「Notebook再現条件」と表示されればモデルを読み込めています。
「CSV一括予測」で入力テンプレートを取得し、入力後に「一括予測を実行」を押してください。
単一物質は「単一物質の予測」で分子量と既知の特徴量を入力します。

`EC3_ARTIFACT_PATH` または `ARTIFACT_PATH` を以前Secrets等に設定している場合は、
既定以外のパスを参照していないか確認してください。通常はどちらも設定不要です。

## 配備モデルに含めるもの

- 学習済み5 FoldのXGBoost、前処理、特徴量順序、AD閾値
- KNN補完・ADに必要な数値の学習参照データ
- Fold別SHAPの数値参照データ、性能指標、識別情報を除いた評価値
- 学習条件、ライブラリ版、原本比較の検証情報

評価表のName・CAS Number・SMILES・No.・元行番号や未知のメモ列を含めません。
推論に使わない全体Boruta用行列等も含めません。モデルは数値の学習情報を保持するため、
すべての学習情報を匿名化したものではありません。リポジトリのPrivate設定と
アプリの閲覧権限は別なので、必要に応じてCommunity Cloudの共有設定も制限してください。

## 管理者による再出力

識別情報を含む検証済み原本モデルは配備用ファイルと別のローカルパスに保持します。
信頼できる原本から次のように再出力できます（学習・数値の再計算はしません）。

```bash
python scripts/export_deployment_model.py \
  --source /path/to/trusted_verified_model.joblib \
  --output artifacts/original_model_artifact.joblib
```

元の `artifacts/notebook_model_artifact.joblib` は比較時のバイト列を維持しています。
配備版は識別列等を除いたためSHA256は異なりますが、予測値・AD・SHAPの数値は同じです。
元のNotebookとの比較と外部予測精度の限界は `validation.md` を参照してください。
