# 実行した検証と制限

検証環境: Linux x86_64 / Python 3.12.14。主要ライブラリは `requirements.txt` の固定バージョン。学習はXGBoost 1スレッド、Boruta 2スレッドで実行し、既存のモデル・原本データは変更していません。

## 入出力・モデル・画面のテスト

`python -m pytest -q` でCSVの文字コード・列名・行形式・数値検査、欠損値、分子量、変換式、Fold集約、ADの独立計算、モデル保存・再読込、単一／一括予測、外部検証、SHAP加法性、Streamlit画面を検証します。

本番モデルを指定した実行は **65 passed / 0 failed / 0 skipped、16.26秒** で完了しました。

```bash
EC3_TEST_ARTIFACT=artifacts/notebook_model_artifact.joblib python -m pytest -q
```

`compileall` による構文確認と `pip check` も成功。CLIでは目的変数を含まない3行のCSVから予測CSVと入力テンプレートを生成し、3件とも有限かつ正のEC3を確認しました。

検証メタデータ付きの既定モデルへ切り替えた後、再利用用インストールスクリプトを再実行し、**65 passed、14.63秒** を確認しました。サーバーを新しい `streamlit_app.py` で再起動し、health応答 `ok` を確認しています。環境設定の `install_script`・`start_skill` は更新してドラフト保存済みです。保存は環境のPublishやGitHubへのpushを行う操作ではありません。

画面テストはStreamlit AppTestを使用します。一括アップロードはAppTestにファイル設定APIがないため、メモリ上のファイルをアップローダーへ渡し、実際のパーサー・予測・結果表示・ダウンロードまで実行しています。HTTPサーバーの起動とhealth応答も別途確認しています。

単一行で分布のViolin図を描くことはできないため、必要な行数を案内し、重要度・Beeswarm・Waterfallは描画できることを回帰テストで確認しています。

## 軽量学習

```bash
python train.py --mode smoke --n-trials 2 --max-estimators 200 \
  --artifact-path artifacts/original_model_artifact.joblib --output-dir outputs/smoke
```

Borutaを100反復上限で実行し、452特徴量から21特徴量を選択。5外側Foldそれぞれで2 trials × 5内側Foldの実際のOptuna探索を行い、保存・再読み込みと外部72件の予測に成功しました。軽量モデルの保存コピーは `artifacts/smoke_model_artifact.joblib` にあります。

| 評価対象 | 件数 | R² (EC3) | RMSE (EC3 %) | MAE (EC3 %) |
| --- | ---: | ---: | ---: | ---: |
| 軽量モデル OOF | 154 | 0.353614 | 49.070790 | 28.630234 |
| 軽量モデル 外部検証（最小EC3） | 72 | -0.713000 | 93.810081 | 68.259275 |

これらは動作確認モデルの値です。精度が良いことを示す結果ではありません。特に外部R²は負であり、外部データへの汎化性能が十分と判断する根拠にはなりません。

## 独立した原本ソースとの比較

`scripts/verify_notebook.py` はNotebookセル15の計算クラスだけを独立に抽出して実行します。原本のローカル絶対パス・ファイル書込・2000seed探索・10000回の部分抽出・発表用図は実行しません。

軽量比較ではOptuna試行数を1000→2、木数上限を100000→200に限定し、その変更を報告します。全Boruta・5外側×5内側CVは省略しません。前処理・Borutaマスク・特徴量順・層別ラベル・Fold行番号・XGBoost設定・各Foldの学習／holdout／外部予測・逆変換・AD・集約・OOF・SHAP・Optuna履歴を比較し、**112項目が許容誤差0で一致**しました。最大差0、スキップなし。詳細はローカルの `outputs/notebook_comparison.json`。

## 本番条件

アプリ側の各Fold 1000 trials / 内部5分割 / 上限100000木 / early stopping15 の学習を完了しました。5つの学習履歴はいずれも1000 trialsです。`artifacts/notebook_model_artifact.joblib` に保存したモデルは1,905,896 bytes（約1.82 MiB）。100 MiB制限未満ですが、ローカル生成物としてGitへ追加しません。

| 本番条件の評価対象 | 件数 | R² (EC3) | RMSE (EC3 %) | MAE (EC3 %) |
| --- | ---: | ---: | ---: | ---: |
| OOF | 154 | 0.371845 | 48.373820 | 28.009199 |
| 外部検証（最小EC3） | 72 | -0.711566 | 93.770791 | 67.973230 |
| 外部検証（AD内Foldのみ） | 55 | -0.603607 | 89.975215 | 65.062603 |

OOFのy空間R²は0.639602、RMSEは0.651222。外部のy空間R²は-0.115959、RMSEは1.108192です。**本番条件でも外部R²は負です。実装の動作や原本計算の再現と、外部で十分な予測精度を示すことは別であり、この結果から実用上十分な性能とは判断できません。**

原本クラスによる独立した同条件（1000 trials × 5 Fold / 100000木上限）の学習と照合も完了し、**112項目がすべて完全一致、最大差0、許容誤差0** でした。`outputs/notebook_comparison_full.json` の `scope=full_notebook_source_parity`、`production_1000_trials_verified=true` を確認しています。当時の未保存出力ではなく、現在の固定環境で原本の計算クラスを独立実行した結果です。

通常の起動先 `artifacts/original_model_artifact.joblib` は本番モデルへ切り替え、検証範囲をメタデータに追記しています。比較した本番ファイル `artifacts/notebook_model_artifact.joblib` はバイト単位でそのまま保持し、SHA256は `0d1dc76ecb32cb3d6b3d00974a51bcf78161d7be323917d38179e679052bb4ad` です。元モデルと既定コピーの予測一致も確認しました。

本番モデルの学習をやり直さず、原本クラスのAD計算と保存した学習／holdout／外部入力を使ってcoverage 0.0 / 0.1 / 0.2を比較した追加検証は **54項目すべて完全一致、最大差0** でした。閾値・AD判定・少なくとも1FoldがAD内か・AD内候補だけの最小EC3を照合しています。モデルのデフォルトは明示的に0.1の枝であり、Notebook末尾の0.2の枝と混同しません。詳細は `outputs/notebook_ad_coverage_comparison.json`。

## 未提供情報・環境外の検証

- 原本Notebookに最終モデルの学習結果・予測値が保存されていないため、**当時のPCでの未保存モデル**との一致は検証不能です。現在の固定環境で原本ソースを実行した結果との比較と区別します。
- 追加in vitro ADに必要な2つのhCLAT実験call資料は未提供です。kNN ADは検証していますが、hCLAT.scoreから代用して複合ADを再現したとは報告しません。必要なファイル名と参照箇所は `notebook_audit.md`。
- 原本のグローバル特徴量選択と内部CV前の補完・標準化による情報の共有は維持しています。性能はOOF・学習・外部で区別し、リークのない改善版に黙って置換しません。
- Codespaces設定・Python版・起動コマンドを整備しました。GitHub上で新しいCodespaceを作る操作やCommunity Cloudへのデプロイは未実施です。
- コードはこのチェックアウトを変更しています。GitHubへのpushやPR作成は行っていません。
- 環境設定の保存と、ユーザー操作による環境Publish・新規タスクへの復元は別です。復元後の実行はまだ検証していません。
