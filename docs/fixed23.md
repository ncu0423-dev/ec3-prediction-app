# 指定23項目モデル

利用者が指定した次の23項目を、その順序のまま入力・学習に使います。
MolWeightも説明変数に含み、さらにEC3の逆変換にも使用します。

1. `OVERALL OH rate constant`
2. `FM advection air`
3. `VdW surface DPSA2`
4. `LUMO Energy`
5. `HOMO Energy`
6. `MolWeight`
7. `CombDipolPolariz`
8. `LogHL_pred`
9. `LogKM_pred`
10. `alert_Protein_binding_by_OASIS`
11. `potency_Protein_binding_alerts_for_skin_sensitization_according_to_GHS`
12. `DPRA.percCysdep`
13. `DPRA.percLysdep`
14. `DPRA.score`
15. `hCLAT.CD86.EC150..ug.ml.`
16. `hCLAT.CD54.EC200..ug.ml.`
17. `h.CLAT.CV75`
18. `hCLAT.MIT`
19. `hCLAT.score`
20. `KS.EC1.5`
21. `KS.EC3`
22. `KS.IC50`
23. `sum_invitro`

## 旧Notebook再現版との違い

| 処理 | 現在の23項目版 | 旧Notebook再現版 |
| --- | --- | --- |
| 前処理へ渡す列 | 指定23項目 | 元データの452項目 |
| KNN補完・標準化 | 23項目でFoldごとにfit | 452項目でFoldごとにfit |
| Boruta | 実行せず23項目すべて固定 | 全学習データで選択、現行環境では21項目 |
| モデル・AD・SHAPの次元 | 23 | Boruta選択後の21 |
| 項目の削除・追加 | なし | Borutaのsupport_による |

23項目版は表示だけを絞ったモデルではなく、前処理から再学習した別モデルです。
元Notebookとの112項目完全一致は旧452項目版の検証結果であり、23項目版には適用しません。
元NotebookとCSVの内容、比較済みの旧モデルは変更していません。

## 維持した学習条件

- random_state=1037、外側5-fold、内側5-foldはrandom_state=0。
- 本番は各FoldでOptuna 1000 trials、上限100000木、early stopping15。
- 探索範囲・y空間R²の目的関数・最良木数の決定規則は従来と同じ。
- y=log10(MolWeight / EC3)、EC3=MolWeight / 10**y。
- Foldの最小EC3を最終予測とし、yも同じFoldから取得。
- kNN ADはk=5、coverage=.1（AD外割合）。AD内Foldだけの最小EC3も別出力。
- 外部72件を学習へ混合しない。未知のhCLAT.Callは推測しない。

固定23項目版ではグローバルBorutaによる特徴量選択を行いません。
内側CVの前に外側学習部分で前処理する既存手順は維持しており、
別の完全なリーク対策方式へ変更したという意味ではありません。

## 入力・表示

必要な列は23項目だけです。列順が異なるCSVはモデルの順に整列します。
MolWeightは有限かつ正の数が必須、その他の欠損セルは学習済みKNNで補完します。
欠けた列を黙って追加したり、実測LLNA_EC3を予測入力に要求したりしません。
化学物質名・CAS番号などの任意列は、利用者がアップロードした予測結果に保持します。
ホームのモデル名は「指定23項目モデル」、入力特徴量数は23です。

## 学習と配備

```bash
python train.py --mode notebook --feature-mode fixed23 --fold-workers 4 \
  --artifact-path artifacts/fixed23_model_artifact.joblib \
  --output-dir outputs/fixed23_notebook
python scripts/export_deployment_model.py \
  --source artifacts/fixed23_model_artifact.joblib \
  --output artifacts/original_model_artifact.joblib
```

`--mode` は探索予算、`--feature-mode` は使用する説明変数方式です。
`notebook`という予算名だけでNotebook原本との計算一致を主張するものではありません。
軽量検証は `--mode smoke --feature-mode fixed23` と明示します。

Git管理する既定ファイルは後方互換のため同じ名前を使いますが、内容は23項目版です。
配備版は評価表の物質名・CAS・SMILES等を除き、必要な数値参照データを保持します。
旧版を再現するときは `--feature-mode boruta` と別の保存先を明示してください。

## 検証結果

23項目の列名・順序、指定外列なしの予測、欠損セル、Borutaを呼び出さない学習、
各Foldの前処理次元、テンプレート順、表示名を追加テストで確認しました。
軽量23項目版を対象に71テストが成功しています。

本番モデルは各Fold 1000 trials（計5000）、4プロセスで完了しました。
本環境は4 CPU / 32 GiBです。各XGBoostとBLASは1スレッドに制限しています。
並列化のため逐次の途中試行を中断して再実行し、計算条件は変えていません。
軽量12 trialsで逐次・並列のOOF・SHAP・試行履歴が完全一致し、
先に逐次実行した本番Fold 1の1000試行のスコア・パラメータも並列版と完全一致しました。

| EC3評価対象 | 件数 | R² | RMSE (%) | MAE (%) |
| --- | ---: | ---: | ---: | ---: |
| OOF | 154 | 0.176512 | 55.386709 | 33.122905 |
| 外部（最小EC3） | 72 | -0.827758 | 96.901433 | 70.245772 |
| 外部（AD内Foldのみ） | 54 | -0.876857 | 97.344444 | 70.848647 |

外部性能は十分とはいえません。説明変数を固定するという要件を実装した結果であり、
旧版からの予測精度改善や、旧Notebookの予測値との一致を意味しません。
詳細なモデルハッシュ・条件・数値は `fixed23_validation.json` に記録しています。

配備用23項目本番モデルを対象に、73テストが成功（失敗0・スキップ0、19.61秒）しました。
モデルの保存・読み込み、全5画面、単一・CSV予測、欠損値、AD、SHAPを含みます。
