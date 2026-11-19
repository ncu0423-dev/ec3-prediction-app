# 指定21項目モデル（現在の配備版）

最新の指定に合わせ、次の21項目をその順序のまま説明変数に固定します。
Borutaによる追加・除外はしません。

1. `OVERALL OH rate constant`
2. `FM advection sediment`
3. `LUMO Energy`
4. `HOMO Energy`
5. `CombDipolPolariz`
6. `LogHL_pred`
7. `LogKM_pred`
8. `alert_Protein_binding_by_OASIS`
9. `judge_Protein_binding_alerts_for_skin_sensitization_according_to_GHS`
10. `potency_Protein_binding_alerts_for_skin_sensitization_according_to_GHS`
11. `DPRA.percCysdep`
12. `DPRA.percLysdep`
13. `DPRA.score`
14. `hCLAT.CD86.EC150..ug.ml.`
15. `h.CLAT.CV75`
16. `hCLAT.MIT`
17. `hCLAT.score`
18. `KS.EC1.5`
19. `KS.EC3`
20. `KS.IC50`
21. `sum_invitro`

## MolWeightは別の換算入力

MolWeightは上の21項目に含めません。KNN補完・標準化・XGBoost・AD・SHAPに渡す行列は21列です。
MolWeightは `y=log10(MolWeight / EC3)` の学習用目的変数の算出と、
`EC3=MolWeight / 10**y` の逆変換にだけ用います。各行で有限・正の値が必要です。

CSV入力は **説明変数21列＋換算専用MolWeight 1列（合計22列）** です。
テンプレートは21項目を上記順に並べ、最後にMolWeightを追加します。
物質名・CAS番号などの識別情報は任意で追加でき、目的変数LLNA_EC3は予測に不要です。

同じ21項目のままMolWeightだけを変えても、補完結果、各Foldの予測y、AD、SHAPは変わりません。
EC3への逆変換値だけがMolWeightに比例して変わります。欠損セルを含む入力でもこの性質をテストしています。

説明変数の欠損セルは学習済みKNNImputerで補完します。21項目がすべて空欄の場合は
近傍を決める情報がなく、学習データの項目平均へフォールバックするため画面で通知します。
MolWeightは補完距離へ入れません。列そのものが欠けている場合はエラーです。

## 以前の23項目版との差分

- `FM advection air` を外し、`FM advection sediment` を使用。
- `VdW surface DPSA2` と `hCLAT.CD54.EC200..ug.ml.` を除外。
- `judge_Protein_binding_alerts_for_skin_sensitization_according_to_GHS` を追加。
- MolWeightを説明変数から除き、換算用に分離。

画面だけでなく、前処理から21項目で再学習しています。以前と予測値は変わります。
旧23項目版は `fixed23.md`、旧452項目/Boruta版の一致検証は `notebook_audit.md` に保持します。
それらの過去の性能指標や112項目一致結果は、この21項目版へ流用しません。

## 学習条件と配備

```bash
python train.py --mode notebook --feature-mode fixed21 --fold-workers 4 \
  --artifact-path artifacts/fixed21_model_artifact.joblib \
  --output-dir outputs/fixed21_notebook
python scripts/export_deployment_model.py \
  --source artifacts/fixed21_model_artifact.joblib \
  --output artifacts/original_model_artifact.joblib
```

外側5-fold（random_state=1037）、内側5-fold（random_state=0）、
各Fold 1000 Optuna trials、上限100000木、early stopping15で実行しました。
探索範囲・y空間R²の目的関数・最良木数の決定規則、EC3最小Foldの採用、k=5/coverage=.1のAD規則は維持しています。
4 CPU / 32 GiBの環境で、独立したFoldを4プロセス、各XGBoost/BLASは1スレッドで学習しました。

`--mode notebook` は探索予算の名前で、Notebook原本と同じ説明変数という意味ではありません。
CLIの既定のfeature-modeはfixed21です。過去の23項目版はfixed23、元Notebookの方式はborutaと明示します。
外部72件を学習へ混合しません。内側CV前の前処理など従来の手順は維持しており、完全なリーク対策への変更ではありません。
未提供のhCLAT.Callを推測して追加ADを計算することもしません。

配備用モデルは評価表の物質名・CAS・SMILES等を除き、必要な数値参照データを保持します。
旧モデル・原本Notebook・原本CSVは変更せず保持しています。

## 本番モデルの評価

| EC3評価対象 | 件数 | R² | RMSE (%) | MAE (%) |
| --- | ---: | ---: | ---: | ---: |
| OOF | 154 | 0.269442 | 52.167996 | 30.589762 |
| 外部（最小EC3） | 72 | -0.680034 | 92.903021 | 67.233221 |
| 外部（AD内Foldのみ） | 53 | -0.636003 | 90.751994 | 66.376270 |

入力項目の要件を反映したモデルであり、予測精度の改善を保証するものではありません。
条件・数値・モデルハッシュは `fixed21_validation.json` に記録しています。

配備用本番モデルを対象に83テストが成功（失敗0・スキップ0、22.45秒）しました。
21項目・列順・MolWeightの分離、欠損値、再学習、保存・読込、単一・CSV予測、AD、SHAP、全5画面を含みます。
