# 現在のCSV入力（24列・21説明変数）

CSVは次の順序に固定したテンプレートを使います。No.・Name・CAS Numberは識別情報で、
モデルにはその後の21項目だけを渡します。SMILES・VdW surface DPSA2は含めません。

1. `No.`
2. `Name`
3. `CAS Number`
4. `OVERALL OH rate constant`
5. `FM advection sediment`
6. `LUMO Energy`
7. `HOMO Energy`
8. `CombDipolPolariz`
9. `LogHL_pred`
10. `LogKM_pred`
11. `alert_Protein_binding_by_OASIS`
12. `judge_Protein_binding_alerts_for_skin_sensitization_according_to_GHS`
13. `potency_Protein_binding_alerts_for_skin_sensitization_according_to_GHS`
14. `DPRA.percCysdep`
15. `DPRA.percLysdep`
16. `DPRA.score`
17. `hCLAT.CD86.EC150..ug.ml.`
18. `h.CLAT.CV75`
19. `hCLAT.MIT`
20. `hCLAT.score`
21. `KS.EC1.5`
22. `KS.EC3`
23. `KS.IC50`
24. `sum_invitro`

[空のCSVテンプレート](../templates/csv21_input_template.csv) またはアプリのダウンロードボタンから取得できます。
ヘッダー24列だけの空ファイルなので、データ行を記入してからアップロードしてください。
列順が異なっても列名でこの順に揃えます。列の不足・重複・指定外列は明示して拒否します。
数値の欠損セルは許可しますが、非数値・無限大は許可しません。識別列の値や行順を保持します。

## 分子量（MolWeight）は画面で入力

CSVテンプレートにMolWeight列はありません。アップロード後に表示される表で、物質ごとに
有限で0より大きい分子量を入力します。表示した入力行番号を対応キーとし、
No.・Name・CASが重複しても別行の重みに混ざりません。編集表を並べ替えても行番号で対応します。
全行が有効になるまで予測ボタンは無効です。推測値・固定値・他行の値で穴埋めしません。
CSVや分子量を変更すると古い結果を隠し、別CSVに以前の分子量を流用しません。
既存データにMolWeight列がある場合だけ、任意の末尾列として読み込み、画面で確認・修正できます。

分子量を一切使わずEC3を出せるという意味ではありません。現在のターゲットは
`y=log10(MolWeight / EC3)` なので、最後の `EC3=MolWeight / 10**y` に分子量が必要です。
MolWeightは予測用説明変数・欠損補完距離・AD・SHAPには入りません。
分子量なしでEC3そのものを予測するにはターゲットを変えた別モデルの学習が必要で、今回の変更では行っていません。

## モデルと旧Notebook版の違い

21項目だけでKNN欠損補完・標準化・学習した、検証済みの本番モデルを再利用します。
各5 Fold1000 trialsの履歴、21列の補完器・XGBoost、モデルSHA、原本データSHA、
9ライブラリ版が過去の検証記録と一致したため、重みの再学習はしません。
サービス直前のAD外割合0.2は維持し、以前の固定21項目モデルの0.1から閾値・AD関連評価だけを再計算しました。

これは452項目の前処理後にBorutaで絞る元Notebook再現版とは異なります。
元Notebook版の112/166項目一致結果を今回の21項目版に流用しません。
元のNotebook・CSV・検証済み452項目モデル・固定21項目の検証元モデルは変更せず保持しています。

通常のアプリは `artifacts/original_model_artifact.joblib` の21項目配備版を読みます。
元の本番21項目モデルは `artifacts/fixed21_model_artifact.joblib`、AD等の再計算後は
`artifacts/csv21_model_artifact.joblib` です。次の操作は学習を行いません。

```bash
python scripts/prepare_csv21_model.py
python scripts/export_deployment_model.py \
  --source artifacts/csv21_model_artifact.joblib \
  --output artifacts/original_model_artifact.joblib
```

配備版は評価表の物質名・CAS・SMILES等を除き、必要な数値参照データを保持します。
追加in vitro ADに必要な元hCLAT資料は未提供のままで、推測して再現したことにはしません。

## 検証

配備用モデルを対象に **134 passed / 0 failed / 0 skipped、27.53秒**。
指定列順、文字コード、欠損、識別列、重複ID、分子量の入力・行対応・新CSV時のリセット、
予測・結果出力・AD・SHAP・全5画面を含みます。
単一・一括でAD候補なしの行も含む場合のFold名のnullable文字列型も統一し、
1件ずつとまとめて予測した結果の一致を確認しています。

外部72件のEC3 R²は約-0.680で、十分な汎化性能を示すものではありません。
条件・モデルハッシュ・数値は `csv21_validation.json` に記録しています。
