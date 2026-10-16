# 元 Notebook の計算仕様と再現範囲

参照元は `★XGBoost_model_refact_tokunaga.ipynb`（286セル）。以下のセル番号は **0始まりの JSON cells 配列の位置**。元の Notebook / DASS / newCE は変更していない。添付 `train.py` と `original_runtime.py` は未完成の Web 化途中コードとして比較した。

## 保存された実行の状態

- Python メタデータは 3.12.4。セル0の保存出力は pandas 3.0.3。それ以外の学習ライブラリのバージョン固定はない。
- 実行記録はセル0–35まで。セル24は `perc=[100,80]`, CV `[5,10]`, 分割 seed `0..1999` を、**Optunaなし / 150木**で比較した履歴。セル25にその R² 辞書が残る。
- 最後の実行セル35は `NameError: print_score_perc100 is not defined` で停止した。関数定義セル34が raw セルになっているため、通常の Run All では定義されない。
- セル41以降の「最適モデル」、予測 CSV、選択特徴量一覧、最適パラメータ、AD、SHAP の保存出力はない。従って過去の1000-trial学習結果と数値が一致したという証拠にはできない。
- セル41–42を、明示された本番再現条件の定義として扱う。末尾セル282は特徴量重要度の探索用で tuning=False に戻す別モデルであり、本番モデルの上書きに使用しない。

## 入力と目的変数（セル5–13、15）

1. 2つの CSV は CP932 で読む。DASS は154行459列、newCE は74行459列。
2. DASS の最終列 `LLNA_EC3` が目的変数。先頭6列（No., Name, CAS Number, SMILES, CAS_Number_qsar, SMILES_qsar）を除き、最終列の手前までの **452列すべてを元の順序で説明変数**にする。`MolWeight` を除かない。
3. newCE だけは `dropna(subset=['MolWeight']).reset_index(drop=True)` を先に実行し、72行になる。学習用と混合しない。予測時は DASS の列名・順序にそろえる。
4. CSV の記述子には既に対数変換等が施されている。Notebook は X に追加の対数変換をしない。
5. `y = np.log10(MolWeight / EC3)`、逆変換は `EC3 = MolWeight / (10 ** y)`。倍率100/1000等の追加、150でのクリップはない。非感作性を150とする変換はCSVで既に実施済み。
6. 層別ラベルは **逆変換した EC3** に対し、`==150 -> 0`, `<=2 -> 2`, その他 `1`。`==150` は厳密な浮動小数点比較。元のEC3から直接ラベルを作ると、現在のNumPyではDASSの0始まり108行目が異なる（生値クラス数 `[38,83,33]`、逆変換後 `[37,84,33]`）。再現モードでは丸めや isclose に変えない。
7. XGBoostの予測 `y` は float32。元式の `10 ** y` はその dtype を保つため、先にfloat64へ変換すると逆変換値がわずかに変わる。比較時はこの演算順序も合わせる。

CSV SHA256:

- DASS: `1d28f0cf2cb4260a95b38de2cdb94feec6676b91fce8e1270376091193398b1d`
- newCE: `f04eed95af213db8d4bd5c25e5e9c99f98966621b3b0bdf5ff41b9496ba17042`

## 前処理とBoruta（セル15、41–42）

`Pipeline([('inputer', KNNImputer(n_neighbors=5, weights='uniform')), ('standarization', StandardScaler())])`。

Boruta は **全DASS** にこのパイプラインを fit/transform した結果と全yを与える。`RandomForestRegressor(n_jobs=-1,max_depth=5)`、`BorutaPy(n_estimators='auto',verbose=0,random_state=0,perc=100)`。その他は使用するBorutaバージョンの既定値（一般に max_iter=100, alpha=.05, two_step=True）。`support_` のみを採用し、暫定 `support_weak_` は含めない。元の452列順でマスクを適用する。

その後、外側Foldごとに452列の前処理パイプラインを **そのFoldの学習行だけで再fit** し、変換後に同一のグローバルBorutaマスクを適用する。外部検証はその学習パイプラインで transform する。元コードは外部用に同じ学習行でdeepcopyパイプラインを再fitしているが、決定的処理なので保存した学習パイプラインを使う場合と同じ。

選択されなかった列も KNNImputer の近傍距離に使われる。選択列だけ入力し、その他をすべて NaN で補う方式は完全な入力に対するNotebook再現と異なる。手入力で欠損を許す場合はその入力条件を利用者に明示し、前処理順は維持する。

## CV、XGBoost、Optuna（セル15、41–42）

外側: `StratifiedKFold(n_splits=5,shuffle=True,random_state=1037)`。XGBoost自体の seed は分割seedとは別に0。

基本パラメータ:

```python
max_depth=6, learning_rate=.03, n_estimators=100000, verbosity=1,
booster='gbtree', n_jobs=1, gamma=0, min_child_weight=1,
max_delta_step=0, subsample=1, colsample_bytree=1,
colsample_bylevel=1, colsample_bynode=1, reg_alpha=.5,
reg_lambda=.1, scale_pos_weight=1, base_score=None,
random_state=0, missing=float('nan'), objective='reg:squarederror',
eval_metric='rmse', early_stopping_rounds=15
```

各外側Foldで独立に `TPESampler(seed=0)`, direction='maximize', **1000 trials**。内部は `StratifiedKFold(5,shuffle=True,random_state=0)`。内部検証を early stopping 用 eval_set にし、各内部Foldの y空間R² の算術平均を最大化する。early stopping 指標はRMSE。内部で前処理を再fitしない。

| 探索パラメータ | 範囲 |
| --- | --- |
| max_depth | int 3–9 |
| gamma | log uniform 1e-8–1 |
| min_child_weight | int 2–10 |
| subsample | uniform .2–1 |
| colsample_bytree | uniform .2–1 |
| reg_alpha, reg_lambda | log uniform 1e-8–1 |

最良trialで5内部Foldの `best_iteration` を記録し、**その最大値をそのまま `n_estimators` に設定する（0始まりへの +1 はしない）**。early_stopping_rounds を削除し、外側Foldの全学習行で最終fitする。Optuna無効の別枝は150木。これは本番再現モードではない。軽量検証でtrial数・木数を制限した場合は、その条件を記録し1000-trial検証と区別する。

## 予測・集約と評価（セル15、65–85、120、224/229、265/270）

- DASS評価は各行を一度だけ外側holdoutとして予測したOOF。学習行自身への予測や外部検証とは区別する。Fold別R²/RMSEと全OOF R²/RMSEを計算する。
- 外部検証・新規予測は5モデルそれぞれで予測。後半の出力セル229/270の代表値は **rowwise max(y), min(EC3)**。MolWeightは行内で共通なので同じFoldに対応する。AD考慮値はAD内Foldだけを候補にして同じ集約。すべてAD外ならAD考慮値はNaN。
- セル76/77には平均値比較もあるが、yの平均とEC3の平均は別々に算出され、互いの逆変換ではない。平均値を最終出力のmin値と置き換えない。
- 出力 `Fold_misprediction` は予測EC3 / 実測EC3（対称化しない）。セル139の距離検討だけは対称比を使う別解析。
- GHS分類: <=2 は1A、2<EC3<=100 は1B、>100 はNC。ECETOC: <.1, [.1,1), [1,10), [10,100), >=100。混同行列は行=予測、列=実測。5倍範囲は strict `true/5 < pred < true*5`。
- アプリでMAEを追加する場合は追加指標とし、単位を y空間 / EC3(%) で分ける。セル149の `mae_diff` は単なる符号付き残差で、MAEではない。

## k近傍AD（セル15、42/45、171、235）

選択・標準化後の特徴量空間、Euclidean `scipy.spatial.distance.cdist`。k=5。Foldの学習点同士の距離を各行でソートし、自己距離を除く1:k+1列の平均を求め、それをソート。閾値は

```python
sorted_training_means[round(n_train * (1 - AD_coverage)) - 1]
```

query は全学習点との距離の小さい順0:k列の平均。`mean_distance <= threshold` がAD内。補間する `quantile` に置換しない。`AD_coverage` は **AD外にする学習割合** であり、.1 は概ね90%包含を意味する。元コードは0、.1、.2を順に検討し、最後の出力枝は.2。設定を記録して比較する（.1と.2を同一視しない）。全体ADは5Foldの少なくとも1つがAD内か。

## in vitroの追加ADに必要な未提供データ（セル90、94、211）

DPRA.score>0、KS.Call（NaNを0にする）に加え、hCLAT.Call を合計して2以上を別のin vitro領域とする。しかし hCLAT.Call はアップロードCSVにはなく、次の外部ファイルから別途読む:

- DASS: `Annex-2-in-vitro-in-silico-in-vivo-defined-approaches.csv`、DASS No. と Sort を対応付け `hCLAT.Call` を取得。
- newCE: `newCE_data.csv`、CAS Number と CAS no. を対応付け `Pos/ Neg` を取得。

これらは提供されていない。`hCLAT.score>0` のような推定規則に勝手に変えない。kNN ADは単独で計算可能。追加ADは未取得と表示し、明示的な実験callがある場合だけ計算する。

## SHAP（セル15、53–56、273–274）

Fold別 `shap.TreeExplainer(model)`。summaryはそのFoldの **学習データ** の選択・標準化後Xに対する `shap_values` を使った violin plot。waterfallは同じFoldモデルの `explainer(X)` による個別値。寄与は y=log10(MW/EC3) 空間であり、EC3(%)に対する加算寄与ではない。holdoutや新規入力のSHAPは追加機能としてFoldと対象データを明示する。

Notebook内の横断的な重要度集計は `abs(sum(shap_values))`（符号を合計してから絶対値）で、通常の `mean(abs(shap_values))` とは異なる。また特徴名の文字列列を含むDataFrameで mean を呼んでおり pandas 3系ではエラーになる可能性がある。平均SHAPや平均予測のSHAPを独自に作らず、Foldごとの値と図を優先する。最終 min(EC3) が選択したモデルと、そのSHAPのFoldを対応させる。

## リークと再現性の限界

- グローバルBorutaに外側testのyを使うため、OOFが完全な未知データ評価にはならない。これは再現モードで保持する元実装の性質。
- 内部CVの前に外側学習データ全体で補完・標準化するため、内部validation由来の分布情報も使う。
- 2000分割seedの性能から分割seedを選ぶ探索、AD閾値の事後探索、10000回の外部部分抽出からR²最高の集合だけ示すセル208–218は評価を楽観的にし得る。アプリの標準性能は全OOF / 全newCEを表示し、最高抽出結果を外部検証全体と呼ばない。
- xgboost の既定tree_method/base_score、Boruta/RFの既定値、NumPyの浮動小数点による層別ラベル、ライブラリバージョン差に依存する。過去の最良trial履歴・選択mask・モデルがないため、元PCでの完全一致は保証できない。
- 本アプリのバージョン固定・データhash・Foldインデックス・前処理・選択mask・trial履歴・モデルをまとめて保存することで、今後の同一環境再実行を検証可能にする。

## 全セルの依存関係・後半解析の扱い

全286セルのsourceとセル種別を走査し、code/rawはASTでも解析した。主要な依存関係は入力5–13 → class15 → 設定41 → 学習42 → 新規予測46 / 評価65–85 / AD171または235 → 出力224/229または265/270。

| セル範囲 | 役割・依存関係 |
| --- | --- |
| 0–13 | import、ローカル絶対パス、2データ読込とX/y抽出 |
| 14–16 | class15に変換、補完、Boruta、二重CV、AD、SHAPを定義 |
| 17–39 | 2000seed探索、pickleのrawセル、集計関数のrawセル、セル35で実際に停止 |
| 40–64 | 本番設定・学習、Foldデータ出力、SHAP、選択特徴量とパラメータ確認 |
| 65–87 | 回帰評価、5倍範囲、GHS/ECETOC、平均と最強予測の比較 |
| 88–124 | 未提供hCLAT元データを使うin vitro ADとkNN ADの組み合わせ |
| 125–136 | newCEから38物質（label0/1/2を9/21/8）をseed別に非復元抽出、10000回評価 |
| 137–168 | coverage0から.989を.001刻みで走査、heatmap、残差、距離分布（図の4.43/3.63は固定描画値でAD閾値式ではない） |
| 169–206 | coverage .1でAD再計算、in/out別評価、発表図、10000回外部部分抽出結果 |
| 207–218 | 上記10000集合からR²が最高の集合だけ取り出す発表用解析 |
| 219–232 | coverage .1のDASS OOF/newCE報告表、CSV保存はrawセル |
| 233–271 | coverage .2で同様の解析を繰り返す（多数の同一定義）、末尾報告表はmin EC3 |
| 272–285 | SHAP再表示、R²に関する探索と注記、tuningFalse別モデルによる重要度確認 |

純粋な計算の再利用と、研究途中の発表用・試行錯誤セルを区別する。10,000回抽出・2000seed探索は画面起動時に実行しない。

実行を妨げる既存Notebook上の問題も残したまま参照する:

- セル3は元PCのMac絶対パス、90/94/211は別PCのWindows絶対パス。
- セル48はFold iのtrain値にFold1の識別情報を結合し、testもFold1固定。行IDの不一致をそのまま採用しない。
- セル124は定義した `inside_ad_outer_pred_max` の代わりに別文脈の `inside_ad_outer_pred_mean` を参照する。
- セル212/214/215の発表図は `log10(EC3).max()` を使う別方式で、ADマスクも各Foldに適用せず、後半CSV出力のmin EC3とは不一致。CSV出力枝を主仕様とし、これらの図を数値照合の基準にしない。
- セル231/271は未定義の `DASS_output_dir` を参照。セル274はclassに存在しない `PRINT_quantile_SHAP_waterfall` を呼ぶ。
- セル278の `newpred_outAD2_ind02plus` は未定義。セル282は必須引数 Name_df を省略。
- セル279は真値を raw EC3 で取り、予測を y空間で比較してしまう。元Notebookの末尾にある通り、部分集合R²の比較は真値分散に強く依存する。これらはモデル自体の変換式を変更する根拠にはしない。

## 添付の途中実装との差分

添付train.pyは utils_io/utils_ml がないとimportできず、`--parameter-tuning` を指定しない既定実行では150木、指定しても既定は100 trials。層別ラベルを生EC3から作成し、MolWeight欠損newCEの除外を呼び出し側で明示していない。固定特徴量リストは元Notebookに保存された選択結果の証拠がないため、再現modeのBorutaの代用にしない。

添付runtimeは選択特徴量しか要求せず、その他をNaNで埋めるため、完全CSVと同じ補完結果を保証しない。最小EC3を出したFoldを選ぶ方式は後半CSV出力仕様に対応する。モデル・全452列の前処理・mask・列順・Foldインデックス・AD参照学習値を一緒に保存・再利用する必要がある。

現在環境での実際の一致検証結果は `outputs/notebook_comparison.json` に記録する。軽量検証の成功だけで、未実行の1000-trial本番学習または過去PCとの一致を実施済みとはしない。

## 独立した照合スクリプト

`scripts/verify_notebook.py` はアプリの学習式を参照値として使わず、信頼する元Notebookのclass定義セル15を抽出して実行する。個人PCパス・CSV保存・2000seed探索・10000回の部分抽出・発表用図は実行しない。通常の軽量検証では、元classの trial数1000と最大木数100000という2箇所だけを2と200に制限し、学習の他の式は変更しない。

軽量検証（既存の軽量artifactを照合、再学習は元class側だけ）:

```bash
python scripts/verify_notebook.py \
  --use-existing-artifact --artifact-path artifacts/smoke_model_artifact.joblib
```

本番条件を現在の環境で独立に再計算して照合:

```bash
python scripts/verify_notebook.py \
  --n-trials 1000 --max-estimators 100000 \
  --use-existing-artifact --artifact-path artifacts/notebook_model_artifact.joblib \
  --output outputs/notebook_comparison_full.json
```

照合では許容差をすべて0とし、入力452列・順序・newCEフィルタ、目的変数とラベル、グローバル補完標準化、Boruta mask、全Foldの行インデックス・XGBoost条件・前処理・y/EC3予測、AD距離閾値と判定、保存OOF、最終集約、Fold別学習SHAP/base値、最終FoldのOptuna trial値・best_iterationを比較する。履歴上の元PCモデルとの照合と、同じ環境で元classを実行した結果との照合は別である。

保存済みモデルから0/.1/.2のAD枝だけを比較する場合（モデル学習なし）:

```bash
python scripts/verify_notebook.py --ad-coverage-only \
  --artifact-path artifacts/notebook_model_artifact.joblib \
  --output outputs/notebook_ad_coverage_comparison.json
```

この比較は元classのAD関数に保存された学習・holdout・外部検証の特徴量行列を与え、閾値、全判定、AD内Foldだけでの集約を照合する。`AD_coverage=.1` は選択した既定の枝、`.2` はNotebook後半の別の報告枝である。

軽量条件の実測結果: 2 trials × 5外側Fold、内部5-fold、最大200 roundsを元classとアプリでそれぞれ計算し、112項目を比較して全項目が許容差0で一致した（`outputs/notebook_comparison.json`）。Borutaは軽量化せず通常の100反復設定を使い、現在環境で21特徴量が選択された。この結果だけを本番1000-trial一致の証拠にはしていない。

本番学習済みartifactに対するAD枝の実測結果: coverage 0 / .1 / .2の54項目がすべて許容差0で一致した（`outputs/notebook_ad_coverage_comparison.json`、学習を実行しない照合）。このファイルには元classの関数を用いたADだけの比較範囲と、使用したartifactのSHA256を記録している。

本番条件の独立照合も実施済み: 元classで **5外側Fold × 1000 trials × 内部5-fold、最大100000 rounds** を実際に再計算し、保存した本番artifactと112項目を比較して **全項目が許容差0・最大絶対差0で一致**した（`outputs/notebook_comparison_full.json`、816.448秒）。通常学習と元class再計算は別々に行った。未提供の過去PC上の最終モデル・予測値との比較には当たらず、照合範囲は **現在の固定ライブラリ環境における元Notebookの実行可能なclass定義**である。

照合した本番artifactのSHA256: `0d1dc76ecb32cb3d6b3d00974a51bcf78161d7be323917d38179e679052bb4ad`。このcanonical artifactは比較後も変更せず、アプリの既定モデルへ検証情報を付ける場合は別のコピーにメタデータを追記する。
