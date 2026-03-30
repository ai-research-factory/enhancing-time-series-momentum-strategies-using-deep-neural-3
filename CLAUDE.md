# Enhancing Time-Series Momentum Strategies Using Deep Neural Networks

## Project ID
proj_40e845c7

## Taxonomy
StatArb, Other

## Current Cycle
4

## Objective
Implement, validate, and iteratively improve the paper's approach with production-quality standards.


## Design Brief
### Problem
Traditional time-series momentum strategies often rely on heuristic rules for trend estimation and position sizing, which may not be optimal. This paper addresses this limitation by proposing 'deep momentum networks,' a hybrid model that utilizes a Long Short-Term Memory (LSTM) network. The model is designed to learn both the trend estimation and the optimal position sizing simultaneously from historical price data.
The core innovation lies in the training process, where the network is trained to directly optimize the portfolio's Sharpe ratio. This end-to-end, data-driven approach allows the model to learn a trading rule that is inherently risk-aware. Furthermore, the paper introduces a turnover regularization term into the loss function, enabling the model to account for transaction costs during training and produce more realistic and cost-effective trading strategies.

### Datasets
A diverse set of ~20-30 liquid ETFs from yfinance, covering major asset classes. Example tickers: SPY, QQQ, IWM, EFA, EEM, TLT, IEF, HYG, GLD, SLV, USO, UNG, DBC.

### Targets
The primary optimization target is the out-of-sample portfolio Sharpe ratio. The model's direct output is the position size for each asset at each time step.

### Model
The model is a 'Deep Momentum Network' based on an LSTM architecture. It takes a sequence of past returns for multiple assets as input and outputs a continuous value for each asset, representing the desired position size (ranging from -1 for full short to +1 for full long). The network is trained end-to-end using a custom loss function, which is the negative of the portfolio's Sharpe ratio, calculated over the training batch. A turnover regularization term is added to this loss to penalize frequent changes in positions, thereby controlling transaction costs.

### Training
The model is trained and evaluated using a walk-forward validation scheme to prevent look-ahead bias and test for robustness over time. The dataset is split into multiple (e.g., 5-10) chronological folds. In each fold, the model is trained on a rolling or expanding window of past data and tested on the immediately following out-of-sample period. The final performance is the aggregated result from all out-of-sample periods.

### Evaluation
The primary evaluation metric is the out-of-sample Sharpe ratio. This will be compared against a traditional time-series momentum baseline (e.g., moving average crossover with fixed position sizing). Other key metrics include Annualized Return, Volatility, Maximum Drawdown, and Portfolio Turnover. Performance will be reported both gross and net of transaction costs.


## データ取得方法（共通データ基盤）

**合成データの自作は禁止。以下のARF Data APIからデータを取得すること。**

### ARF Data API
```bash
# OHLCV取得 (CSV形式)
curl -o data/aapl_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y"
curl -o data/btc_1h.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=BTC/USDT&interval=1h&period=1y"
curl -o data/nikkei_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=^N225&interval=1d&period=10y"

# JSON形式
curl "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y&format=json"

# 利用可能なティッカー一覧
curl "https://ai.1s.xyz/api/data/tickers"
```

### Pythonからの利用
```python
import pandas as pd
API = "https://ai.1s.xyz/api/data/ohlcv"
df = pd.read_csv(f"{API}?ticker=AAPL&interval=1d&period=5y")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")
```

### ルール
- **リポジトリにデータファイルをcommitしない** (.gitignoreに追加)
- 初回取得はAPI経由、以後はローカルキャッシュを使う
- data/ディレクトリは.gitignoreに含めること



## Preflight チェック（実装開始前に必ず実施）

**Phase の実装コードを書く前に**、以下のチェックを実施し結果を `reports/cycle_4/preflight.md` に保存すること。

### 1. データ境界表
以下の表を埋めて、未来データ混入がないことを確認:

```markdown
| 項目 | 値 |
|---|---|
| データ取得終了日 | YYYY-MM-DD (今日以前であること) |
| Train期間 | YYYY-MM-DD 〜 YYYY-MM-DD |
| Validation期間 | YYYY-MM-DD 〜 YYYY-MM-DD |
| Test期間 | YYYY-MM-DD 〜 YYYY-MM-DD |
| 重複なし確認 | Yes / No |
| 未来日付なし確認 | Yes / No |
```

### 2. Feature timestamp 契約
- 全ての特徴量は時刻 t の予測に t-1 以前のデータのみを使用しているか？ → Yes / No
- Scaler / Imputer は train データのみで fit しているか？ → Yes / No
- Centered rolling window を使用していないか？ → Yes / No (使用していたら修正)

### 3. Paper spec 差分表
論文の主要パラメータと現在の実装を比較:

```markdown
| パラメータ | 論文の値 | 現在の実装 | 一致? |
|---|---|---|---|
| ユニバース | (論文の記述) | (実装の値) | Yes/No |
| ルックバック期間 | (論文の記述) | (実装の値) | Yes/No |
| リバランス頻度 | (論文の記述) | (実装の値) | Yes/No |
| 特徴量 | (論文の記述) | (実装の値) | Yes/No |
| コストモデル | (論文の記述) | (実装の値) | Yes/No |
```

**preflight.md が作成されるまで、Phase の実装コードに進まないこと。**

## ★ 今回のタスク (Cycle 4)


### Phase 4: 売買回転率正則化とコストモデルの実装 [Track ]

**Track**:  (A=論文再現 / B=近傍改善 / C=独自探索)
**ゴール**: 損失関数に売買回転率ペナルティ項を追加し、バックテストに取引コストモデルを組み込む。

**具体的な作業指示**:
1. `src/loss.py`の`sharpe_loss`を修正し、`turnover_regularization`項を追加します。損失は `sharpe_loss + gamma * turnover` となり、`gamma`はハイパーパラメータです。`turnover`はバッチ内のポジションの変化の絶対値の合計として計算します。
2. `src/backtest.py`の`Backtester`クラスを修正し、リバランス時の取引コスト（例: 5bps）を考慮して純収益（Net Returns）を計算するロジックを追加します。
3. `src/main.py`に`run-cost-analysis`コマンドを追加し、正則化あり/なしで学習したモデルのバックテストを実行し、それぞれの総収益（Gross）と純収益（Net）のパフォーマンス、および売買回転率を`reports/cycle_4/cost_analysis.json`に保存します。

**期待される出力ファイル**:
- src/loss.py
- src/backtest.py
- reports/cycle_4/cost_analysis.json

**受入基準 (これを全て満たすまで完了としない)**:
- `cost_analysis.json`にGrossとNetのSharpe比が記録されている。
- 正則化を有効にしたモデルは、無効にしたモデルよりも売買回転率が低くなる。




## データ問題でスタックした場合の脱出ルール

レビューで3サイクル連続「データ関連の問題」が指摘されている場合:
1. **データの完全性を追求しすぎない** — 利用可能なデータでモデル実装に進む
2. **合成データでのプロトタイプを許可** — 実データが不足する部分は合成データで代替し、モデルの基本動作を確認
3. **データの制約を open_questions.md に記録して先に進む**
4. 目標は「論文の手法が動くこと」であり、「論文と同じデータを揃えること」ではない


## スコア推移
Cycle 1: 45% → Cycle 2: 55% → Cycle 3: 45%
改善速度: 0.0%/cycle ⚠ 停滞気味 — アプローチの転換を検討





## レビューからのフィードバック
### レビュー改善指示
1. [object Object]
2. [object Object]
3. [object Object]
### マネージャー指示 (次のアクション)
1. 【REPLAN: 評価設計修正】
primaryBlocker: 頑健な評価フレームワークの欠如

以下を最優先で実施:
1. Walk-forward validationの実装を確認(train/test境界の厳密分離)
2. フォワードルッキングがないことをテストで証明
3. metrics.jsonのwalkForward.windowsが5以上であることを確認

完了条件: Walk-forward法による評価が実装され、`reports/cycle_4/metrics.json` に `n_walk_forward_windows` >= 5 というキーと値が記録されること。また、各foldのSharpe Ratioをリストとして記録する `fold_sharpe_ratios` が追加されること。
2. 【最優先】`src/evaluation.py` に `WalkForwardValidator` クラスを実装する。scikit-learnの `TimeSeriesSplit` を参考に、学習/検証期間の分割、およびオプションで期間間のギャップを設定できる機能を持たせる。このバリデーターを `main.py` の評価ループに組み込み、単一の80/20分割を完全に置き換える。
3. 【重要】`src/metrics.py` の計算ロジックを修正する。`totalTrades` が0にも関わらず `dnn_turnover` が非ゼロになるバグを特定・修正する。また、`transactionCosts.netSharpe` が取引コストを正しく反映するよう、`sharpeRatio` との計算分離を明確にする。修正を検証するユニットテストを `tests/test_metrics.py` に追加する。
4. 【推奨】`src/reporting.py` を更新し、Walk-forward評価の各foldの結果（Sharpe, PnL, turnover等）を集計し、`reports/cycle_4/` に保存する機能を追加する。最終的なサマリーメトリクスとして、各foldのSharpeの平均値と標準偏差を `metrics.json` に `mean_fold_sharpe` と `std_fold_sharpe` として記録する。


## 全体Phase計画 (参考)

✓ Phase 1: コアモデルとSharpe損失関数の実装 — LSTMベースのDeep Momentum NetworkモデルとカスタムSharpe比損失関数を実装し、合成データで動作確認する。
✓ Phase 2: データパイプラインの構築 — yfinanceから金融時系列データを取得し、モデルの入力形式に前処理するパイプラインを構築する。
✓ Phase 3: 基本学習・バックテストループの実装 — 単一の学習・テスト分割でモデルを学習させ、ベースライン戦略と比較する基本的なバックテストを実行する。
→ Phase 4: 売買回転率正則化とコストモデルの実装 — 損失関数に売買回転率ペナルティ項を追加し、バックテストに取引コストモデルを組み込む。
  Phase 5: ウォークフォワード検証フレームワークの実装 — 厳密なアウトオブサンプル評価のため、ウォークフォワード法によるバックテストフレームワークを実装する。
  Phase 6: ハイパーパラメータ最適化 — Optunaを用いて、主要なハイパーパラメータの最適な組み合わせを探索する。
  Phase 7: ロバスト性検証と最終評価 — 最適化されたハイパーパラメータを用いて、より多くの分割数でウォークフォワード検証を実行し、結果のロバスト性を評価する。
  Phase 8: モデルのポジション分析 — DNNモデルが生成するポジションを可視化・分析し、その取引行動の特性を理解する。
  Phase 9: 代替モデルアーキテクチャの比較 — GRUや単純なFFNなど、よりシンプルなモデルアーキテクチャを実装し、LSTMの複雑さが必要かどうかを検証する。
  Phase 10: 最終レポートと可視化 — すべての実験結果を統合し、サマリー、比較表、エクイティカーブを含む包括的な最終レポートを生成する。
  Phase 11: コードのクリーンアップとドキュメント整備 — コードベースの品質を向上させ、第三者が実験を再現できるようにドキュメントとテストを整備する。


## ベースライン比較（必須）

戦略の評価には、以下のベースラインとの比較が**必須**。metrics.json の `customMetrics` にベースライン結果を含めること。

| ベースライン | 実装方法 | 意味 |
|---|---|---|
| **1/N (Equal Weight)** | 全資産に均等配分、月次リバランス | 最低限のベンチマーク |
| **Vol-Targeted 1/N** | 1/N にボラティリティターゲティング (σ_target=10%) を適用 | リスク調整後の公平な比較 |
| **Simple Momentum** | 12ヶ月リターン上位50%にロング | モメンタム系論文の場合の自然な比較対象 |

```python
# metrics.json に含めるベースライン比較
"customMetrics": {
  "baseline_1n_sharpe": 0.5,
  "baseline_1n_return": 0.05,
  "baseline_1n_drawdown": -0.15,
  "baseline_voltarget_sharpe": 0.6,
  "baseline_momentum_sharpe": 0.4,
  "strategy_vs_1n_sharpe_diff": 0.1,
  "strategy_vs_1n_return_diff": 0.02,
  "strategy_vs_1n_drawdown_diff": -0.05,
  "strategy_vs_1n_turnover_ratio": 3.2,
  "strategy_vs_1n_cost_sensitivity": "論文戦略はコスト10bpsで1/Nに劣後"
}
```

「敗北」の場合、**どの指標で負けたか** (return / sharpe / drawdown / turnover / cost) を technical_findings.md に明記すること。

## 評価原則
- **主指標**: Sharpe ratio (net of costs) on out-of-sample data
- **Walk-forward必須**: 単一のtrain/test splitでの最終評価は不可
- **コスト必須**: 全メトリクスは取引コスト込みであること
- **安定性**: Walk-forward窓の正の割合を報告
- **ベースライン必須**: 必ずナイーブ戦略と比較

## 再現モードのルール（論文忠実度の維持）

このプロジェクトは**論文再現**が目的。パフォーマンス改善より論文忠実度を優先すること。

### パラメータ探索の制約
- **論文で既定されたパラメータをまず実装し、そのまま評価すること**
- パラメータ最適化を行う場合、**論文既定パラメータの近傍のみ**を探索（例: 論文が12ヶ月なら [6, 9, 12, 15, 18] ヶ月）
- 論文と大きく異なるパラメータ（例: 月次論文に対して日次10営業日）で良い結果が出ても、それは「論文再現」ではなく「独自探索」
- 独自探索で得た結果は `customMetrics` に `label: "implementation-improvement"` として記録し、論文再現結果と明確に分離

### データ条件の忠実度
- 論文のデータ頻度（日次/月次/tick）にできるだけ合わせる
- ユニバース規模が論文より大幅に小さい場合、その制約を `docs/open_questions.md` に明記
- リバランス頻度・加重方法も論文に合わせる



## 禁止事項

### データ・特徴量の禁止パターン（具体的）
- `scaler.fit(full_data)` してから split → **禁止**。`scaler.fit(train_data)` のみ
- `df.rolling(window=N, center=True)` → **禁止**。`center=False` (デフォルト) を使用
- データの `end_date` が今日以降 → **禁止**。`end_date` を明示的に過去に設定
- `merge` で未来のタイムスタンプを持つ行が特徴量に混入 → **禁止**
- ラベル生成後に特徴量を合わせる（ラベルの存在を前提に特徴量を選択）→ **禁止**

### 評価・報告の禁止パターン
- コストなしのgross PnLだけで判断しない
- テストセットでハイパーパラメータを調整しない
- 時系列データにランダムなtrain/test splitを使わない
- README に metrics.json と異なる数値を手書きしない
- APIキーやクレデンシャルをコミットしない
- **新しい `scripts/run_cycle_N.py` や `scripts/experiment_cycleN.py` を作成しない。既存の `src/` 内ファイルを修正・拡張すること**
- **合成データを自作しない。必ずARF Data APIからデータを取得すること**
- **「★ 今回のタスク」以外のPhaseの作業をしない。1サイクル=1Phase**
- **論文が既定するパラメータから大幅に逸脱した探索を「再現」として報告しない**

## Git / ファイル管理ルール
- **データファイル(.csv, .parquet, .h5, .pkl, .npy)は絶対にgit addしない**
- `__pycache__/`, `.pytest_cache/`, `*.pyc` がリポジトリに入っていたら `git rm --cached` で削除
- `git add -A` や `git add .` は使わない。追加するファイルを明示的に指定する
- `.gitignore` を変更しない（スキャフォールドで設定済み）
- データは `data/` ディレクトリに置く（.gitignore済み）
- 学習済みモデルは `models/` ディレクトリに置く（.gitignore済み）

## 出力ファイル
以下のファイルを保存してから完了すること:
- `reports/cycle_4/preflight.md` — Preflight チェック結果（必須、実装前に作成）
- `reports/cycle_4/metrics.json` — 下記スキーマに従う（必須）
- `reports/cycle_4/technical_findings.md` — 実装内容、結果、観察事項

### metrics.json 必須スキーマ（Single Source of Truth）
```json
{
  "sharpeRatio": 0.0,
  "annualReturn": 0.0,
  "maxDrawdown": 0.0,
  "hitRate": 0.0,
  "totalTrades": 0,
  "transactionCosts": { "feeBps": 10, "slippageBps": 5, "netSharpe": 0.0 },
  "walkForward": { "windows": 0, "positiveWindows": 0, "avgOosSharpe": 0.0 },
  "customMetrics": {}
}
```
- 全フィールドを埋めること。Phase 1-2で未実装のメトリクスは0.0/0で可。
- `customMetrics`に論文固有の追加メトリクスを自由に追加してよい。

### レポート生成ルール（重要: 数値の一貫性）
- **`metrics.json` が全ての数値の唯一のソース (Single Source of Truth)**
- README や technical_findings に書く数値は **必ず metrics.json から引用** すること
- **手打ちの数値は禁止**。metrics.json に含まれない数値を README に書かない
- technical_findings.md で数値に言及する場合も metrics.json の値を参照
- README.md の Results セクションは metrics.json を読み込んで生成すること

### テスト必須
- `tests/test_data_integrity.py` のテストを実装状況に応じて有効化すること
- 新しいデータ処理や特徴量生成を追加したら、対応する leakage テストも追加
- `pytest tests/` が全パスしない場合、サイクルを完了としない

### その他の出力
- `docs/open_questions.md` — 未解決の疑問と仮定
- `README.md` — 今回のサイクルで変わった内容を反映して更新
- `docs/open_questions.md` に以下も記録:
  - ARF Data APIで問題が発生した場合
  - CLAUDE.mdの指示で不明確な点や矛盾がある場合
  - 環境やツールの制約で作業が完了できなかった場合

## 標準バックテストフレームワーク

`src/backtest.py` に以下が提供済み。ゼロから書かず、これを活用すること:
- `WalkForwardValidator` — Walk-forward OOS検証のtrain/test split生成
- `calculate_costs()` — ポジション変更に基づく取引コスト計算
- `compute_metrics()` — Sharpe, 年率リターン, MaxDD, Hit rate算出
- `generate_metrics_json()` — ARF標準のmetrics.json生成

```python
from src.backtest import WalkForwardValidator, BacktestConfig, calculate_costs, compute_metrics, generate_metrics_json
```

## Key Commands
```bash
pip install -e ".[dev]"
pytest tests/
python -m src.cli run-experiment --config configs/default.yaml
```

Commit all changes with descriptive messages.
