# Preflight Check — Cycle 4

## 1. データ境界表

| 項目 | 値 |
|---|---|
| データ取得終了日 | 2026-03-26 (今日 2026-03-30 以前) |
| Train期間 | 2016-06-22 〜 2024-04-10 |
| Validation期間 | Walk-forward folds内で動的に分割 |
| Test期間 | 2024-04-11 〜 2026-03-26 |
| 重複なし確認 | Yes |
| 未来日付なし確認 | Yes |

## 2. Feature timestamp 契約

- 全ての特徴量は時刻 t の予測に t-1 以前のデータのみを使用しているか？ → **Yes** (`create_rolling_windows` で `returns[i-lookback:i]` を使用)
- Scaler / Imputer は train データのみで fit しているか？ → **Yes** (`_scale_features` で `scaler.fit_transform(train_flat)` のみ)
- Centered rolling window を使用していないか？ → **Yes** (使用していない。デフォルトの `center=False`)

## 3. Paper spec 差分表

| パラメータ | 論文の値 | 現在の実装 | 一致? |
|---|---|---|---|
| ユニバース | ~88 commodity/equity futures | 20 ETFs | No (データ制約により縮小) |
| ルックバック期間 | 60日 | 60日 | Yes |
| リバランス頻度 | 日次 | 日次 | Yes |
| 特徴量 | 過去リターン系列 | 過去リターン系列 (log returns) | Yes |
| コストモデル | 売買回転率ペナルティ (gamma) | Cycle 4で実装 (gamma=0.1) | Yes (今回実装) |
| 損失関数 | -Sharpe + gamma * turnover | Cycle 4で実装 | Yes (今回実装) |
| 取引コスト | 論文では5-10bps想定 | 5bps (fee) + 0bps (slippage) = 5bps | Yes |
| LSTM hidden size | 64 | 64 | Yes |
| LSTM layers | 2 | 2 | Yes |
