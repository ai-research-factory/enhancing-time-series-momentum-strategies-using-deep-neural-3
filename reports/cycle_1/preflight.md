# Preflight Check — Cycle 1

## 1. データ境界表

| 項目 | 値 |
|---|---|
| データ取得終了日 | 2026-03-28 (今日以前であること) |
| Train期間 | 2016-03-30 〜 2022-03-30 |
| Validation期間 | 2022-03-31 〜 2024-03-29 |
| Test期間 | 2024-04-01 〜 2026-03-28 |
| 重複なし確認 | Yes |
| 未来日付なし確認 | Yes |

**Note**: Data fetched from ARF Data API (`^N225`, interval=1d, period=10y). Total rows: ~2441.
Train/Val/Test split is approximately 60/20/20 by time. Walk-forward validation is used
instead of a single static split for final evaluation.

## 2. Feature Timestamp 契約

- 全ての特徴量は時刻 t の予測に t-1 以前のデータのみを使用しているか？ → Yes
  - Features are rolling returns over past N days (lookback window), computed with `center=False`
  - Position at time t is determined by returns up to t-1
- Scaler / Imputer は train データのみで fit しているか？ → Yes
  - StandardScaler fitted on training data only, applied to val/test via transform()
- Centered rolling window を使用していないか？ → Yes (not used)
  - All rolling operations use default `center=False`

## 3. Paper Spec 差分表

| パラメータ | 論文の値 | 現在の実装 | 一致? |
|---|---|---|---|
| ユニバース | 88 futures across asset classes | Nikkei 225 index (single asset) | No — simplified for v1 |
| ルックバック期間 | Multiple (various horizons) | 60 days | Partial — matches design brief |
| リバランス頻度 | Daily | Daily | Yes |
| 特徴量 | Past returns at multiple horizons | Past 60-day daily returns | Partial — simplified |
| モデル | LSTM | MLP (per design brief) | No — simplified per design brief |
| 損失関数 | Differentiable Sharpe ratio | Differentiable Sharpe ratio | Yes |
| ポジションサイジング | tanh output (-1, +1) | tanh output (-1, +1) | Yes |
| コストモデル | Turnover regularization | Fee + slippage (15 bps total) | Partial |

**Key simplifications** (per design brief):
- Single asset instead of multi-asset universe
- MLP instead of LSTM
- Fixed 60-day lookback instead of multi-horizon features
