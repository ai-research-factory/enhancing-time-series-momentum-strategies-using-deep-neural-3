# Preflight Check — Cycle 5

## 1. データ境界表

| 項目 | 値 |
|---|---|
| データ取得終了日 | 2026-03-26 (今日 2026-03-30 以前) |
| Train期間 | 2016-06-22 〜 各フォールドで動的 (expanding window) |
| Validation期間 | Walk-forward folds内で動的に分割 (5 folds) |
| Test期間 | 各フォールド約1年、最終: ~2025-03 〜 2026-03-26 |
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
| ルックバック期間 | 12ヶ月 (~252日) | 60日 → **252日に変更** | Yes (Cycle 5で修正) |
| リバランス頻度 | 月次 | 日次 → **月次に変更** | Yes (Cycle 5で修正) |
| 特徴量 | 過去リターン系列 | 過去リターン系列 (log returns) | Yes |
| コストモデル | 売買回転率ペナルティ (gamma) | gamma=0.1 | Yes |
| 損失関数 | -Sharpe + gamma * turnover | 実装済み | Yes |
| 取引コスト | 論文では5-10bps想定 | 10bps (fee=5, slippage=5) | Yes |
| LSTM hidden size | 64 | 64 | Yes |
| LSTM layers | 2 | 2 | Yes |
| Walk-forward検証 | 5-10 folds | 5 folds (train 3yr, test 1yr rolling) | Yes |

## 4. Cycle 5 変更点

1. **月次リバランス**: 日次から月次に変更。各月末にのみポジション更新。
2. **ルックバック期間**: 60日から252日（約12ヶ月）に変更。論文仕様に準拠。
3. **WalkForwardValidator**: `src/evaluation.py`に新クラスとして実装。月次リバランス対応。
4. **run-walk-forward コマンド**: `src/main.py`に追加。
