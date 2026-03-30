# Preflight Check — Cycle 2

## 1. データ境界表

| 項目 | 値 |
|---|---|
| データ取得終了日 | 2026-03-28 (今日 2026-03-30 以前) |
| Train期間 | データ依存 (walk-forward, 各ETFの利用可能期間による) |
| Validation期間 | N/A (Phase 2はデータパイプライン構築のみ) |
| Test期間 | N/A (Phase 2はデータパイプライン構築のみ) |
| 重複なし確認 | Yes (walk-forwardはPhase 3以降で実装) |
| 未来日付なし確認 | Yes (APIデータは過去のみ、end_dateは今日以前) |

**注**: Phase 2はデータパイプライン構築フェーズのため、Train/Validation/Test分割は実施しない。データの取得・前処理・形状確認が主な作業。

## 2. Feature timestamp 契約

- 全ての特徴量は時刻 t の予測に t-1 以前のデータのみを使用しているか？ → **Yes** (lookback window は i-lookback:i の範囲、forward return は i+1)
- Scaler / Imputer は train データのみで fit しているか？ → **Yes** (Phase 2では scaler 未使用。Phase 3以降で train-only fit を保証)
- Centered rolling window を使用していないか？ → **Yes** (center=False、デフォルト使用)

## 3. Paper spec 差分表

| パラメータ | 論文の値 | 現在の実装 | 一致? |
|---|---|---|---|
| ユニバース | ~88 liquid futures across asset classes | 20 liquid ETFs (代替) | No (ETFで代替、docs/open_questions.mdに記録) |
| ルックバック期間 | 複数期間 (短期〜長期) | 60日固定 | No (Phase 2では60日、Phase 6で拡張予定) |
| リバランス頻度 | 日次 | 日次 | Yes |
| 特徴量 | 過去リターン系列 | 対数リターンのlookback window | Yes |
| コストモデル | 取引コスト考慮 | 10bps fee + 5bps slippage | Yes (Phase 4で正則化追加) |

**差分の理由**: 論文は先物を使用するが、ARF Data APIではETFで代替。ユニバースサイズは論文より小さいが、主要アセットクラスをカバー。
