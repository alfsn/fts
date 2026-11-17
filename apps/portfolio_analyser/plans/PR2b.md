# PR 2b: Transaction Ledger & Real Holdings History

**Status:** done (PRs #45, #46, #47; 2026-11-16) · **Depends on:** PR 2a (raw prices, cached splits) · **Unblocks:** PR 7 onwards
**Packages:** `quant_core` (ledger model + pure replay/valuation), `quant_data` (YAML), analyser (`analyze`, `client-review`)

## Why

`client-review` draws a random walk, and `analyze` values positions with the YAML `current_price`, not market data. A snapshot only shows today. Back-casting today's holdings over 5 years has survivorship and look-ahead bias. Real performance needs a **ledger**: holdings rebuilt day by day, valued at market, with external flows separated from returns.

## Decisions

D1–D4 made with the user on 2026-11-15, D5–D14 on 2026-11-16.

| # | Topic | Decision | Rejected alternative (why) |
|---|---|---|---|
| D1 | Ledger source | **A hand-written `transactions:` list in `portfolio.yaml`.** Broker importers come in later PRs. | — |
| D2 | Reporting currency | **USD at CCL.** `base_currency: USD`; ARS converts at CCL, which is the current behaviour. | — |
| D3 | Splits | **Automatically from PR 2a's cached corporate actions** (as brokers and Sharesight do). `split` lines in the YAML are rejected, so nothing is counted twice. | Typed splits: counted twice. |
| D4 | Reconciliation | **Quantities rebuilt from the ledger on the snapshot date must match the YAML positions, or the run stops with a diff.** This is the standard control against "price adjusted, quantity not". QuantConnect had exactly that bug with an 80:1 reverse split ([forum report](https://www.quantconnect.com/forum/discussion/21380/bug-report-goss-80-for-1-reverse-split-not-reflected-in-quantconnect-live-holdings/)). | — |
| D5 | Opening holdings | **`transfer_in` / `transfer_out` lines**: shares moved in kind. An **external flow** at that day's raw close × quantity (GIPS / Sharesight convention). | Cash-only start: needs the full historical trade list, and the history before it is lost. |
| D6 | Fees and taxes | **`fee` lines** (custody, Bienes Personales, withholding). Internal: cash leaves the portfolio and the return drops by it. | Leave them out: the return is overstated and cash never reconciles. |
| D7 | Reconciliation tolerance | **A quantity break is an error** (missing trade or split); **a cash break is a warning** that prints the unexplained amount per currency. | Both errors: a hand-typed ledger fails on every small cash item. Both warnings: a missed split silently gives a 10× value. |
| D8 | Bonds and coupons | **Rejected for now.** No price source exists (Yahoo has no AL30/GD30). The validator rejects `coupon` lines and bond listings. A backlog PR (**PR 2c**) researches a bond data source and adds them. | Accept coupons with NaN-valued bonds: every day is NaN while a bond is held, so no report. Exclude unpriced listings: wrong by the bond's weight. |
| D9 | Forward-fill limit | **5 sessions** for prices and the CCL rate, the same constant as PR 2a's `MAX_CALENDAR_GAP`. Beyond that a price is NaN, not stale. | No fill: every holiday on either exchange is a NaN day. Unlimited: hides delistings as flat returns. |
| D10 | NaN days | **Dropped with a warning that lists them.** A deposit/withdrawal on a dropped day still raises (`_aligned_flows`): a flow without a valuation is an error. | Stop the run: one missing bar blocks every report. |
| D11 | Dividend date | **Pay date, as on the broker statement.** The value dips between ex-date and pay date and recovers; total return over the period is unchanged. | Ex-date accrual: needs a receivable bucket and ex-date lookups. |
| D12 | Delivery | **Three stacked PRs** (see *Delivery*). | Two PRs: a bigger second review. |
| D13 | Listing key | **`(instrument_id, currency)`**, one `market_symbol` per listing, as `Position` already works. | `instrument_id` only: breaks when YPF is held in ARS and USD. |
| D14 | Trade date | **`Transaction.date` is the exchange's local trade date**, matched to that day's close. For the Americas it equals the UTC date PR 2 uses to index daily closes. Splits are matched on the local ex-date (PR 2a). | A UTC timestamp: nobody types one from a statement, and daily data can't use it. |

## Delivery: three stacked PRs (D12)

Stacked on `feature/PR-2a-3-cache-integrity`. Each PR targets the branch below it.

| PR | Branch | Scope | Base |
|---|---|---|---|
| 2b-1 | `feature/PR-2b-1-ledger-maths` | §1 model, §2 pure replay/valuation/reconciliation in `quant_core.ledger`. No app changes. | 2a-3 |
| 2b-2 | `feature/PR-2b-2-analyze-market-close` | §4 NaN instead of `or 0.0`; `analyze` at the latest market close; reconciliation printed. | 2b-1 |
| 2b-3 | `feature/PR-2b-3-client-review` | §4 `client-review` on real values and flows; mock baseline removed; backlog updated. | 2b-2 |

## Quant rules this PR must respect

- **Only deposits, withdrawals and in-kind transfers are external cash flows.** Dividends, fees and taxes are *internal*: they land in, or leave, the portfolio's own cash and are part of the return.
  - The current backlog text ("deposits, withdrawals **and dividends** become `cash_flows`") is wrong. Treating a dividend as a deposit subtracts it from TWR and understates the return by the yield.
  - A `transfer_in` is a flow of raw close × quantity on its date (D5); the shares are in the value from that day's close, so the flow and the value agree.
- **Raw × shares.** Value = raw close × share count, with splits applied to the share count on the ex-date. Never mix adjusted prices into valuation.
- **Timing convention** (matches `period_returns`): every event on date *d* is reflected in the close-of-*d* value, and each flow is already included in the value at its timestamp.
- **Undefined is NaN.** A held position with no price makes that day's value NaN. It never counts as 0 (which fixes `current_price or 0.0`).

## Design

### 1. Domain model (`quant_core.models`)

Extend the existing, unused `Transaction`. Set `model_config = ConfigDict(extra="forbid")`, because the YAML is hand-typed and a typo must fail.

```python
class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: date                                    # local trade date (D14); pay date for dividends (D11)
    type: TransactionType                         # buy|sell|transfer_in|transfer_out|deposit|withdrawal|dividend|fee
    currency: Currency
    instrument_id: Optional[str] = None           # required for buy/sell/transfer_in/transfer_out/dividend
    provider_symbol: Optional[str] = None         # same meaning as Position.provider_symbol
    quantity: Optional[float] = Field(None, gt=0) # shares; direction comes from `type`
    price: Optional[float] = Field(None, gt=0)    # raw, per share, in `currency`; buy/sell only
    amount: Optional[float] = Field(None, gt=0)   # cash, for deposit/withdrawal/dividend/fee
    fees: float = Field(0.0, ge=0)                # commission on buy/sell, in `currency`
    note: Optional[str] = None
```

- **Every number is positive, and `type` gives the direction.** Hand-typed signs are the most common ledger error.
- A `model_validator` enforces the required fields per type. `split` (D3) and `coupon` (D8) are rejected with a message that says why.
- `TransactionType` gains `TRANSFER_IN`, `TRANSFER_OUT` (D5) and `FEE` (D6). `COUPON` and `SPLIT` stay in the enum (fts may use them) but the ledger validator refuses them.
- A **listing** is identified by `(instrument_id, currency)` (D13). YPF in ARS and YPF in USD are different listings. A validator on `Portfolio` checks that each listing maps to one `market_symbol` across positions and transactions (DRY, one mapping).
- `Portfolio.transactions: list[Transaction] = []`.

### 2. Pure replay and valuation (`quant_core.ledger`, new module)

| Function | Returns |
|---|---|
| `holdings(transactions, splits, dates) -> DataFrame` | Share count per listing per date. Cumulative signed buys, sells and transfers. On a split's ex-date, the quantity held **at the previous close** × ratio, then that day's trades are added (a trade dated on the ex-date is already in new shares). Splits are keyed by `market_symbol`, so the YPFD.BA split hits YPF-in-ARS only. |
| `cash(transactions, dates) -> DataFrame` | Cash per currency per date. Deposits +, withdrawals −, buys −(q·p + fees), sells +(q·p − fees), dividends +amount, fees −amount. Transfers don't touch cash. |
| `external_flows(transactions, closes, fx) -> Series` | Deposits (+), withdrawals (−), transfers in (+ q × raw close) and out (−), converted to base currency at the flow date's FX. These are the `cash_flows` that `compute_performance_metrics` takes. A transfer whose listing has no (forward-filled) close on its date raises: a flow without a valuation. |
| `portfolio_values(holdings, cash, closes, fx) -> Series` | Σ q·P·FX + Σ cash·FX in base currency, NaN where any held listing or needed FX rate is missing. |
| `reconcile(holdings_at, cash_at, portfolio) -> Breaks` | Per-listing quantity differences (tolerance 1e-9, an **error**) and per-currency cash differences (a **warning**, D7). |

- **Calendar:** union of the close dates of the held listings and of the FX series, from the first transaction up to the snapshot date.
- **The ledger must start with a `deposit` or `transfer_in`** (validator), so the first value is positive. `period_returns` divides by `V_{t−1}`, and `compute_mwr` takes the first value as the initial contribution.
- **Prices and FX are forward-filled across local holidays with a limit** (D9): 5 sessions, the same constant as PR 2a's `MAX_CALENDAR_GAP`. Beyond that a price is NaN, not stale. The CCL series itself is filled, not its inputs: filling a stale local close against a fresh ADR close would make up a rate.
- **Days with a NaN value are dropped, with a warning that lists them** (D10). If a deposit falls on a dropped day, `_aligned_flows` already raises, and that's what we want: a flow without a valuation is an error, not something to skip.
- FX is daily CCL from PR 2a: `ccl_from_pairs(adjust(closes, actions, SPLIT), pairs)`, in ARS per USD, inverted to USD per ARS.

### 3. YAML (`LocalYamlConnector`)

Nothing to write. `Portfolio.model_validate` picks up `transactions:`. Example:

```yaml
transactions:
  - {date: 2024-01-02, type: transfer_in, currency: USD, instrument_id: AAPL, quantity: 5}   # held before the ledger starts
  - {date: 2024-01-02, type: deposit,     currency: USD, amount: 10000}
  - {date: 2024-01-03, type: buy,         currency: USD, instrument_id: AAPL, quantity: 10, price: 185.6, fees: 1.0}
  - {date: 2024-02-16, type: dividend,    currency: USD, instrument_id: AAPL, amount: 3.60}  # pay date
  - {date: 2024-03-01, type: deposit,     currency: ARS, amount: 1500000}
  - {date: 2024-03-04, type: buy,         currency: ARS, instrument_id: YPF, provider_symbol: YPFD.BA, quantity: 100, price: 23500}
  - {date: 2024-06-28, type: fee,         currency: ARS, amount: 12000, note: custody Q2}
```

### 4. Analyser wiring

- **`analyze`** values positions at the **latest cached raw close**, not the YAML `current_price`. Price and FX use one common as-of date: the latest day on which every held listing and the CCL both have data. That date is printed. `current_price` in the YAML becomes optional and is ignored for valuation. If a ledger is present, the reconciliation result is printed: quantity breaks stop the run, cash breaks are warnings (D7).
- **`Portfolio.total_value`, `MaxAssetWeight`, `DeviationRebalanceStrategy`:** remove `current_price or 0.0`. A missing price gives NaN. In weight checks an unpriced position is reported as a violation ("unpriced"), written so that NaN fails: `not weight <= max_weight`.
- **`client-review`:**
  - Builds the real value series and external flows from the ledger and passes them to `backward_analytics_figures(values, cash_flows)`.
  - Refuses to run when reconciliation fails.
  - The mock SPY baseline is **removed**. The honest, total-return, date-aligned benchmark comes in PR 7.
  - It needs `--db-url` like `analyze`. Share the option (DRY).

## Does each call do what we intend?

| Call | Intended | Verified / risk |
|---|---|---|
| `compute_performance_metrics(values, cash_flows)` | TWR immune to deposits | Yes: `period_returns` = (V_t − CF_t) / V_{t−1} − 1. It needs CF in base currency on a valued date. |
| `compute_mwr(values, cash_flows)` | XIRR | Uses the same flows. The first value is treated as the initial contribution, so a ledger starting from a deposit must start `values` on that deposit's date. |
| `wealth_index` with a NaN value | — | **Worse than poisoning** (checked 2026-11-16): pandas `cumprod` skips NaN, so a NaN value gives two NaN returns and the return *across* the hole is silently lost (100, 110, NaN, 121 ends at 1.21 instead of 1.331). Dropping NaN days *before* the call keeps that return (110 → 121). |
| `_aligned_flows` | a flow on a dropped day | Raises `Cash flows without a valuation`, which is what D10 wants. |
| Splits from PR 2a | ratio = new shares per old share | Defined that way in PR 2a §1. Test the YPFD.BA 10:1 split on 2026-08-03 end to end: value continuous across the ex-date. |
| `fetch_daily_closes` index | local trade date | Yes: daily bars are cached at 00:00 UTC on the local date (PR 2a §3), so the UTC date equals `Transaction.date` (D14). |
| `ccl_from_pairs` | ARS per USD | Only on split-adjusted closes (PR 2a §6). Forward-fill the resulting CCL series, not its inputs (D9). |
| `Position.current_price` | — | Ignored for valuation after this PR; the `or 0.0` at `models.py`, `risk.py` and `rebalance.py` becomes NaN. |

## Tests

- **`quant_core.ledger`:**
  - Buy/sell/dividend/fee arithmetic.
  - A `transfer_in` adds shares, leaves cash alone, and is a flow of close × quantity, so TWR is unchanged by it.
  - The split multiplies quantity on the ex-date and the **value is continuous** across it, both with raw closes and split-adjusted ones.
  - A dividend raises TWR and is not a flow; a deposit leaves TWR unchanged; a fee lowers TWR.
  - A 2-currency portfolio at a CCL that jumps.
  - Forward-fill limit; a NaN day is dropped; a flow on a dropped day raises.
- **Model:** `extra="forbid"` rejects a typo (`quantitiy`); `split` and `coupon` are rejected; a sell without `price` is rejected; a `transfer_in` with a `price` is rejected; a ledger that starts with a `buy` is rejected.
- **Ex-date ordering:** a buy dated on the split's ex-date is not multiplied.
- **Reconciliation:** a missing trade → a quantity break naming the listing (error); a missing fee → a cash break (warning).
- **Analyser:** `analyze` uses the market close and prints the as-of date; `client-review` produces a report from a fixture ledger with no network (cached DB fixture).

## fts (backlog questions 1 and 2)

1. **Prior art:** fts `risk_management/portfolio.py` processes fills statefully (average cost, realised PnL, single currency). Its signed-delta logic is the same idea as `holdings`, but it's event-driven live state, so it isn't refactored.
2. **Applies to fts:** `holdings` + `cash` + `portfolio_values` can rebuild a backtest's equity from `TradeLog` fills. That gives a cheap **consistency test** against `BacktestEquityLog` (and catches equity bookkeeping bugs). Follow-up, not in this PR.

## Out of scope

- **Bonds and coupons** (D8): PR 2c in the backlog.
- Broker importers (D1), ex-date dividend accrual (D11), a benchmark (PR 7).
