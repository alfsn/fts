# PR 2b: Transaction Ledger & Real Holdings History

**Status:** planned · **Depends on:** PR 2a (raw prices, cached splits) · **Unblocks:** PR 7 onwards
**Packages:** `quant_core` (ledger model + pure replay/valuation), `quant_data` (YAML), analyser (`analyze`, `client-review`)

## Why

`client-review` draws a random walk, and `analyze` values positions with the YAML `current_price`, not market data. A snapshot only shows today. Back-casting today's holdings over 5 years has survivorship and look-ahead bias. Real performance needs a **ledger**: holdings rebuilt day by day, valued at market, with external flows separated from returns.

## Decisions (made with the user, 2026-11-15)

1. **Ledger source:** a hand-written `transactions:` list in `portfolio.yaml`. Broker importers come in later PRs.
2. **Reporting currency:** USD at CCL. `base_currency: USD`; ARS converts at CCL, which is the current behaviour.
3. **Splits come automatically from PR 2a's cached corporate actions** (as brokers and Sharesight do). `split` lines in the YAML are rejected, so nothing is counted twice.
4. **Reconciliation:** quantities rebuilt from the ledger on the snapshot date must match the YAML positions, or the run stops with a diff. This is the standard control against "price adjusted, quantity not". QuantConnect had exactly that bug with an 80:1 reverse split ([forum report](https://www.quantconnect.com/forum/discussion/21380/bug-report-goss-80-for-1-reverse-split-not-reflected-in-quantconnect-live-holdings/)).

## Quant rules this PR must respect

- **Only deposits and withdrawals are external cash flows.** Dividends, coupons, fees and taxes are *internal*: they land in, or leave, the portfolio's own cash and are part of the return.
  - The current backlog text ("deposits, withdrawals **and dividends** become `cash_flows`") is wrong. Treating a dividend as a deposit subtracts it from TWR and understates the return by the yield.
- **Raw × shares.** Value = raw close × share count, with splits applied to the share count on the ex-date. Never mix adjusted prices into valuation.
- **Timing convention** (matches `period_returns`): every event on date *d* is reflected in the close-of-*d* value, and each flow is already included in the value at its timestamp.
- **Undefined is NaN.** A held position with no price makes that day's value NaN. It never counts as 0 (which fixes `current_price or 0.0`).

## Design

### 1. Domain model (`quant_core.models`)

Extend the existing, unused `Transaction`. Set `model_config = ConfigDict(extra="forbid")`, because the YAML is hand-typed and a typo must fail.

```python
class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: date                                    # trade/ex date, local market day
    type: TransactionType                         # buy|sell|deposit|withdrawal|dividend|coupon|fee
    currency: Currency
    instrument_id: Optional[str] = None           # required for buy/sell/dividend/coupon
    provider_symbol: Optional[str] = None         # same meaning as Position.provider_symbol
    quantity: Optional[float] = Field(None, gt=0) # shares; direction comes from `type`
    price: Optional[float] = Field(None, gt=0)    # raw, per share, in `currency`
    amount: Optional[float] = Field(None, gt=0)   # cash, for deposit/withdrawal/dividend/coupon/fee
    fees: float = Field(0.0, ge=0)                # commission on buy/sell, in `currency`
    note: Optional[str] = None
```

- **Every number is positive, and `type` gives the direction.** Hand-typed signs are the most common ledger error.
- A `model_validator` enforces the required fields per type and rejects `split`.
- Add `FEE` to `TransactionType` for custody fees and taxes withheld. See open question 2.
- A **listing** is identified by `(instrument_id, currency)`. YPF in ARS and YPF in USD are different listings. A validator on `Portfolio` checks that each listing maps to one `market_symbol` across positions and transactions (DRY, one mapping).
- `Portfolio.transactions: list[Transaction] = []`.

### 2. Pure replay and valuation (`quant_core.ledger`, new module)

| Function | Returns |
|---|---|
| `holdings(transactions, splits, dates) -> DataFrame` | Share count per listing per date. Cumulative signed buys and sells. On a split's ex-date, quantity held × ratio. Splits are keyed by `market_symbol`, so the YPFD.BA split hits YPF-in-ARS only. |
| `cash(transactions, dates) -> DataFrame` | Cash per currency per date. Deposits +, withdrawals −, buys −(q·p + fees), sells +(q·p − fees), dividends, coupons and fees ±amount. |
| `external_flows(transactions, fx) -> Series` | Deposits (+) and withdrawals (−), converted to base currency at the flow date's FX. These are the `cash_flows` that `compute_performance_metrics` takes. |
| `portfolio_values(holdings, cash, closes, fx) -> Series` | Σ q·P·FX + Σ cash·FX in base currency, NaN where any held listing or needed FX rate is missing. |
| `reconcile(holdings_at, cash_at, portfolio) -> Breaks` | Per-listing quantity differences (tolerance 1e-9) and per-currency cash differences. |

- **Calendar:** union of the close dates of the held listings and of the FX series, from the first transaction up to the snapshot date.
- **Prices and FX are forward-filled across local holidays with a limit**: 5 sessions, the same constant as PR 2a's `MAX_CALENDAR_GAP`. Beyond that a price is NaN, not stale.
- **Days with a NaN value are dropped, with a warning that lists them.** If a deposit falls on a dropped day, `_aligned_flows` already raises, and that's what we want: a flow without a valuation is an error, not something to skip.
- FX is daily CCL from PR 2a: `ccl_from_pairs(adjust(closes, actions, SPLIT), pairs)`, in ARS per USD, inverted to USD per ARS.

### 3. YAML (`LocalYamlConnector`)

Nothing to write. `Portfolio.model_validate` picks up `transactions:`. Example:

```yaml
transactions:
  - {date: 2024-01-02, type: deposit,  currency: USD, amount: 10000}
  - {date: 2024-01-03, type: buy,      currency: USD, instrument_id: AAPL, quantity: 10, price: 185.6, fees: 1.0}
  - {date: 2024-02-16, type: dividend, currency: USD, instrument_id: AAPL, amount: 2.40}
  - {date: 2024-03-01, type: deposit,  currency: ARS, amount: 1500000}
  - {date: 2024-03-04, type: buy,      currency: ARS, instrument_id: YPF, provider_symbol: YPFD.BA, quantity: 100, price: 23500}
```

### 4. Analyser wiring

- **`analyze`** values positions at the **latest cached raw close**, not the YAML `current_price`. Price and FX use one common as-of date: the latest day on which every held listing and the CCL both have data. That date is printed. `current_price` in the YAML becomes optional and is ignored for valuation. If a ledger is present, the reconciliation result is printed.
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
| `wealth_index` with a NaN value | — | NaN poisons the cumprod. That's why NaN days are dropped *before* the call. |
| Splits from PR 2a | ratio = new shares per old share | Defined that way in PR 2a §1. Test the YPFD.BA 10:1 split on 2026-08-03 end to end: value continuous across the ex-date. |
| `ccl_from_pairs` | ARS per USD | Only on split-adjusted closes (PR 2a §6). |

## Tests

- **`quant_core.ledger`:**
  - Buy/sell/dividend/fee arithmetic.
  - The split multiplies quantity on the ex-date and the **value is continuous** across it, both with raw closes and split-adjusted ones.
  - A dividend raises TWR and is not a flow; a deposit leaves TWR unchanged.
  - A 2-currency portfolio at a CCL that jumps.
  - Forward-fill limit; a NaN day is dropped; a flow on a dropped day raises.
- **Model:** `extra="forbid"` rejects a typo (`quantitiy`); `split` is rejected; a sell without `price` is rejected.
- **Reconciliation:** a missing trade → a quantity break naming the listing.
- **Analyser:** `analyze` uses the market close and prints the as-of date; `client-review` produces a report from a fixture ledger with no network (cached DB fixture).

## fts (backlog questions 1 and 2)

1. **Prior art:** fts `risk_management/portfolio.py` processes fills statefully (average cost, realised PnL, single currency). Its signed-delta logic is the same idea as `holdings`, but it's event-driven live state, so it isn't refactored.
2. **Applies to fts:** `holdings` + `cash` + `portfolio_values` can rebuild a backtest's equity from `TradeLog` fills. That gives a cheap **consistency test** against `BacktestEquityLog` (and catches equity bookkeeping bugs). Follow-up, not in this PR.

## Open questions (with proposed answers)

1. **Opening holdings.** A client often already holds shares when the ledger starts.
   - *Proposal:* add `transfer_in` / `transfer_out` types. These are securities moved in kind, and count as **external flows at that day's market value** (the industry convention).
   - *Alternative:* the ledger must start with cash only, and you type the historical buys.
2. **Fees and taxes.** Should custody fees and withholding taxes (e.g. Argentine Bienes Personales) be typed in?
   - *Proposal:* yes, as `fee` lines. They're internal, so they lower the return honestly.
3. **Cash reconciliation tolerance.** A hand-typed ledger will miss tiny items (interest, rounding).
   - *Proposal:* a quantity break is an error; a cash break is a **warning** showing the unexplained amount per currency.
4. **Bonds and coupons** (AL30/GD30). There's no market data for them yet (PR 2 found Yahoo has none).
   - *Proposal:* accept `coupon` lines in the model, but bond positions fail valuation (NaN) until a data source exists. Out of scope here.
