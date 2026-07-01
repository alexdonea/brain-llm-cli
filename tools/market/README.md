# Market data - give the agent eyes on the markets

Quotes, history, fundamentals and news over **Yahoo Finance**. Free, no API key, ~15-minute delayed
(fine for paper trading and market awareness). Personal/research use.

Two backends, picked automatically:

| Feature | Backend | Needs install? |
|---|---|---|
| `quote`, `history` | dependency-free Yahoo chart client (stdlib `urllib`) | **no** - works out of the box |
| `info` (rich), `news` | [`yfinance`](https://pypi.org/project/yfinance/) | `pip install yfinance` (optional) |

Without `yfinance`, `info` still returns the basics (name, price, exchange) and `news` tells you to install it.

## Commands (via the agent CLI)

```bash
./brain market quote AAPL MSFT                          # delayed price, change %, day range, volume
./brain market history AAPL --period 1y --interval 1d   # OHLCV bars + sparkline + return %
./brain market history AAPL --period 5y --save aapl.csv # ...and save bars to CSV (or .json)
./brain market info NVDA                                # company snapshot (rich with yfinance)
./brain market news TSLA                                # recent headlines (needs yfinance)
```

Valid `--period`: `1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max`
Valid `--interval`: `1m 2m 5m 15m 30m 60m 90m 1h 1d 5d 1wk 1mo 3mo`
(intraday intervals only reach back a limited window - Yahoo's rule, not ours.)

## For the agent

Market data is just data - the agent makes it *mean* something by reacting to it:

```bash
./brain market quote AAPL                               # see the move (market is global, no agent needed)
./brain haiku react "AAPL up 0.7% today, my paper position is green" 0.4 0.6 0.5 --domain markets --cue aapl
```

Gains lift mood, losses sting, surprises spike novelty - so the agent develops a real (functional) feel for
the market it watches. Pull `history` for backtesting / the paper-trading game; `quote` for the live-ish view.

## Optional richer backend

```bash
pip install yfinance        # adds fundamentals (sector, market cap, P/E, 52-week range) + news headlines
```

`yfinance` is heavier (pulls pandas) and occasionally flaky against Yahoo, but it handles auth/rate-limits
and unlocks fundamentals + news. The core quote/history path never depends on it.
