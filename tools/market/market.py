"""
market.py - give the agent market-data powers (quotes, history, fundamentals, news) over Yahoo Finance.

Two backends, picked automatically:
  * quote + history  -> a dependency-free Yahoo chart client (stdlib urllib only). Works out of the box,
                        no install, no API key.
  * info + news      -> yfinance if installed (`pip install yfinance`), for fundamentals & headlines;
                        info degrades to the chart metadata when yfinance is absent.

Data is delayed (~15 min) - free, keyless, fine for paper trading and market awareness. Personal/research use.

CLI (also wired into the agent CLI as `./brain market <action>`):
  python3 tools/market/market.py quote AAPL MSFT
  python3 tools/market/market.py history AAPL --period 1y --interval 1d --save aapl.csv
  python3 tools/market/market.py info NVDA
  python3 tools/market/market.py news TSLA
"""
import csv
import datetime
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (brain-llm market tool)"}
CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{t}?range={r}&interval={i}&includePrePost=false"

# valid Yahoo values, surfaced for help text / validation
PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
INTERVALS = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]


def _yf():
    """Lazily import yfinance only when a feature needs it (keeps quote/history fast & dependency-free)."""
    try:
        import yfinance
        return yfinance
    except Exception:
        return None


def _http_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25) as r:
        try:
            return json.load(r)
        except json.JSONDecodeError as e:                   # Yahoo HTML rate-limit/gateway 200 → treat as no data
            raise urllib.error.URLError(f"non-JSON response: {e}")


def _fmt_date(ts, interval):
    intraday = interval.endswith("m") or interval.endswith("h")
    dt = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M" if intraday else "%Y-%m-%d")


def yahoo_chart(ticker, period="1y", interval="1d"):
    """Stdlib Yahoo chart call -> (meta dict, [bar,...]); each bar = date/open/high/low/close/volume."""
    url = CHART.format(t=urllib.parse.quote(ticker), r=period, i=interval)
    res = _http_json(url)
    results = (res.get("chart") or {}).get("result") or []
    if not results:
        return None, []
    r0 = results[0]
    meta = r0.get("meta", {}) or {}
    ts = r0.get("timestamp", []) or []
    q = ((r0.get("indicators", {}).get("quote") or [{}])[0]) or {}
    o, h, lo, c, v = (q.get("open") or [], q.get("high") or [], q.get("low") or [],
                      q.get("close") or [], q.get("volume") or [])
    bars = []
    for i in range(len(ts)):
        close = c[i] if i < len(c) else None
        if close is None:
            continue
        bars.append({"date": _fmt_date(ts[i], interval),
                     "open": o[i] if i < len(o) else None, "high": h[i] if i < len(h) else None,
                     "low": lo[i] if i < len(lo) else None, "close": close,
                     "volume": v[i] if i < len(v) else None})
    return meta, bars


def quote(ticker):
    """Latest (delayed) snapshot for one ticker, from the chart metadata. Returns a dict or None."""
    try:
        meta, _ = yahoo_chart(ticker, "1d", "1m")
    except (urllib.error.URLError, json.JSONDecodeError, KeyError, ValueError, TypeError):
        meta = None
    if not meta:                                           # last-ditch fallback via yfinance, if present
        yf = _yf()
        if yf:
            try:
                fi = dict(yf.Ticker(ticker).fast_info)
                price, prev = fi.get("lastPrice") or fi.get("last_price"), fi.get("previousClose") or fi.get("previous_close")
                return _quote_obj(ticker, ticker, price, prev, fi.get("dayHigh"), fi.get("dayLow"),
                                  fi.get("lastVolume"), fi.get("currency"), fi.get("exchange"))
            except (urllib.error.URLError, json.JSONDecodeError, KeyError, ValueError, TypeError):
                return None
        return None
    price = meta.get("regularMarketPrice")
    prev = meta.get("chartPreviousClose") or meta.get("previousClose")
    return _quote_obj(meta.get("symbol", ticker), meta.get("longName") or meta.get("shortName") or ticker,
                      price, prev, meta.get("regularMarketDayHigh"), meta.get("regularMarketDayLow"),
                      meta.get("regularMarketVolume"), meta.get("currency"), meta.get("exchangeName"))


def _quote_obj(symbol, name, price, prev, hi, lo, vol, currency, exchange):
    chg = (price - prev) if (price is not None and prev is not None) else None
    pct = (chg / prev * 100) if (chg is not None and prev) else None   # prev truthiness guards div-by-zero
    return {"symbol": symbol, "name": name, "price": price, "prev_close": prev, "change": chg,
            "change_pct": pct, "day_high": hi, "day_low": lo, "volume": vol,
            "currency": currency or "", "exchange": exchange or ""}


def history(ticker, period="1y", interval="1d"):
    """Historical OHLCV bars. Prefers yfinance (robust against Yahoo auth/rate-limits) when installed."""
    yf = _yf()
    if yf:
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
            bars = []
            for idx, row in df.iterrows():
                bars.append({"date": str(idx)[:16], "open": _f(row.get("Open")), "high": _f(row.get("High")),
                             "low": _f(row.get("Low")), "close": _f(row.get("Close")), "volume": _f(row.get("Volume"))})
            if bars:
                return {"symbol": ticker}, bars
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass
    try:
        return yahoo_chart(ticker, period, interval)
    except urllib.error.URLError:                          # offline / Yahoo down → graceful empty, no crash
        return None, []


def _f(x):
    try:
        return None if x is None else float(x)
    except Exception:
        return None


def info(ticker):
    """Company snapshot. Rich via yfinance; basic (name/price/exchange) via chart metadata otherwise."""
    yf = _yf()
    if yf:
        try:
            d = yf.Ticker(ticker).info or {}
            if d.get("longName") or d.get("shortName"):
                return {"symbol": d.get("symbol", ticker), "name": d.get("longName") or d.get("shortName"),
                        "sector": d.get("sector"), "industry": d.get("industry"),
                        "market_cap": d.get("marketCap"), "pe": d.get("trailingPE"),
                        "wk52_high": d.get("fiftyTwoWeekHigh"), "wk52_low": d.get("fiftyTwoWeekLow"),
                        "currency": d.get("currency"), "country": d.get("country"), "website": d.get("website"),
                        "summary": (d.get("longBusinessSummary") or "")[:400], "rich": True}
        except (urllib.error.URLError, json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass
    meta, _ = yahoo_chart(ticker, "1d", "1d")
    if not meta:
        return None
    return {"symbol": meta.get("symbol", ticker), "name": meta.get("longName") or meta.get("shortName") or ticker,
            "currency": meta.get("currency"), "exchange": meta.get("exchangeName"),
            "price": meta.get("regularMarketPrice"), "rich": False}


def news(ticker, limit=6):
    """Recent headlines (yfinance only). Returns a list (possibly empty) or None if yfinance is absent."""
    yf = _yf()
    if not yf:
        return None
    try:
        out = []
        for it in (yf.Ticker(ticker).news or [])[:limit]:
            c = it.get("content") or it                    # yfinance changed shape across versions
            title = c.get("title") or it.get("title")
            pub = (c.get("provider") or {}).get("displayName") if isinstance(c.get("provider"), dict) else it.get("publisher")
            link = (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else it.get("link")
            if title:
                out.append({"title": title, "publisher": pub or "", "link": link or ""})
        return out
    except (urllib.error.URLError, json.JSONDecodeError, KeyError, ValueError, TypeError):
        return []


# ── presentation ───────────────────────────────────────────────────────────────────────────────────
def sparkline(values):
    vals = [v for v in values if v is not None and v == v]   # drop None and NaN (NaN != NaN)
    if not vals:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return "".join(blocks[int((v - lo) / rng * (len(blocks) - 1))] for v in vals)


def summarize(bars):
    closes = [b["close"] for b in bars if b["close"] is not None]
    if not closes:
        return {}
    ret = (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] else None
    return {"bars": len(bars), "first": closes[0], "last": closes[-1], "min": min(closes), "max": max(closes),
            "return_pct": ret, "from": bars[0]["date"], "to": bars[-1]["date"]}


def _n(x, d=2):
    return f"{x:,.{d}f}" if isinstance(x, (int, float)) else "n/a"


def fmt_quote(q):
    if not q:
        return "(no data)"
    head = f"{q['symbol']}  {q['name']}".strip()
    if q["change"] is None:
        line = f"  {_n(q['price'])} {q['currency']}"
    else:
        arrow = "▲" if q["change"] >= 0 else "▼"
        line = f"  {_n(q['price'])} {q['currency']}   {arrow} {q['change']:+,.2f} ({q['change_pct']:+.2f}%)"
    stats = f"  day {_n(q['day_low'])}–{_n(q['day_high'])}   prev {_n(q['prev_close'])}   vol {_n(q['volume'], 0)}   {q['exchange']}"
    return f"{head}\n{line}\n{stats}"


def fmt_history(ticker, bars, period, interval):
    if not bars:
        return f"{ticker}: (no history for period={period} interval={interval})"
    s = summarize(bars)
    if not s:                                              # bars present but all closes were None
        return f"{ticker}: (no usable closes for period={period} interval={interval})"
    spark = sparkline([b["close"] for b in bars])
    ret = f"{s['return_pct']:+.2f}%" if s.get("return_pct") is not None else "n/a"
    return (f"{ticker}  {period}/{interval}  ({s['from']} → {s['to']}, {s['bars']} bars)\n"
            f"  {spark}\n"
            f"  first {_n(s['first'])}   last {_n(s['last'])}   low {_n(s['min'])}   high {_n(s['max'])}   return {ret}")


def fmt_info(d):
    if not d:
        return "(no data)"
    if not d.get("rich"):
        return (f"{d['symbol']}  {d['name']}\n  {_n(d.get('price'))} {d.get('currency','')}   {d.get('exchange','')}\n"
                f"  (install yfinance for sector, market cap, P/E, 52-week range, summary)")
    rows = [f"{d['symbol']}  {d['name']}",
            f"  sector: {d.get('sector') or 'n/a'} / {d.get('industry') or 'n/a'}",
            f"  market cap: {_n(d.get('market_cap'), 0)} {d.get('currency','')}   P/E: {_n(d.get('pe'))}",
            f"  52-week: {_n(d.get('wk52_low'))} – {_n(d.get('wk52_high'))}   {d.get('country') or ''}"]
    if d.get("summary"):
        rows.append(f"  {d['summary'].strip()}…")
    return "\n".join(rows)


def fmt_news(items, ticker=""):
    if items is None:
        return "(news needs yfinance - run: pip install yfinance)"
    if not items:
        return f"{ticker}: (no recent headlines)"
    return "\n".join(f"  • {it['title']}" + (f"  - {it['publisher']}" if it['publisher'] else "") for it in items)


def save_bars(bars, path):
    if path.endswith(".csv"):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "open", "high", "low", "close", "volume"])
            for b in bars:
                w.writerow([b["date"], b["open"], b["high"], b["low"], b["close"], b["volume"]])
    else:
        with open(path, "w") as f:
            json.dump(bars, f, indent=2)


def main(argv):
    if not argv:
        print("usage: quote <T..> | history <T> [--period 1y --interval 1d --save f] | info <T> | news <T>")
        return
    action = argv[0]
    rest = argv[1:]
    opts = {"--period": "1y", "--interval": "1d", "--save": None}
    tickers = []
    i = 0
    while i < len(rest):
        if rest[i] in opts:
            if i + 1 >= len(rest):                          # trailing flag with no value → clean message, not a traceback
                raise SystemExit(f"{rest[i]} needs a value, e.g. {rest[i]} 1y")
            opts[rest[i]] = rest[i + 1]
            i += 2
        else:
            tickers.append(rest[i])
            i += 1
    try:
        if action == "quote":
            print("\n\n".join(fmt_quote(quote(t)) for t in tickers))
        elif action == "history":
            _, bars = history(tickers[0], opts["--period"], opts["--interval"])
            print(fmt_history(tickers[0], bars, opts["--period"], opts["--interval"]))
            if opts["--save"]:
                save_bars(bars, opts["--save"])
                print(f"  saved {len(bars)} bars → {opts['--save']}")
        elif action == "info":
            print(fmt_info(info(tickers[0])))
        elif action == "news":
            print(fmt_news(news(tickers[0]), tickers[0]))
        else:
            print(f"unknown action: {action}")
    except urllib.error.URLError as e:
        raise SystemExit(f"Market request failed (network/Yahoo): {e}")
    except IndexError:
        raise SystemExit("need a ticker, e.g. market quote AAPL")


if __name__ == "__main__":
    main(sys.argv[1:])
