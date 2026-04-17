import time
import pandas as pd
from vnstock import Quote  # replace with vnstock_data if sponsored


def fetch_all_intraday(
    symbol: str, source: str = "vci", max_pages: int = 60, delay: float = 3.5
) -> pd.DataFrame:
    """Fetch all intraday ticks for the current session, respecting rate limits.

    delay: seconds between requests — 3.5s keeps free tier under 20 req/min.
    """
    quote = Quote(source=source, symbol=symbol)
    pages = []
    for page in range(1, max_pages + 1):
        try:
            df = quote.intraday(symbol=symbol, page_size=500, page=page)
        except Exception as e:
            print(f"  Page {page} error: {e} — stopping.")
            break
        if df.empty:
            break
        pages.append(df)
        print(f"  Page {page:>3}: {len(df)} rows | "
              f"time range {df['time'].min()} → {df['time'].max()}")
        if len(df) < 500:
            break
        time.sleep(delay)
    return pd.concat(pages, ignore_index=True) if pages else pd.DataFrame()


def analyze_large_orders(symbol: str, source: str = "vci", top_pct: float = 0.10):
    print(f"\n{'='*55}")
    print(f"  Large Order Analysis: {symbol} | Top {int(top_pct*100)}%")
    print(f"{'='*55}")

    df = fetch_all_intraday(symbol, source)
    if df.empty:
        print("No data returned.")
        return

    print(f"Total matched ticks fetched : {len(df):,}")

    # Threshold: top 10% by volume
    threshold = df["volume"].quantile(1 - top_pct)
    print(f"Volume threshold (p{int((1-top_pct)*100)}) : {threshold:,.0f}")

    large = df[df["volume"] >= threshold].copy()
    print(f"Large orders (>= threshold): {len(large):,} ticks\n")

    # --- Overall split ---
    summary = (
        large.groupby("match_type")
        .agg(
            order_count=("volume", "count"),
            total_volume=("volume", "sum"),
            avg_volume=("volume", "mean"),
            avg_price=("price", "mean"),
            vwap=("price", lambda x: (x * large.loc[x.index, "volume"]).sum()
                  / large.loc[x.index, "volume"].sum()),
            min_price=("price", "min"),
            max_price=("price", "max"),
        )
        .reset_index()
    )
    summary["pct_of_large_vol"] = (
        summary["total_volume"] / summary["total_volume"].sum() * 100
    ).round(2)

    print("--- Large Order Summary by Side ---")
    print(summary.to_string(index=False))

    # --- Net flow ---
    buy_vol  = summary.loc[summary["match_type"] == "Buy",  "total_volume"].sum()
    sell_vol = summary.loc[summary["match_type"] == "Sell", "total_volume"].sum()
    net      = buy_vol - sell_vol
    direction = "NET BUY" if net > 0 else "NET SELL"
    print(f"\nNet large-order flow : {net:+,.0f} ({direction})")

    # --- VWAP comparison ---
    buy_vwap  = summary.loc[summary["match_type"] == "Buy",  "vwap"].values
    sell_vwap = summary.loc[summary["match_type"] == "Sell", "vwap"].values
    if len(buy_vwap) and len(sell_vwap):
        print(f"VWAP Buy  (top {int(top_pct*100)}%) : {buy_vwap[0]:,.2f}")
        print(f"VWAP Sell (top {int(top_pct*100)}%) : {sell_vwap[0]:,.2f}")

    # --- Time distribution: large orders per 30-min bucket ---
    large["bucket"] = large["time"].dt.floor("30min")
    time_dist = (
        large.groupby(["bucket", "match_type"])["volume"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    time_dist["net"] = time_dist.get("Buy", 0) - time_dist.get("Sell", 0)
    print(f"\n--- Large Order Flow by 30-min Bucket ---")
    print(time_dist.to_string(index=False))

    return large, summary


if __name__ == "__main__":
    large_df, summary_df = analyze_large_orders(
        symbol="HPG",
        source="vci",   # switch to "kbs" on Colab/GCP
        top_pct=0.10,
    )
