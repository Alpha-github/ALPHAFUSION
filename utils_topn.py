from typing import Tuple, List
import pandas as pd

__all__ = ["get_top_n_by_history", "get_top_n_by_news"]


def get_top_n_by_history(df: pd.DataFrame, ticker_col: str = "ticker", date_col: str = "date", n: int = 50) -> Tuple[pd.DataFrame, List[str]]:
    """Return rows for the top-`n` tickers with the longest history.

    History length is measured as the number of unique values in `date_col` if present,
    otherwise by the number of rows per ticker.

    Returns (subset_df, top_ticker_list).
    """
    if ticker_col not in df.columns:
        raise KeyError(f"ticker_col '{ticker_col}' not found in DataFrame columns")
    if date_col in df.columns:
        counts = df.groupby(ticker_col)[date_col].nunique()
    else:
        counts = df.groupby(ticker_col).size()
    # if n == -1 return all tickers (entire dataframe)
    if n == -1:
        top_tickers = counts.sort_values(ascending=False).index.tolist()
        subset = df.copy()
        return subset, top_tickers
    top_tickers = counts.sort_values(ascending=False).head(n).index.tolist()
    subset = df[df[ticker_col].isin(top_tickers)].copy()
    return subset, top_tickers


def get_top_n_by_news(news_df: pd.DataFrame, ticker_col: str = "ticker", news_count_col: str | None = None, n: int = 50) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    """Return rows for the top-`n` tickers with the most news.

    If `news_count_col` is provided (or if the DataFrame contains a `news_count` column),
    the function sums that column per ticker. Otherwise it counts rows per ticker.

    Returns (subset_df, top_ticker_list, totals_series_for_top).
    """
    if ticker_col not in news_df.columns:
        raise KeyError(f"ticker_col '{ticker_col}' not found in DataFrame columns")
    if news_count_col and news_count_col in news_df.columns:
        totals = news_df.groupby(ticker_col)[news_count_col].sum()
    elif "news_count" in news_df.columns:
        totals = news_df.groupby(ticker_col)["news_count"].sum()
    else:
        totals = news_df.groupby(ticker_col).size()
    # if n == -1 return all tickers (entire dataframe)
    if n == -1:
        top_tickers = totals.sort_values(ascending=False).index.tolist()
        subset = news_df.copy()
        return subset, top_tickers, totals.loc[top_tickers]
    top_tickers = totals.sort_values(ascending=False).head(n).index.tolist()
    subset = news_df[news_df[ticker_col].isin(top_tickers)].copy()
    # return totals in the same order as top_tickers
    return subset, top_tickers, totals.loc[top_tickers]
