import numpy as np
import yfinance as yf
import pandas as pd

def read_csv(filename):
    data=pd.read_csv(filename, header=None)
    l= data.values.tolist()
    list=[]
    for i in l:
        list.append(i[0])
    return list

def compare_portfolio_to_benchmark(portfolio_dict, start, end):
    """
    Takes in a portfolio, compares its return within start and end dates
    to the benchmark average return. For testing purposes.
    """

    tickers = list(portfolio_dict.keys())
    weights = np.array(list(portfolio_dict.values()))

    data = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    daily_ret = data.pct_change().dropna().dot(weights)
    portfolio_total = (1 + daily_ret).prod() - 1

    tsx = yf.download("^GSPTSE", start=start, end=end, progress=False)["Close"]
    spx = yf.download("^GSPC", start=start, end=end, progress=False)["Close"]


    if isinstance(tsx, pd.DataFrame):
        tsx = tsx.squeeze()
    if isinstance(spx, pd.DataFrame):
        spx = spx.squeeze()

    tsx_total = tsx.iloc[-1] / tsx.iloc[0] - 1
    spx_total = spx.iloc[-1] / spx.iloc[0] - 1
    benchmark_total = (tsx_total + spx_total) / 2

    diff = portfolio_total - benchmark_total

    return {
        "Portfolio_Total_Return_%": round(float(portfolio_total * 100), 2),
        "Benchmark_Avg_Return_%": round(float(benchmark_total * 100), 2),
        "Difference_%": round(float(diff * 100), 2)
    }


def check_ticker(list):
    valid_tickers=[]
    invalid_tickers=[]
    start = "2024-10-01"
    end = "2025-10-01"

    market = yf.Ticker("^GSPC")
    market_data = market.history(start=start, end=end, interval="1d")
    market_data["Market_Return"] = market_data["Close"].pct_change()


    for ticker in list:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start, end=end, interval="1d")

        if data.empty:
            invalid_tickers.append(ticker)
            continue

        avg_volume = data["Volume"].mean()


        if avg_volume < 5000:
            invalid_tickers.append(ticker)
            continue

        valid_tickers.append(ticker)

        # For Companies in S&P 500/TSX, add them to the final list
        market_of_ticker = stock.info.get("market")
        if market_of_ticker not in ["us_market", "ca_market"]:
            invalid_tickers.append(ticker)
            valid_tickers.remove(ticker)
            continue

    return valid_tickers, invalid_tickers

def beta_calculate(valid_tickers):
    benchmark="^GSPC"
    start="2025-05-15"
    end="2025-11-15"
    valid_stocks_with_beta = []
    for i in valid_tickers:
        data = yf.download([i,benchmark], start=start, end=end)["Close"]

        rets = data.pct_change().dropna()

        stock_ret = rets[i]
        bench_ret = rets[benchmark]

        cov = np.cov(stock_ret, bench_ret)[0][1]
        var = np.var(bench_ret)

        beta = cov / var
        valid_stocks_with_beta .append([i,float(np.round(beta, 5))])
    return valid_stocks_with_beta


def beta_filtration(valid_tickers):
    x = beta_calculate(valid_tickers)
    final = []
    remaining = []
    for i in x:
        if 0.8 <= i[1] <= 1.2:
            final.append(i)
        else:
            remaining.append(i)
    return remaining, final


def main():
    tickers_list = read_csv("Tickers.csv")
    valid, invalid = check_ticker(tickers_list)

    x,y = beta_filtration(valid)

    print("Valid:", valid)
    print()
    
    print("Invalid:", invalid)
    print()

    print(x)

    print()
    print(y)

if __name__ == "__main__":
    main()



