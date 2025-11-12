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

def blended_benchmark(start, end):
    data = yf.download(["^GSPC", "^GSPTSE"], start=start, end=end)["Close"]
    rets = data.pct_change().dropna()
    blended = rets.mean(axis=1)
    return blended.rename("Benchmark")
    
def score_data(valid_tickers):
    start="2025-05-15"
    end="2025-11-15"


    def get_sector_safe(ticker):
        try:
            info = yf.Ticker(ticker).get_info()  # more reliable than .info on newer yfinance
        except Exception:
            info = {}
        sector = (
            info.get("sector")
            or info.get("industry")
            or info.get("categoryName")
            or "Unknown"
        )
        return sector
    
    valid_stocks_with_data = []
    bench = blended_benchmark(start, end)


    bench_vol_ann = float(bench.std() * np.sqrt(252))
    if not np.isfinite(bench_vol_ann) or bench_vol_ann <= 0:
        bench_vol_ann = np.nan
    

    for i in valid_tickers:
        stock_ret = yf.download(i, start=start, end=end, auto_adjust=True)["Close"].pct_change().dropna()

        stock_ret.name = i 
        df = pd.concat([stock_ret, bench], axis=1).dropna()
        if len(df) < 5:
            continue

        var_bench = df["Benchmark"].var()
        if var_bench == 0 or np.isnan(var_bench):
            continue
        # Beta Calculation
        beta = df[i].cov(df["Benchmark"]) / df["Benchmark"].var()

        var_bench = df["Benchmark"].var()
        if var_bench == 0 or np.isnan(var_bench):
            continue

        # Correlation Calculation
        corr = df[i].corr(df["Benchmark"])

        # Volatility Calculation
        vol_ann = float(stock_ret.std() * np.sqrt(252))
        raw_ratio = (vol_ann / bench_vol_ann) if np.isfinite(bench_vol_ann) else np.nan
        sigma_rel = float(raw_ratio / (1 + raw_ratio)) if np.isfinite(raw_ratio) else np.nan

        sector = get_sector_safe(i)


        valid_stocks_with_data.append([i, {'Beta': float(np.round(beta, 5)), 'Correlation': float(np.round(corr, 5)), 'Volatility_Ann': float(np.round(vol_ann, 5)), 'Sigma_Rel': float(np.round(sigma_rel, 5)), 'Sector': sector}])
        
    return valid_stocks_with_data


def score_calculate(valid_tickers):
    x = score_data(valid_tickers)
    final = []
    total_weight = 0.0
    for ticker, m in x:
        beta = m['Beta']
        corr = m['Correlation']
        sigma_rel = m['Sigma_Rel']
        score = 0.5 * (1 - abs(beta - 1)) + 0.3 * (1 - sigma_rel) + 0.2 * corr

        if not np.isfinite(score):
            continue
        score = max(score, 0.0)

        weight = score
        total_weight += weight

        final.append([ticker, float(np.round(score, 5)), weight, m['Sector']])

    for i in final:
        weight_in_percent = (i[2] / total_weight) * 100 if total_weight != 0 else 0.0
        i.append(float(np.round(weight_in_percent, 5)))
    
    return final

'''def beta_filtration(valid_tickers):
    x = score_calculate(valid_tickers)
    final = []
    remaining = []
    for i in x:
        if 0.8 <= i[1] <= 1.2:
            final.append(i)
        else:
            remaining.append(i)
    return remaining, final'''


def compare_portfolio_to_benchmark(portfolio_dict, start, end):
    """
    Takes in a portfolio, compares its return within start and end dates
    to the benchmark average return. For testing purposes.
    """

    tickers = list(portfolio_dict.keys())
    weights = np.array(list(portfolio_dict.values()))

    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)["Close"]

    daily_ret = data.pct_change().dropna().dot(weights)
    portfolio_total = (1 + daily_ret).prod() - 1

    tsx  = yf.download("^GSPTSE", start=start, end=end, progress=False, auto_adjust=True)["Close"]
    spx  = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True)["Close"]


    if isinstance(tsx, pd.DataFrame):
        tsx = tsx.squeeze()
    if isinstance(spx, pd.DataFrame):
        spx = spx.squeeze()

    tsx_total = tsx.iloc[-1] / tsx.iloc[0] - 1
    spx_total = spx.iloc[-1] / spx.iloc[0] - 1
    bench = blended_benchmark(start, end)
    benchmark_total = (1 + bench).prod() - 1


    diff = portfolio_total - benchmark_total

    return {
        "Portfolio_Total_Return_%": round(float(portfolio_total * 100), 2),
        "Benchmark_Avg_Return_%": round(float(benchmark_total * 100), 2),
        "Difference_%": round(float(diff * 100), 2)
    }


def main():
    tickers_list = read_csv("Tickers.csv")
    valid, invalid = check_ticker(tickers_list)

    x = score_calculate(valid)

    print("Valid:", valid)
    print()
    
    print("Invalid:", invalid)
    print()

    print(x)
    print()

if __name__ == "__main__":
    main()



