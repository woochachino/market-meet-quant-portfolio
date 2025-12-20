## About

This repository contains **Team 13’s submission** to the **CFM 101 annual team competition** at the **University of Waterloo**.  
The project placed **1st overall**, achieving a **3.2% portfolio return over one week** during a **3.6% benchmark return period**.


# Market Meet — Quantitative Portfolio Construction

A benchmark-relative quantitative portfolio construction system built in Python.

This project implements a **systematic, rules-based equity portfolio pipeline** that selects, scores, and allocates stocks based on their risk characteristics relative to a blended market benchmark. The focus is on **risk control, diversification, and evaluation discipline**, rather than short-term alpha prediction.

The system was originally developed in a team setting and has since been refactored into a **configurable, standalone portfolio builder**.


---

## Key Features

- Ticker validation and liquidity filtering
- Blended benchmark construction (S&P 500 + TSX)
- Rolling risk estimation (beta and correlation)
- Multi-factor scoring and portfolio weighting
- Defensive asset overlay
- Sector, position, and currency constraints
- Fully configurable via command-line arguments
- Notebook-based walkthrough for explanation and visualization

---

## Methodology Overview

### 1. Ticker Validation
Candidate tickers are filtered based on:
- Availability of historical price data
- Average daily trading volume
- Market listing (U.S. or Canadian equities)

This ensures that only liquid, tradable securities are considered.

---

### 2. Benchmark Construction
A **blended benchmark** is constructed using:
- S&P 500 (`^GSPC`)
- TSX Composite (`^GSPTSE`)

Daily benchmark returns are computed as the equal-weight average of both indices.

---

### 3. Risk Estimation
For each valid ticker, the system estimates:
- **Rolling beta** relative to the blended benchmark
- **Rolling correlation** with the benchmark
- **Annualized volatility**
- **Relative volatility penalty** compared to benchmark volatility

Rolling windows are used to emphasize recent behavior and reduce noise.

---

### 4. Scoring Model
Each stock is scored based on its distance from an “ideal” benchmark-relative profile:
- Beta close to 1
- High correlation with the benchmark
- Controlled relative volatility

A composite score is computed and normalized to produce portfolio weights.

---
### 5. Portfolio Construction & Constraints
The raw portfolio is refined through multiple constraint layers:
- Maximum position weight
- Maximum sector exposure
- Minimum and maximum number of holdings
- Minimum position size
- Defensive asset allocation (utilities, healthcare, consumer defensive)
- CAD / USD exposure balancing
- Market capitalization diversity

Weights are normalized after each step to ensure full investment.

---

## Repository Structure

```
.
├── main.py
├── notebooks/
│   └── market_meet_report.ipynb
├── requirements.txt
├── Tickers.csv
└── README.md
```

---


## Usage

Prepare a CSV file containing one ticker per line:

```
AAPL
MSFT
NVDA
TD.TO
ENB.TO
```

Run the portfolio builder:

```bash
python main.py --tickers Tickers.csv
```

Example with custom parameters:

```bash
python main.py \
  --tickers Tickers.csv \
  --portfolio-cad 250000 \
  --start 2025-01-01 \
  --end 2025-06-01 \
  --max-position 10 \
  --max-sector 30 \
  --max-holdings 20 \
  --output-name portfolio \
  --outdir outputs
```

---

## Outputs

- `portfolio_full.csv` — full portfolio with prices, weights, and share counts
- `orders.csv` — simplified order list (ticker and shares)



---

## Summary

This project demonstrates systematic portfolio construction, risk-aware allocation, practical constraint handling, and clean separation between research and execution logic. It is intended as a foundational quantitative finance project showcasing both analytical reasoning and software engineering practices.
