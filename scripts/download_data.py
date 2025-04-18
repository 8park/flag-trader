import pandas as pd
import ta
import os

def download_msft(start="2020-07-01", end="2020-08-15", input_file="scripts/MSFT_1986-03-13_2025-02-04.csv"):
    try:
        os.makedirs("data", exist_ok=True)
        print("Reading MSFT data from CSV...")
        # CSV 파일 읽기
        df = pd.read_csv(input_file, parse_dates=["Date"])
        # 날짜 필터링
        df = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
        # 필요한 열 선택
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        if df.empty:
            raise ValueError("No data found for the specified date range")
        print(f"Processed {len(df)} rows")
        print("Calculating RSI...")
        # RSI 계산
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi().fillna(0)
        print("Saving to data/MSFT.csv...")
        df.to_csv("data/MSFT.csv", index=False)
        print("Data preparation complete!")
        return df
    except Exception as e:
        print(f"Data processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    download_msft()