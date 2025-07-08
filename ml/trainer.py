import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_buy_model(ticker="AAPL"):
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo")  # últimos 6 meses
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)

    df.dropna(inplace=True)

    X = df[["Open", "High", "Low", "Close", "Volume"]]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Precisión del modelo ({ticker}): {score:.2f}")

    return model
