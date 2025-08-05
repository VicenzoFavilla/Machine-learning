# ml/trainer_optimizado.py
import yfinance as yf
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_buy_model_optimizado(ticker="AAPL", periodo="2y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=periodo)

    df["Return"] = df["Close"].pct_change() #retorno diario procedural
    df["MA5"] = df["Close"].rolling(5).mean() # media movil 5 dias
    df["MA10"] = df["Close"].rolling(10).mean() #media movil 10 dias
    df["Volatility"] = df["Close"].rolling(5).std() # volatilidad 5 dias
    df["Target"] = (df["Close"].shift(-1) > df["Close"] * 1.01).astype(int) # objetivo: si el cierre del dia siguiente es mayor al 1% del cierre actual

    df.dropna(inplace=True)

    X = df[["Open", "High", "Low", "Close", "Volume", "Return", "MA5", "MA10", "Volatility"]]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    model = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo ({ticker}): {accuracy:.2f}")

    return model, accuracy
