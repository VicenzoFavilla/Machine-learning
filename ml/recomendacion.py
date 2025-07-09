from ml.trainer import train_buy_model_optimizado
from config.db import get_db
from datetime import datetime
import yfinance as yf
import pandas as pd

def basic_recommendation(change):
    if change is None:
        return "No hay suficiente información para recomendar."

    if change < -2:
        return "📉 El precio bajó bastante hoy. Podría ser un buen momento para comprar."
    elif -2 <= change <= 2:
        return "🤔 El precio está estable. Podés observar unos días más."
    else:
        return "🚀 El precio subió bastante. Tal vez sea mejor esperar una baja."


def smart_recommendation(ticker="AAPL", registrar=False):
    model, accuracy = train_buy_model_optimizado(ticker)
    if accuracy < 0.70:
        return f"⚠️ La precisión del modelo es del {accuracy:.2f}.\nNo es recomendable comprar basándose en esta predicción."

    stock = yf.Ticker(ticker)
    data = stock.history(period="2y")  # suficiente para cálculos

    if data.empty or len(data) < 15:
        return "No hay suficientes datos para predecir."

    # Crear los mismos indicadores técnicos usados en el entrenamiento
    df = data.copy()
    df["Return"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["Volatility"] = df["Close"].rolling(5).std()
    df["EMA12"] = df["Close"].ewm(span=12).mean()
    df["EMA26"] = df["Close"].ewm(span=26).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["RSI"] = 100 - (100 / (1 + df["Return"].rolling(14).mean() / df["Return"].rolling(14).std()))
    df["Momentum"] = df["Close"] - df["Close"].shift(5)

    df.dropna(inplace=True)
    if df.empty:
        return "No hay suficientes datos válidos después del procesamiento."

    row = df.iloc[-1][["Open", "High", "Low", "Close", "Volume", "Return", "MA5", "MA10", "Volatility"]]
    row_df = pd.DataFrame([row])

    try:
        pred = model.predict(row_df)[0]
    except Exception as e:
        return f"Error al predecir: {e}"

    recomendacion = "comprar" if pred == 1 else "no_comprar"

    if registrar:
        db = get_db()
        db.acciones_usuario.insert_one({
            "ticker": ticker,
            "fecha": datetime.now(),
            "precio": row["Close"],
            "recomendacion_ml": recomendacion,
            "decision_usuario": None,
            "modelo_usado": "xgboost_v2",
            "contexto": "modelo optimizado con indicadores tecnicos"
        })

    if pred == 1:
        return "🟢 El modelo predice que el precio subirá. Podría ser buen momento para comprar."
    else:
        return "🔴 El modelo predice que el precio bajará. Mejor esperar."
