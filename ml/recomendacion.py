from ml.trainer import train_buy_model
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
    model = train_buy_model(ticker)
    data = yf.Ticker(ticker).history(period="1d")
    
    if data.empty:
        return "No hay datos recientes para predecir."

    latest = data.iloc[-1]
    
    row = pd.DataFrame([{
        "Open": latest["Open"],
        "High": latest["High"],
        "Low": latest["Low"],
        "Close": latest["Close"],
        "Volume": latest["Volume"]
    }])

    pred = model.predict(row)[0]
    recomendacion = "comprar" if pred == 1 else "no_comprar"

    if registrar:
        db = get_db()
        db.acciones_usuario.insert_one({
            "ticker": ticker,
            "fecha": datetime.now(),
            "precio": latest["Close"],
            "recomendacion_ml": recomendacion,
            "decision_usuario": None,  # luego lo podés actualizar
            "modelo_usado": "random_forest",
            "contexto": "modelo inicial sin feedback"
        })

    if pred == 1:
        return "🟢 El modelo predice que el precio subirá. Podría ser buen momento para comprar."
    else:
        return "🔴 El modelo predice que el precio bajará. Mejor esperar."