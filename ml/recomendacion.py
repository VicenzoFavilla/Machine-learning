from ml.trainer import train_buy_model_optimizado
from config.db import get_db
from datetime import datetime
import yfinance as yf
import pandas as pd
import joblib
import io
from bson.binary import Binary


def basic_recommendation(change):
    if change is None:
        return "No hay suficiente información para recomendar."

    if change < -2:
        return "📉 El precio bajó bastante hoy. Podría ser un buen momento para comprar."
    elif -2 <= change <= 2:
        return "🤔 El precio está estable. Podés observar unos días más."
    else:
        return "🚀 El precio subió bastante. Tal vez sea mejor esperar una baja."

def guardar_modelo_en_mongo(ticker, modelo):
    buffer = io.BytesIO()
    joblib.dump(modelo, buffer)
    buffer.seek(0)

    db = get_db()
    db.modelos_binarios.replace_one(
        {"ticker": ticker},
        {
            "ticker": ticker,
            "modelo": Binary(buffer.read()),
            "ultima_actualizacion": datetime.now()
        },
        upsert=True
    )
    print(f"[INFO] Modelo para {ticker} guardado en MongoDB.")

def cargar_modelo_de_mongo(ticker):
    db = get_db()
    doc = db.modelos_binarios.find_one({"ticker": ticker})
    if doc and "modelo" in doc:
        buffer = io.BytesIO(doc["modelo"])
        modelo = joblib.load(buffer)
        print(f"[INFO] Modelo para {ticker} cargado desde MongoDB.")
        return modelo
    return None

def smart_recommendation(ticker="AAPL", registrar=False):
    db = get_db()
    accuracy = None

    model = cargar_modelo_de_mongo(ticker)
    if model is None:
        try:
            model, accuracy = train_buy_model_optimizado(ticker)
        except Exception as e:
            return f"No se pudo entrenar el modelo para {ticker}. Error: {e}"
        guardar_modelo_en_mongo(ticker, model)

    if accuracy is not None and accuracy < 0.80:
        return f"⚠️ La precisión del modelo es del {accuracy:.2f}.\nNo es recomendable comprar basándose en esta predicción."

    stock = yf.Ticker(ticker)
    data = stock.history(period="2y")

    if data.empty or len(data) < 15:
        return "No hay suficientes datos para predecir."

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

    row = df.iloc[-1][[
        "Open", "High", "Low", "Close", "Volume",
        "Return", "MA5", "MA10", "Volatility"
    ]]
    row_df = pd.DataFrame([row])

    try:
        pred = model.predict(row_df)[0]
    except Exception as e:
        return f"Error al predecir: {e}"

    recomendacion = "comprar" if pred == 1 else "no_comprar"

    if registrar:
        db.acciones_usuario.insert_one({
            "ticker": ticker,
            "fecha": datetime.now(),
            "precio": row["Close"],
            "recomendacion_ml": recomendacion,
            "decision_usuario": None,
            "modelo_usado": "xgboost_v2_mongo",
            "contexto": "modelo optimizado con indicadores tecnicos",
            "parametros_modelo": {
                "n_estimators": 150,
                "learning_rate": 0.05,
                "max_depth": 5
            }
        })

    # Registrar uso del modelo
    db.modelos_uso.update_one(
        {"ticker": ticker},
        {
            "$inc": {"veces_usado": 1},
            "$set": {"ultima_vez": datetime.now()}
        },
        upsert=True
    )

    if pred == 1:
        return "🟢 El modelo predice que el precio subirá. Podría ser buen momento para comprar."
    else:
        return "🔴 El modelo predice que el precio bajará. Mejor esperar."
    return "No hay suficiente información para recomendar."