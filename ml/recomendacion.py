from ml.trainer import train_buy_model
import yfinance as yf

def basic_recommendation(change):
    if change is None:
        return "No hay suficiente información para recomendar."

    if change < -2:
        return "📉 El precio bajó bastante hoy. Podría ser un buen momento para comprar."
    elif -2 <= change <= 2:
        return "🤔 El precio está estable. Podés observar unos días más."
    else:
        return "🚀 El precio subió bastante. Tal vez sea mejor esperar una baja."


def smart_recommendation(ticker="AAPL"):
    model = train_buy_model(ticker)
    data = yf.Ticker(ticker).history(period="1d")
    
    if data.empty:
        return "No hay datos recientes para predecir."

    latest = data.iloc[-1]
    row = [[
        latest["Open"], latest["High"],
        latest["Low"], latest["Close"], latest["Volume"]
    ]]
    
    pred = model.predict(row)[0]

    if pred == 1:
        return "🟢 El modelo predice que el precio subirá. Podría ser buen momento para comprar."
    else:
        return "🔴 El modelo predice que el precio bajará. Mejor esperar."
