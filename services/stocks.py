import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from config.db import get_db

def get_stock_info(ticker):
    stock = yf.Ticker(ticker)

    try:
        info = stock.info
    except Exception:
        print("No se pudo obtener la información del ticker.")
        return None

    current_price = info.get("currentPrice")
    previous_close = info.get("previousClose")
    volume = info.get("volume")
    name = info.get("shortName", ticker)

    change = round((current_price - previous_close) / previous_close * 100, 2) if current_price and previous_close else None

    print(f"\n📈 {name} ({ticker.upper()})")
    print(f"Precio actual: ${current_price}")
    print(f"Variación diaria: {change}%")
    print(f"Volumen: {volume}")

    # Historial de 7 días
    hist = stock.history(period="7d")
    if not hist.empty:
        hist["Close"].plot(title=f"Precio de cierre - Últimos 7 días ({ticker.upper()})")
        plt.ylabel("Precio ($)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Guardar en MongoDB
    db = get_db()
    db.history.insert_one({
        "ticker": ticker,
        "name": name,
        "price": current_price,
        "change": change,
        "volume": volume,
        "timestamp": datetime.now()
    })

    return {
        "ticker": ticker,
        "price": current_price,
        "change": change,
        "volume": volume
    }
