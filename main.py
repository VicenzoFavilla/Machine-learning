from services.stocks import get_stock_info
from ml.recomendacion import basic_recommendation, smart_recommendation

def main():
    print("=== ASESOR DE INVERSIONES ===")
    while True:
        ticker = input("\n🔍 Ingresá el símbolo de una acción (o 'salir'): ").upper()
        if ticker.lower() == "salir":
            break

        info = get_stock_info(ticker)
        if info:
            recomendacion = basic_recommendation(info["change"])
            print(f"\n🧠 Recomendación básica: {recomendacion}")

            usar_ml = input("\n🤖 ¿Querés usar ML para predecir si conviene comprar? (s/n): ").lower()
            if usar_ml == "s":
                ml_recomendacion = smart_recommendation(ticker)
                print(f"\n📊 Recomendación con ML: {ml_recomendacion}")

if __name__ == "__main__":
    main()
