from services.stocks import get_stock_info
from ml.recomendacion import basic_recommendation, smart_recommendation
from config.db import get_db

def main():
    print("=== ASESOR DE INVERSIONES ===")
    while True:
        ticker = input("\n🔍 Ingresá el símbolo de una acción (o 'salir'): ").upper()
        if ticker.lower() == "salir":
            break

        info = get_stock_info(ticker)
        if info:
            recomendacion = basic_recommendation(info["change"])
            print(f"\n Recomendación básica: {recomendacion}")

            usar_ml = input("\n ¿Querés usar Machine Learning para predecir si conviene comprar? (s/n): ").lower()
            if usar_ml == "s":
                ml_recomendacion = smart_recommendation(ticker, registrar= True)
                print(f"\n📊 Recomendación con Machine Learninig: {ml_recomendacion}")


                decision = input("🧾 ¿Qué hiciste? (compré / no compré / skip): ").strip().lower()
                if decision in ["compré", "no compré"]:
                    db = get_db()
                    db.acciones_usuario.update_one(
                        {"ticker": ticker, "decision_usuario": None},
                        {"$set": {"decision_usuario": decision}}
                    )

if __name__ == "__main__":
    main()
