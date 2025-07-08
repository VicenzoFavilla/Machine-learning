def basic_recommendation(change):
    if change is None:
        return "No hay suficiente información para recomendar."

    if change < -2:
        return "📉 El precio bajó bastante hoy. Podría ser un buen momento para comprar."
    elif -2 <= change <= 2:
        return "🤔 El precio está estable. Podés observar unos días más."
    else:
        return "🚀 El precio subió bastante. Tal vez sea mejor esperar una baja."
