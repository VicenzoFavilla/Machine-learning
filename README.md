# Asesor de Inversiones con Machine Learning

Este proyecto es una aplicación interactiva que recomienda acciones y ayuda a tomar decisiones de inversión utilizando técnicas de Machine Learning y datos financieros en tiempo real.

## Características

- Consulta de información de acciones por símbolo (ticker).
- Recomendación básica basada en el cambio de precio.
- Recomendación avanzada usando modelos de Machine Learning.
- Registro de decisiones del usuario en MongoDB.
- Entrenamiento y persistencia de modelos ML.

## Estructura del proyecto

```
proyecto-ml/
│
├── config/              # Configuración de la base de datos
├── data/models/         # Modelos entrenados guardados
├── ml/                  # Lógica de Machine Learning
├── services/            # Servicios externos (APIs)
├── tests/               # Pruebas unitarias
├── main.py              # Script principal
├── requirements.txt     # Dependencias
└── README.md            # Documentación
```

## Instalación

1. Clona el repositorio:
   ```sh
   git clone <URL-del-repo>
   cd proyecto-ml/Machine-learning
   ```

2. Instala las dependencias:
   ```sh
   pip install -r requirements.txt
   ```

3. Asegúrate de tener MongoDB corriendo en tu máquina.

## Uso

Ejecuta el programa principal:
```sh
python main.py
```

Sigue las instrucciones en pantalla para consultar acciones y registrar tus decisiones.

## Entrenamiento de modelos

Los modelos se entrenan automáticamente al consultar una acción y se guardan en `data/models/`.

## Pruebas

Para ejecutar las pruebas unitarias:
```sh
pytest tests/
```

## Contribuciones

¡Las contribuciones son bienvenidas! Por favor, abre un issue o envía un pull request.

## Licencia

MIT