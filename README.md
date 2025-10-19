# Asesor de Inversiones con Machine Learning

Aplicación de consola que recomienda acciones y ayuda a tomar decisiones de inversión usando técnicas de Machine Learning y datos financieros.

## Características

- Consulta información de acciones por símbolo (ticker).
- Recomendación básica basada en variación diaria del precio.
- Recomendación avanzada con modelos de Machine Learning.
- Registro de decisiones del usuario en MongoDB.
- Entrenamiento y persistencia de modelos ML.

## Estructura del proyecto

```
Machine-learning/
  config/            # Configuración de la base de datos
  ml/                # Lógica de Machine Learning
  models/            # Modelos entrenados guardados (si se usa filesystem)
  services/          # Integraciones externas (APIs)
  main.py            # Script principal (CLI)
  requirements.txt   # Dependencias
  README.md          # Documentación
```

## Instalación

1) Clona el repositorio y entra en el directorio del proyecto:

```
git clone <URL-del-repo>
cd proyecto-ml/Machine-learning
```

2) Instala dependencias:

```
pip install -r requirements.txt
```

3) Asegúrate de tener MongoDB corriendo en tu máquina (por defecto en `mongodb://localhost:27017/`).

## Uso

Ejecuta el programa principal:

```
python main.py
```

Sigue las instrucciones para consultar acciones, ver el gráfico y registrar decisiones.

## Entrenamiento de modelos

Los modelos se entrenan automáticamente al consultar una acción (si no existen) y pueden guardarse en MongoDB o en `models/`.

## Contribuciones

¡Las contribuciones son bienvenidas! Abre un issue o envía un pull request.

## Licencia

MIT

