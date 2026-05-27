# Usa una imagen oficial de Python ligera para producción
FROM python:3.10-slim

# Evita que Python escriba archivos .pyc en el contenedor
ENV PYTHONDONTWRITEBYTECODE=1

# Asegura que las salidas de consola (stdout y stderr) se envíen directamente al log sin búfer
ENV PYTHONUNBUFFERED=1

# Define el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instala dependencias del sistema necesarias
# (Útil si en el futuro decides añadir OpenCV, Pillow u otras herramientas de IA)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala las dependencias de Python primero para aprovechar la caché de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código de la aplicación
COPY . .

# Expone el puerto por defecto (informativo para Docker)
EXPOSE 8000

# Comando de inicio del servidor ASGI Uvicorn
# Render inyecta dinámicamente la variable de entorno $PORT, por lo que usamos
# shell execution (sh -c) para leerla en tiempo de ejecución, con un fallback al puerto 8000.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
