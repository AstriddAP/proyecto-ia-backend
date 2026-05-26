import time
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Inicialización de la aplicación FastAPI
app = FastAPI(
    title="AI Image Analysis Backend",
    description="Backend independiente en Python usando FastAPI para el análisis de imágenes con IA.",
    version="1.0.0"
)

# Configuración de CORS (Cross-Origin Resource Sharing)
# Permite que aplicaciones móviles (Android) o web se conecten sin restricciones de dominio en desarrollo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, se recomienda restringir a dominios o IPs específicas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lista de tipos MIME permitidos para validación de imágenes
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

def mock_ai_analysis(filename: str) -> Dict[str, Any]:
    """
    Simulación de procesamiento de imagen usando un modelo de Inteligencia Artificial (por ejemplo, detección de objetos).
    
    Args:
        filename (str): Nombre del archivo procesado.
        
    Returns:
        dict: Estructura de datos simulando las predicciones de un modelo de Deep Learning.
    """
    # Simulamos un pequeño retraso que tomaría procesar la imagen con una red neuronal (p. ej. YOLO, ResNet)
    time.sleep(1.2) 
    
    # Respuesta simulada basada en detección de objetos y clasificación
    return {
        "success": True,
        "message": "Imagen analizada exitosamente por el motor de IA.",
        "metadata": {
            "filename": filename,
            "timestamp": time.time(),
            "processing_time_sec": 1.2
        },
        "predictions": [
            {
                "label": "Persona",
                "confidence": 0.985,
                "bounding_box": {"xmin": 100, "ymin": 150, "xmax": 300, "ymax": 500}
            },
            {
                "label": "Teléfono Celular",
                "confidence": 0.912,
                "bounding_box": {"xmin": 250, "ymin": 300, "xmax": 320, "ymax": 420}
            },
            {
                "label": "Computadora Portátil",
                "confidence": 0.874,
                "bounding_box": {"xmin": 400, "ymin": 200, "xmax": 700, "ymax": 600}
            }
        ],
        "summary": "Se detectaron 3 objetos principales en la imagen con altos niveles de confianza."
    }

@app.get("/", tags=["General"])
async def root():
    """
    Endpoint de Health Check para verificar que el servicio está activo.
    """
    return {
        "status": "online",
        "service": "AI Image Analysis API",
        "version": "1.0.0"
    }

@app.post(
    "/analizar-imagen/", 
    status_code=status.HTTP_200_OK,
    tags=["Análisis de IA"],
    summary="Analiza una imagen enviada por el cliente y devuelve predicciones simuladas de IA."
)
async def analizar_imagen(file: UploadFile = File(...)):
    """
    Endpoint que recibe una imagen mediante POST (multipart/form-data), realiza la validación de formato
    y retorna los resultados simulados de la Inteligencia Artificial.
    """
    # 1. Validación de seguridad básica: verificar que se haya recibido un archivo
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se ha proporcionado ningún archivo."
        )

    # 2. Validación de tipo de archivo (MIME Type) para asegurar que es una imagen
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tipo de archivo no permitido: {file.content_type}. Solo se aceptan imágenes (JPEG, PNG, WEBP, GIF)."
        )

    try:
        # Nota para el desarrollo futuro con IA real (TensorFlow, PyTorch, OpenCV, Hugging Face, etc.):
        # -----------------------------------------------------------------------------------------
        # content = await file.read()  # Lee los bytes del archivo cargado
        # image = Image.open(io.BytesIO(content))  # Carga la imagen usando PIL (Pillow)
        # result = mi_modelo_ia.predict(image)  # Pasa la imagen a tu modelo
        
        # Ejecutamos la simulación de análisis
        resultado = mock_ai_analysis(file.filename)
        return resultado

    except Exception as e:
        # Manejo genérico de errores durante el procesamiento de la imagen en el backend
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ocurrió un error interno al procesar la imagen: {str(e)}"
        )
