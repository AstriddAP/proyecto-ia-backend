import time
from typing import Dict, Any
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Inicialización de la aplicación FastAPI
app = FastAPI(
    title="AI Image Analysis Robust Backend",
    description="Backend en Python con FastAPI, OpenCV y soporte YOLO para análisis de imágenes.",
    version="2.0.0"
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, restringir según sea necesario
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tipos de imágenes permitidos
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

# Carga diferida de YOLO (Ultralytics) con manejo de fallos para entornos con recursos limitados (Render Free)
YOLO_MODEL = None
try:
    from ultralytics import YOLO
    # Cargamos el modelo YOLOv8n (Nano), que es ligero (~6MB) y consume menos RAM
    YOLO_MODEL = YOLO("yolov8n.pt")
    print("[INIT] Modelo YOLOv8n cargado correctamente para detección real.")
except ImportError:
    print("[INIT] Librería 'ultralytics' no encontrada. El servidor funcionará en MODO SIMULADO.")
except Exception as e:
    print(f"[INIT] No se pudo cargar el modelo YOLO ({e}). El servidor funcionará en MODO SIMULADO.")


@app.get("/", tags=["General"])
async def root():
    """
    Endpoint de Health Check para verificar el estado de la API.
    """
    return {
        "status": "online",
        "service": "AI Image Analysis API",
        "yolo_active": YOLO_MODEL is not None
    }


@app.post(
    "/analizar-imagen/", 
    status_code=status.HTTP_200_OK,
    tags=["Análisis de IA"],
    summary="Analiza una imagen y devuelve los resultados de detección (YOLO o simulación robusta)."
)
async def analizar_imagen(file: UploadFile = File(...)):
    # 1. Logs de entrada en consola (visibles en los logs en tiempo real de Render)
    print(f"\n[LOG RENDER] >>> Solicitud entrante recibida!")
    print(f"[LOG RENDER] Nombre del archivo: {file.filename}")
    print(f"[LOG RENDER] Tipo de contenido MIME: {file.content_type}")

    # 2. Validación de presencia del archivo
    if not file:
        print("[LOG RENDER] Error: Archivo vacío o no suministrado.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se ha proporcionado ningún archivo."
        )

    # 3. Validación de tipo MIME
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        print(f"[LOG RENDER] Error: Tipo de archivo no permitido ({file.content_type}).")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tipo de archivo no permitido: {file.content_type}. Solo se aceptan imágenes (JPEG, PNG, WEBP, GIF)."
        )

    try:
        start_time = time.time()

        # 4. Procesamiento y decodificación de imagen limpia usando OpenCV y NumPy
        # Leemos los bytes crudos del archivo enviado desde Android
        image_bytes = await file.read()
        
        # Convertimos los bytes a un array de una dimensión de NumPy
        np_array = np.frombuffer(image_bytes, np.uint8)
        
        # Decodificamos la imagen a una matriz OpenCV (formato BGR por defecto)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # Si OpenCV devuelve None, la imagen es corrupta o inválida
        if image is None:
            print("[LOG RENDER] Error: OpenCV no pudo decodificar la imagen.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Error al procesar la imagen: los datos están corruptos o el formato no es válido."
            )

        print(f"[LOG RENDER] Imagen decodificada con éxito. Dimensiones: {image.shape[1]}x{image.shape[0]} (WxH)")

        # 5. Detección con IA Real (YOLO) o Simulación Robusta
        if YOLO_MODEL is not None:
            print("[LOG RENDER] Procesando imagen con modelo YOLOv8n...")
            
            # Ejecutamos la predicción en la imagen OpenCV decodificada
            results = YOLO_MODEL(image)
            
            predictions = []
            detected_labels = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    label = YOLO_MODEL.names[class_id]
                    confidence = float(box.conf[0])
                    coords = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
                    
                    predictions.append({
                        "label": label,
                        "confidence": round(confidence, 3),
                        "bounding_box": {
                            "xmin": int(coords[0]),
                            "ymin": int(coords[1]),
                            "xmax": int(coords[2]),
                            "ymax": int(coords[3])
                        }
                    })
                    detected_labels.append(f"{label} ({int(confidence * 100)}%)")

            # Formateamos una respuesta de texto simple para fácil parseo
            if detected_labels:
                result_summary = f"Detección exitosa: {', '.join(detected_labels)}"
            else:
                result_summary = "No se detectó ningún objeto común en la imagen."

            processing_time = round(time.time() - start_time, 3)
            print(f"[LOG RENDER] Análisis completado con éxito en {processing_time} segundos.")

            return {
                "status": "success",
                "result": result_summary,
                "predictions": predictions,
                "processing_time": processing_time
            }

        else:
            # Simulación Robusta en caso de que YOLO no esté disponible
            print("[LOG RENDER] Ejecutando simulación de IA...")
            time.sleep(0.5)  # Simula un ligero retraso de cómputo
            
            processing_time = round(time.time() - start_time, 3)
            print(f"[LOG RENDER] Simulación finalizada en {processing_time} segundos.")
            
            # Formato JSON estructurado simple y estándar que previene excepciones en Android
            return {
                "status": "success",
                "result": "Laptop detectada correctamente (Simulado)",
                "predictions": [
                    {
                        "label": "laptop",
                        "confidence": 0.950,
                        "bounding_box": {"xmin": 120, "ymin": 180, "xmax": 420, "ymax": 460}
                    }
                ],
                "processing_time": processing_time
            }

    except HTTPException as http_exc:
        # Re-lanzar excepciones HTTP ya controladas
        raise http_exc
    except Exception as e:
        # Captura cualquier excepción de procesamiento interno para que el backend no colapse
        print(f"[LOG RENDER] Excepción interna crítica: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en el servidor al procesar la imagen: {str(e)}"
        )
