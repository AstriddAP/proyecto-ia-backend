import os
import gc
# Desactivar restricción estricta de weights_only en PyTorch 2.6+ (para evitar problemas de des-serialización de YOLOv8)
# Solo se cargan pesos de confianza descargados directamente de la web oficial de Ultralytics.
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

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

# Diccionario de traducción de clases COCO (inglés a español)
COCO_SPANISH = {
    "person": "persona",
    "bicycle": "bicicleta",
    "car": "carro/auto",
    "motorcycle": "motocicleta",
    "airplane": "avión",
    "bus": "autobús",
    "train": "tren",
    "truck": "camión",
    "boat": "barco",
    "traffic light": "semáforo",
    "fire hydrant": "hidrante de incendios",
    "stop sign": "señal de pare",
    "parking meter": "parquímetro",
    "bench": "banca",
    "bird": "pájaro",
    "cat": "gato",
    "dog": "perro",
    "horse": "caballo",
    "sheep": "oveja",
    "cow": "vaca",
    "elephant": "elefante",
    "bear": "oso",
    "zebra": "cebra",
    "giraffe": "jirafa",
    "backpack": "mochila",
    "umbrella": "paraguas",
    "handbag": "cartera/bolso",
    "tie": "corbata",
    "suitcase": "maleta",
    "frisbee": "disco volador",
    "skis": "esquís",
    "snowboard": "tabla de nieve",
    "sports ball": "balón deportivo",
    "kite": "cometa/papalote",
    "baseball bat": "bate de béisbol",
    "baseball glove": "guante de béisbol",
    "skateboard": "patineta",
    "surfboard": "tabla de surf",
    "tennis racket": "raqueta de tenis",
    "bottle": "botella",
    "wine glass": "copa de vino",
    "cup": "taza/vaso",
    "fork": "tenedor",
    "knife": "cuchillo",
    "spoon": "cuchara",
    "bowl": "tazón/plato hondo",
    "banana": "plátano/banano",
    "apple": "manzana",
    "sandwich": "sándwich",
    "orange": "naranja",
    "broccoli": "brócoli",
    "carrot": "zanahoria",
    "hot dog": "perro caliente/pancho",
    "pizza": "pizza",
    "donut": "dona",
    "cake": "pastel/torta",
    "chair": "silla",
    "couch": "sofá",
    "potted plant": "planta en maceta",
    "bed": "cama",
    "dining table": "mesa de comedor",
    "toilet": "inodoro/taza de baño",
    "tv": "televisor",
    "laptop": "computadora portátil",
    "mouse": "mouse/ratón",
    "remote": "control remoto",
    "keyboard": "teclado",
    "cell phone": "teléfono celular",
    "microwave": "microondas",
    "oven": "horno",
    "toaster": "tostadora",
    "sink": "fregadero/lavabo",
    "refrigerator": "refrigerador",
    "book": "libro",
    "clock": "reloj",
    "vase": "florero/jarrón",
    "scissors": "tijeras",
    "teddy bear": "oso de peluche",
    "hair drier": "secador de pelo",
    "toothbrush": "cepillo de dientes"
}

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

        original_height, original_width = image.shape[:2]
        print(f"[LOG RENDER] Imagen decodificada con éxito. Dimensiones originales: {original_width}x{original_height} (WxH)")

        # Redimensionamos la imagen para ahorrar memoria RAM y acelerar la inferencia en CPU (Render Free tiene límite de 512MB)
        # Nota: YOLO trabaja a 640px por defecto, por lo que procesar imágenes grandes en CPU consume RAM innecesariamente.
        max_dimension = 640
        if max(original_height, original_width) > max_dimension:
            scale = max_dimension / max(original_height, original_width)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"[LOG RENDER] Imagen redimensionada para ahorro de RAM a: {new_width}x{new_height} (WxH)")
        else:
            new_width, new_height = original_width, original_height

        # 5. Detección con IA Real (YOLO) o Simulación Robusta
        if YOLO_MODEL is not None:
            print("[LOG RENDER] Procesando imagen con modelo YOLOv8n...")
            
            # Ejecutamos la predicción en la imagen OpenCV redimensionada
            results = YOLO_MODEL(image)
            
            predictions = []
            detected_labels = []

            # Factores de escala para regresar las coordenadas al tamaño original de la foto
            scale_x = original_width / new_width
            scale_y = original_height / new_height

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    label_en = YOLO_MODEL.names[class_id]
                    # Traducir etiqueta a español si está disponible
                    label_es = COCO_SPANISH.get(label_en, label_en)
                    confidence = float(box.conf[0])
                    coords = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax] en tamaño redimensionado
                    
                    predictions.append({
                        "label": label_es,
                        "confidence": round(confidence, 3),
                        "bounding_box": {
                            "xmin": int(coords[0] * scale_x),
                            "ymin": int(coords[1] * scale_y),
                            "xmax": int(coords[2] * scale_x),
                            "ymax": int(coords[3] * scale_y)
                        }
                    })
                    detected_labels.append(f"{label_es} ({int(confidence * 100)}%)")

            # Formateamos una respuesta de texto simple para fácil parseo
            if detected_labels:
                result_summary = f"Detección exitosa: {', '.join(detected_labels)}"
            else:
                result_summary = "No se detectó ningún objeto común en la imagen."

            processing_time = round(time.time() - start_time, 3)
            print(f"[LOG RENDER] Análisis completado con éxito en {processing_time} segundos.")

            # Liberamos memoria
            del image
            del results
            gc.collect()

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
            
            # Liberamos memoria
            del image
            gc.collect()

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
