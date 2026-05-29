import os
import gc
import time
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import google.generativeai as genai
import pypdf

# Inicialización de la aplicación FastAPI
app = FastAPI(
    title="AI Image Analysis Robust Backend (Gemini)",
    description="Backend en Python con FastAPI y Google Gemini para descripción visual de imágenes.",
    version="3.0.0"
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tipos de imágenes permitidos
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

# Carga y configuración de Gemini API con manejo de fallos
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_ACTIVE = False

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_ACTIVE = True
        print("[INIT] Google Gemini API configurada correctamente para descripción en tiempo real.")
    except Exception as e:
        print(f"[INIT] No se pudo configurar la API de Gemini: {e}")
else:
    print("[INIT] Variable 'GEMINI_API_KEY' no encontrada. El servidor funcionará en MODO SIMULADO.")


@app.get("/", tags=["General"])
async def root():
    """
    Endpoint de Health Check para verificar el estado de la API.
    """
    return {
        "status": "online",
        "service": "AI Image Analysis API (Gemini)",
        "gemini_active": GEMINI_ACTIVE
    }


@app.get("/modelos/", tags=["General"])
async def list_models():
    """
    Endpoint temporal de depuración para listar los modelos de Gemini disponibles para esta API Key.
    """
    if not GEMINI_ACTIVE:
        return {"status": "inactive", "error": "Gemini no está configurado (falta GEMINI_API_KEY)."}
    try:
        models = [m.name for m in genai.list_models()]
        return {"status": "active", "models": models}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post(
    "/analizar-imagen/", 
    status_code=status.HTTP_200_OK,
    tags=["Análisis de IA"],
    summary="Analiza una imagen y devuelve una descripción descriptiva usando Google Gemini."
)
async def analizar_imagen(file: UploadFile = File(...)):
    # 1. Logs de entrada
    print(f"\n[LOG RENDER] >>> Solicitud entrante recibida!")
    print(f"[LOG RENDER] Nombre del archivo: {file.filename}")
    print(f"[LOG RENDER] Tipo de contenido MIME: {file.content_type}")

    # 2. Validación del archivo
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

        # 4. Decodificación de la imagen a PIL
        image_bytes = await file.read()
        
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            # Si la imagen tiene canal alpha (RGBA/LA), convertir a RGB
            if pil_image.mode in ('RGBA', 'LA'):
                pil_image = pil_image.convert('RGB')
        except Exception as e:
            print(f"[LOG RENDER] Error: PIL no pudo decodificar la imagen: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Error al procesar la imagen: los datos están corruptos o el formato no es válido."
            )

        print(f"[LOG RENDER] Imagen procesada con éxito. Dimensiones: {pil_image.width}x{pil_image.height} (WxH)")

        # 5. Generación de descripción con Gemini o Modo Simulado
        if GEMINI_ACTIVE:
            print("[LOG RENDER] Enviando imagen a Gemini...")
            
            # Usamos gemini-2.5-flash que es rápido y optimizado para visión
            model = genai.GenerativeModel("gemini-2.5-flash")
            
            prompt = (
                "Describe de forma muy concisa en español (máximo 1 o 2 oraciones sencillas) "
                "lo que ves en la imagen, enfocándote en los objetos principales que están en primer plano. "
                "Esta descripción se leerá en voz alta para asistir a personas con discapacidad visual, "
                "por lo que debe ser amigable, natural y directa. Ejemplo: 'Hay un teléfono celular negro y "
                "unos lentes con montura negra sobre la mesa'."
            )
            
            response = model.generate_content([prompt, pil_image])
            result_summary = response.text.strip()
            
            processing_time = round(time.time() - start_time, 3)
            print(f"[LOG RENDER] Análisis Gemini completado en {processing_time} segundos.")
            
            # Liberamos memoria explícitamente
            del pil_image
            gc.collect()

            return {
                "status": "success",
                "result": result_summary,
                "predictions": [],
                "processing_time": processing_time
            }
            
        else:
            # Simulación Robusta
            print("[LOG RENDER] Ejecutando simulación de IA (Gemini inactivo)...")
            time.sleep(0.5)
            
            processing_time = round(time.time() - start_time, 3)
            print(f"[LOG RENDER] Simulación finalizada en {processing_time} segundos.")
            
            # Liberamos memoria
            del pil_image
            gc.collect()

            return {
                "status": "success",
                "result": "Un teléfono celular y unos lentes de color negro sobre la mesa (Simulado)",
                "predictions": [],
                "processing_time": processing_time
            }

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[LOG RENDER] Excepción interna crítica: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en el servidor al procesar la imagen: {str(e)}"
        )


@app.post(
    "/resumir-texto/",
    status_code=status.HTTP_200_OK,
    tags=["Resumen de IA"],
    summary="Resume un texto largo usando Google Gemini."
)
async def resumir_texto(data: dict):
    """
    Recibe un JSON con la clave 'texto' (o 'text') y devuelve un resumen conciso en español.
    """
    text = data.get("texto") or data.get("text")
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se proporcionó ningún texto para resumir. Debe incluir la clave 'texto' o 'text'."
        )

    print(f"\n[LOG RENDER] >>> Solicitud de resumen de texto recibida!")
    print(f"[LOG RENDER] Longitud del texto: {len(text)} caracteres.")

    start_time = time.time()

    try:
        if GEMINI_ACTIVE:
            model = genai.GenerativeModel("gemini-2.5-flash")
            prompt = (
                "Resume de forma clara, natural y concisa el siguiente texto en español, "
                "en un párrafo corto de máximo 3 o 4 oraciones. "
                "Este resumen será leído en voz alta, por lo que debe ser fluido y directo:\n\n"
                f"{text}"
            )
            response = model.generate_content(prompt)
            summary = response.text.strip()

            processing_time = round(time.time() - start_time, 3)
            print(f"[LOG RENDER] Resumen de texto completado en {processing_time} segundos.")

            return {
                "status": "success",
                "resumen": summary,
                "processing_time": processing_time
            }
        else:
            print("[LOG RENDER] Ejecutando simulación de IA para resumen de texto...")
            time.sleep(0.5)
            processing_time = round(time.time() - start_time, 3)
            return {
                "status": "success",
                "resumen": "Este es un resumen simulado de texto. La API de Gemini no está configurada, por lo que el servidor funciona en modo de prueba local.",
                "processing_time": processing_time
            }
    except Exception as e:
        print(f"[LOG RENDER] Error interno en /resumir-texto/: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno al resumir el texto: {str(e)}"
        )


@app.post(
    "/resumir-archivo/",
    status_code=status.HTTP_200_OK,
    tags=["Resumen de IA"],
    summary="Resume un archivo (Imagen, PDF o Texto) usando Google Gemini."
)
async def resumir_archivo(file: UploadFile = File(...)):
    """
    Recibe un archivo (Imagen, TXT o PDF) y devuelve un resumen conciso en español.
    """
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No se ha proporcionado ningún archivo."
        )

    print(f"\n[LOG RENDER] >>> Solicitud de resumen de archivo recibida!")
    print(f"[LOG RENDER] Nombre del archivo: {file.filename}")
    print(f"[LOG RENDER] Tipo MIME: {file.content_type}")

    start_time = time.time()
    file_bytes = await file.read()

    # Identificar el tipo de archivo y extraer su contenido para Gemini
    content_type = file.content_type
    filename_lower = file.filename.lower()

    is_image = content_type in ALLOWED_IMAGE_TYPES or filename_lower.endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif'))
    is_pdf = content_type == "application/pdf" or filename_lower.endswith('.pdf')
    is_txt = content_type.startswith("text/") or filename_lower.endswith(('.txt', '.csv', '.md', '.json'))

    try:
        if is_image:
            print("[LOG RENDER] Procesando archivo como imagen...")
            try:
                pil_image = Image.open(io.BytesIO(file_bytes))
                if pil_image.mode in ('RGBA', 'LA'):
                    pil_image = pil_image.convert('RGB')
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Error al decodificar la imagen: {str(e)}"
                )

            if GEMINI_ACTIVE:
                model = genai.GenerativeModel("gemini-2.5-flash")
                prompt = (
                    "Describe y resume de forma clara y concisa en español (en un párrafo corto de "
                    "máximo 3 o 4 oraciones) lo que ves en esta imagen. Esta descripción se leerá en "
                    "voz alta, por lo que debe ser fluida, natural y directa."
                )
                response = model.generate_content([prompt, pil_image])
                summary = response.text.strip()
                
                # Liberar memoria
                del pil_image
                gc.collect()
            else:
                summary = "Esta es una descripción simulada de la imagen. Gemini no está configurado."

        elif is_pdf:
            print("[LOG RENDER] Procesando archivo como PDF...")
            extracted_text = ""
            try:
                pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text += text + "\n"
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Error al leer el archivo PDF: {str(e)}"
                )

            if not extracted_text.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="El archivo PDF no contiene texto legible (podría estar escaneado sin OCR o vacío)."
                )

            print(f"[LOG RENDER] Texto extraído del PDF con éxito. Longitud: {len(extracted_text)} caracteres.")

            if GEMINI_ACTIVE:
                model = genai.GenerativeModel("gemini-2.5-flash")
                prompt = (
                    "Resume de forma clara, natural y concisa el contenido extraído del siguiente archivo PDF "
                    "en español, en un párrafo corto de máximo 4 oraciones. El resumen será leído en voz alta:\n\n"
                    f"{extracted_text}"
                )
                response = model.generate_content(prompt)
                summary = response.text.strip()
            else:
                summary = f"Este es un resumen simulado del PDF con texto extraído (longitud: {len(extracted_text)} caracteres). Gemini no está activo."

        elif is_txt:
            print("[LOG RENDER] Procesando archivo como texto plano...")
            try:
                file_text = file_bytes.decode("utf-8", errors="ignore")
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Error al decodificar el archivo de texto: {str(e)}"
                )

            if not file_text.strip():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="El archivo de texto está vacío."
                )

            if GEMINI_ACTIVE:
                model = genai.GenerativeModel("gemini-2.5-flash")
                prompt = (
                    "Resume de forma clara, natural y concisa el siguiente texto del archivo en español, "
                    "en un párrafo corto de máximo 4 oraciones. El resumen será leído en voz alta:\n\n"
                    f"{file_text}"
                )
                response = model.generate_content(prompt)
                summary = response.text.strip()
            else:
                summary = f"Este es un resumen simulado del archivo de texto (longitud: {len(file_text)} caracteres). Gemini no está activo."

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Formato de archivo no soportado. Tipo MIME: {content_type}. Solo se aceptan imágenes, PDFs y archivos de texto plano."
            )

        processing_time = round(time.time() - start_time, 3)
        print(f"[LOG RENDER] Resumen de archivo completado en {processing_time} segundos.")

        return {
            "status": "success",
            "resumen": summary,
            "filename": file.filename,
            "processing_time": processing_time
        }

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"[LOG RENDER] Error crítico en /resumir-archivo/: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno del servidor al procesar el archivo: {str(e)}"
        )
