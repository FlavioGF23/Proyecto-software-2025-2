import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import warnings
import gradio as gr
from PIL.ExifTags import TAGS, GPSTAGS
import librosa 
import matplotlib.pyplot as plt
from io import BytesIO 

warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')

# ---------------------------------------------------------------------------
# CONFIGURACIÓN 
# ---------------------------------------------------------------------------

#Pesos del ensamble 
MODEL_WEIGHTS = {
    "XCEPTION": 0,   
    "CNN_DEEPFAKE": 1  
}

ENSEMBLE_MODEL_PATHS = {
   
    "XCEPTION": "xception_deepfake_video.h5",
    "CNN_DEEPFAKE": "cnn_model.h5"
    
}
MODEL_INPUT_SIZES = {
    "XCEPTION": (299, 299),   # Típico de Xception. 
    "CNN_DEEPFAKE": (128, 128) # Asumimos 128x128 
}
AUDIO_MODEL_PATH = "deepfake_audio_model.h5"
SERVER_PORT = 7860
AUDIO_HEIGHT = 128 
AUDIO_WIDTH = 256
USE_PUBLIC_SHARE = True

# ---------------------------------------------------------------------------
# CARGA DE MODELOS 
# ---------------------------------------------------------------------------
ensemble_models = {}
print("Cargando Modelos para Ensamble (IMAGEN/VIDEO)...")

for name, path in ENSEMBLE_MODEL_PATHS.items():
    if not os.path.exists(path):
        print(f" Modelo {name} no encontrado en: {path}. Se omitirá.")
        continue

    try:
        model = tf.keras.models.load_model(path, compile=False)
        ensemble_models[name] = model
        print(f" Modelo {name} cargado correctamente.")
    except Exception as e:
        print(f" Error al cargar {name}: {e}")
        print(f" Intentando modo de compatibilidad para {name}...")
        try:
            model = tf.keras.models.load_model(path, compile=False, safe_mode=True)
            ensemble_models[name] = model
            print(f"Modelo {name} cargado en modo compatibilidad.")
        except Exception as e2:
            print(f"No se pudo cargar {name} ni en modo compatibilidad: {e2}")

# Verificación de éxito
if not ensemble_models:
    print("Ningún modelo del ensamble se cargó correctamente.")
    image_model_flag = False
else:
    image_model_flag = True

# --- Modelo de Audio  ---
print("Cargando modelo de AUDIO...")
if not os.path.exists(AUDIO_MODEL_PATH):
    print(f"INFO: No se encontró el modelo de AUDIO en: {AUDIO_MODEL_PATH}.")
    print("La pestaña de AUDIO usará marcadores de posición.")
    audio_model = None
# ... (código para cargar audio_model)
else:
    try:
        audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH, compile=False)
        print(f"Modelo de AUDIO cargado desde {AUDIO_MODEL_PATH}")
    except Exception as e:
        print(f"Error al cargar el modelo de AUDIO: {e}")
        audio_model = None


# ---------------------------------------------------------------------------
# HAAR CASCADES 
# ---------------------------------------------------------------------------
print("Cargando detectores de rasgos (Haar Cascades)...")
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(haar_cascade_path)
eye_cascade_path = "haarcascade_eye.xml"
smile_cascade_path = "haarcascade_smile.xml"
profile_cascade_path = "haarcascade_profileface.xml"


eye_detector = cv2.CascadeClassifier(eye_cascade_path)
smile_detector = cv2.CascadeClassifier(smile_cascade_path)
profile_detector = cv2.CascadeClassifier(profile_cascade_path)

# ===========================================================================
# SECCIÓN 1: LÓGICA DE ANÁLISIS DE IMAGEN 
# ===========================================================================

def resize_face_for_model(face_arr, target_size):
    """Redimensiona el array de rostro a la dimensión específica requerida por un modelo."""
    
    target_wh = (target_size[1], target_size[0]) # Asegurar formato (H, W) para comparación con shape
    
    # Compara (alto, ancho) del array con el tamaño objetivo (alto, ancho)
    if face_arr.shape[:2] != target_size:
        # Redimensiona la imagen usando PIL
        face_pil = Image.fromarray(face_arr).resize(target_size, Image.LANCZOS)
        return np.array(face_pil).astype(np.uint8)
    return face_arr

def classify_ensemble(face_image, ensemble_models):
    if not ensemble_models: return None, None
    if face_image is None: return None, None
    
    # Usamos la configuración de pesos y tamaños
    MODEL_INPUT_SIZES = {
        "XCEPTION": (299, 299),
        "CNN_DEEPFAKE": (128, 128)
    }
    MODEL_WEIGHTS = {
        "XCEPTION": 0.2,   
        "CNN_DEEPFAKE": 0.8  
    }
    
    weighted_sum_pred = 0  # Suma ponderada 
    total_weight = 0       # Suma de los pesos de los modelos cargados
    
    # 1. Acumular Predicciones Ponderadas
    for name, model in ensemble_models.items():
        try:
            target_size = MODEL_INPUT_SIZES.get(name)
            weight = MODEL_WEIGHTS.get(name, 0.0) # Obtener el peso

            if target_size is None or weight == 0:
                print(f"ADVERTENCIA: Saltando {name}. Peso o tamaño de entrada no definido.")
                continue

            face_input = resize_face_for_model(face_image, target_size)
            
            x = (face_input.astype(np.float32).copy() / 255.0)
            x = np.expand_dims(x, axis=0)

            # Predecir
            pred = model.predict(x, verbose=0)
            prob_fake = float(np.squeeze(pred))
            
            # CAMBIO CLAVE: Sumar la predicción multiplicada por su peso
            weighted_sum_pred += prob_fake * weight
            total_weight += weight
            
            print(f"DEBUG: {name} (Peso: {weight}) predice: {prob_fake:.2%}")
            
        except Exception as e:
            print(f"Error en predicción del modelo {name}: {e}")
            
    if total_weight == 0:
        return "Error de Predicción", "0.00%"

    # 2. Promedio y Clasificación por Votación Ponderada
    # Dividimos la suma ponderada entre la suma total de los pesos cargados (debería ser 1.0)
    prob_fake = np.clip(weighted_sum_pred / total_weight, 0.0, 1.0)
    
    threshold = 0.5
    if prob_fake > threshold:
        label = "Potencialmente Falso (Fake) [Ponderado]"
        confidence = prob_fake
    else:
        label = "Potencialmente Real [Ponderado]"
        confidence = 1.0 - prob_fake
        
    return label, f"{confidence:.2%}"
def find_last_conv_layer(model):
    # Función original para el Grad-CAM
    for layer in reversed(model.layers):
        shape = layer.output_shape if hasattr(layer, "output_shape") else None
        if shape is None: continue
        if len(shape) == 4: return layer.name
    return None

def extract_image_metadata(pil_img):
    
    meta_text = "Metadatos encontrados:\n"
    try:
        exif_data = pil_img._getexif() if hasattr(pil_img, "_getexif") else None
        if exif_data:
            for tag, value in exif_data.items():
                meta_text += f"{tag}: {value}\n"
        else:
            meta_text += "No hay metadatos EXIF.\n"
    except Exception as e:
        meta_text += f"Error leyendo EXIF: {e}\n"

    if hasattr(pil_img, "info") and pil_img.info:
        meta_text += "\nDatos adicionales (XMP / PNGInfo / prompts IA):\n"
        for key, value in pil_img.info.items():
            meta_text += f"{key}: {value}\n"
    else:
        meta_text += "\nNo hay información extra en PIL.info.\n"
    return meta_text.strip()

def analyze_metadata_for_ai(metadata_text):
    # (Función original, no modificada)
    clues = []
    metadata_lower = metadata_text.lower()
    ai_keywords = [
        "stable diffusion", "sd-v1", "sd-v2", "sdxl",
        "midjourney", "parameters:", "prompt:", "negative prompt:",
        "model hash:", "comfyui", "a1111", "dall-e", "civitai",
        "generat(ed|ive) ai", "photoshop (ai|generative)","photoshop","x0cai","ai"
    ]
    for key in ai_keywords:
        if key in metadata_lower:
            if key not in clues:
                clues.append(key)
    if not clues:
        return "Metadatos Limpios: No se encontraron pistas obvias de IA."
    else:
        return f"⚠️ Pistas de IA Encontradas: {', '.join(clues)}"

def preprocess_face(image_array_rgb, target_size=(244,244)):
    # (Función original, no modificada)
    try:
        image_np = np.asarray(image_array_rgb).astype(np.uint8)
        img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return None, image_np, None, gray
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        face_rgb = image_np[y:y+h, x:x+w]
        face_pil = Image.fromarray(face_rgb).resize(target_size, Image.LANCZOS)
        face_arr = np.array(face_pil).astype(np.uint8)
        return face_arr, image_np, (x, y, w, h), gray
    except Exception as e:
        print(f"Error preprocess_face: {e}")
        return None, None, None, None

def classify_ensemble(face_image, ensemble_models):
    
    if not ensemble_models: return None, None
    if face_image is None: return None, None
        
    total_pred = 0
    model_count = 0
    
    # Definir input sizes DENTRO de la función para acceder a la constante
    MODEL_INPUT_SIZES = {
        "XCEPTION": (299, 299),
        "CNN_DEEPFAKE": (128, 128)
    }

    # 1. Acumular Predicciones de Probabilidad (0=Real, 1=Fake)
    for name, model in ensemble_models.items():
        try:
            
            target_size = MODEL_INPUT_SIZES.get(name)
            
            if target_size is None:
                # Si no está definido el tamaño, intentamos usar el original y registramos
                print(f"ADVERTENCIA: Tamaño de entrada no definido para {name}. Usando tamaño original.")
                face_input = face_image
            else:
                face_input = resize_face_for_model(face_image, target_size)
            
            # Preparación para la predicción (normalización y batch dimension)
            x = (face_input.astype(np.float32).copy() / 255.0)
            x = np.expand_dims(x, axis=0)

            # 2. Predecir
            pred = model.predict(x, verbose=0)
            total_pred += float(np.squeeze(pred))
            model_count += 1
            print(f"DEBUG: {name} procesado con {target_size}") # DEBUG para verificar el tamaño
            
        except Exception as e:
            print(f"Error en predicción del modelo {name}: {e}")

    # Si ningún modelo predijo, devolvemos error
    if model_count == 0:
        return "Error de Predicción", "0.00%"

    # Promedio de las predicciones
    prob_fake = np.clip(total_pred / model_count, 0.0, 1.0)

    threshold = 0.5
    if prob_fake > threshold:
        label = "Potencialmente Falso (Fake) [Ensamble]"
        confidence = prob_fake
    else:
        label = "Potencialmente Real [Ensamble]"
        confidence = 1.0 - prob_fake

    return label, f"{confidence:.2%}"


def generate_grad_cam(face_image, model):
    if face_image is None: return None
    
    SPECIFIC_CONV_LAYER_NAME = 'block14_sepconv2_act' 
    last_conv_layer_name = None
    
    # 1. Definir la capa convolucional
    try:
        model.get_layer(SPECIFIC_CONV_LAYER_NAME)
        last_conv_layer_name = SPECIFIC_CONV_LAYER_NAME
    except ValueError:
        # Fallback si no es Xception o la capa fue renombrada
        last_conv_layer_name = find_last_conv_layer(model) 

    if last_conv_layer_name is None:
        print("ADVERTENCIA: No se encontró una capa convolucional adecuada para Grad-CAM.")
        return None
        
    # --- Lógica de Grad-CAM (TensorFlow) ---
    try:
        face = face_image.astype(np.float32)
        x = (face.copy() / 255.0)
        x_exp = np.expand_dims(x, axis=0)
        
        
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            # 3. Llamada al modelo, una tupla de 2 elementos.
            outputs = grad_model(x_exp)
            
            # FORZAR A TENSOR antes de indexar.
            conv_outputs = tf.convert_to_tensor(outputs[0]) # Activaciones CONV
            predictions_tensor = tf.convert_to_tensor(outputs[1]) # Predicción Final
            
            # 4. Indexación segura: Lote 0, Canal 0 (clase 'Fake')
            class_channel = predictions_tensor[0, 0] 
            
        # 5. Cálculo de gradientes y mapa de calor
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # 6. Generación del Heatmap
        heatmap = tf.reduce_sum(tf.multiply(conv_outputs[0], pooled_grads), axis=-1) 
        
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
        
        # --- Visualización (OpenCV) ---
        heatmap = heatmap.numpy()
        
        #  BLINDAJE CONTRA EL ERROR DE CV2.RESIZE 
        h_face, w_face = face_image.shape[0], face_image.shape[1]
        
        if h_face <= 0 or w_face <= 0:
            print(f"ADVERTENCIA: Dimensiones del rostro no válidas: {w_face}x{h_face}. Saltando Grad-CAM.")
            return None 

        heatmap = cv2.resize(heatmap, (w_face, h_face))
        
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        superimposed_bgr = cv2.addWeighted(face_bgr, 0.6, heatmap_color, 0.4, 0)
        superimposed_rgb = cv2.cvtColor(superimposed_bgr, cv2.COLOR_BGR2RGB)
        
        return superimposed_rgb
        
    except Exception as e:
        # Esto captura cualquier error interno en el flujo de Grad-CAM
        print(f"Error en el cálculo y visualización de Grad-CAM para XCEPTION: {e}")
        return None

# --- FUNCIÓN PRINCIPAL DE IMAGEN ---
image_model_flag = True
def analyze_image(pil_img):
    if not image_model_flag:
        raise gr.Error("ERROR: Ningún modelo de ensamble de imagen está cargado.")

    image_np = np.array(pil_img.convert("RGB"))
    face_arr, original_img, coords, original_gray = preprocess_face(image_np)

    metadata_summary = extract_image_metadata(pil_img)
    ai_metadata_verdict = analyze_metadata_for_ai(metadata_summary)

    if face_arr is None:
        label, confidence, heatmap_img = "No se detectó rostro", "N/A", None
    else:
        # 1. Clasificación por Ensamble 
        label, confidence = classify_ensemble(face_arr, ensemble_models)
        
        # 2. XAI: Solo se genera Grad-CAM para XCEPTION 
        xception_model = ensemble_models.get("XCEPTION")
        
        if xception_model:
            
            # Usamos el tamaño definido en la configuración para XCEPTION
            XCEPTION_SIZE = (299, 299)
            face_for_xception = resize_face_for_model(face_arr, XCEPTION_SIZE)
            
            # Ahora pasamos el rostro redimensionado, compatible con el modelo.
            heatmap_img = generate_grad_cam(face_for_xception, xception_model)
        else:
            heatmap_img = None

    # .Código de detección de rasgos con Haar Cascades
    img_box = original_img.copy()
    img_box_bgr = cv2.cvtColor(img_box, cv2.COLOR_RGB2BGR)

    if not profile_detector.empty():
        profiles = profile_detector.detectMultiScale(original_gray, 1.1, 4, minSize=(30, 30))
        for (px, py, pw, ph) in profiles:
            cv2.rectangle(img_box_bgr, (px, py), (px + pw, py + ph), (0, 255, 255), 2)

    if coords is not None:
        x, y, w, h = coords
        cv2.rectangle(img_box_bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)
        face_roi_gray = original_gray[y:y+h, x:x+w]

        if not eye_detector.empty():
            eyes = eye_detector.detectMultiScale(face_roi_gray, 1.1, 5)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img_box_bgr, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)

        if not smile_detector.empty():
            smiles = smile_detector.detectMultiScale(face_roi_gray, 1.7, 20, minSize=(25, 25))
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(img_box_bgr, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 0, 255), 2)

    img_box = cv2.cvtColor(img_box_bgr, cv2.COLOR_BGR2RGB)

    # --- Devolver 5 valores ---
    return (
        Image.fromarray(img_box),
        f"{label} (Confianza: {confidence})",
        Image.fromarray(heatmap_img) if heatmap_img is not None else None,
        metadata_summary,
        ai_metadata_verdict  
    )

# ===========================================================================
# SECCIÓN 1.5: LÓGICA DE ANÁLISIS DE VIDEO 
# ===========================================================================

def analyze_video(video_path):
    """
    Analiza un archivo de video cuadro por cuadro (1fps) 
    aplicando el Ensamble y la Votación.
    """
    if video_path is None:
        return "Por favor, sube un archivo de video."

    # Validar que los modelos de ensamble estén cargados
    if not ensemble_models:
         raise gr.Error("ERROR: No hay modelos de ensamble cargados para el análisis de video.")

    print(f"Iniciando análisis de video: {video_path}")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "Error: No se pudo abrir el archivo de video."

        fps = cap.get(cv2.CAP_PROP_FPS)
        # Estrategia de Mitigación: Procesar aproximadamente 1 frame por segundo
        frame_skip = int(fps) if fps > 0 else 1

        fake_scores = []
        real_scores = []
        frames_analyzed = 0
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                frames_analyzed += 1

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_arr, _, _, _ = preprocess_face(frame_rgb)

                if face_arr is not None:
                    # Usamos la función del ensamble para la clasificación por frame
                    label, confidence_str = classify_ensemble(face_arr, ensemble_models) 

                    # Convertir "95.00%" a un número 0.95
                    score = float(confidence_str.replace('%', '')) / 100.0

                    if "Falso" in label:
                        fake_scores.append(score)
                    else:
                        real_scores.append(score)

            frame_count += 1

        cap.release()

        # --- Generar el Reporte Final (Votación Agregada) ---
        total_seconds = frame_count / (fps if fps > 0 else 1)

        if not fake_scores and not real_scores:
            return (
                f"Análisis Completo (Duración: {total_seconds:.1f}s):\n"
                f"No se detectaron rostros en los {frames_analyzed} cuadros analizados."
            )

        avg_fake = np.mean(fake_scores) if fake_scores else 0
        max_fake = np.max(fake_scores) if fake_scores else 0
        
        # La decisión final se basa en la mayoría de cuadros clasificados como 'Fake'
        fake_ratio = len(fake_scores) / (len(fake_scores) + len(real_scores))

        report = (
            f"--- Reporte de Análisis de Video (Ensamble) ---\n"
            f"Duración Total: {total_seconds:.1f} segundos\n"
            f"Cuadros Analizados: {frames_analyzed}\n"
            f"Ratio de Cuadros Falsos: {fake_ratio:.2%}\n\n"
            f"VEREDICTO FINAL: {'FALSO' if fake_ratio > 0.5 else 'REAL'} \n"
            f"(Basado en la mayoría de cuadros del ensamble)\n\n"
            f"--- Estadísticas Frame-a-Frame ---\n"
            f"Puntuación Máxima de Falsedad (Frame): {max_fake:.2%}\n"
            f"Cuadros con Rostros 'Fake': {len(fake_scores)}\n"
            f"Cuadros con Rostros 'Real': {len(real_scores)}\n"
        )

        print("Análisis de video completado.")
        return report

    except Exception as e:
        print(f"Error en analyze_video: {e}")
        return f"Error durante el análisis: {e}"


# ===========================================================================
# SECCIÓN 2: LÓGICA DE ANÁLISIS DE AUDIO 
# ===========================================================================

def preprocess_audio(audio_path, target_height=AUDIO_HEIGHT, target_width=AUDIO_WIDTH):
    
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_height)
        log_mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
        target_size = (target_width, target_height)
        fig = plt.figure(figsize=(target_width/100, target_height/100), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        librosa.display.specshow(log_mel_spect, sr=sr, ax=ax)
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        spectrogram_pil_rgb = Image.open(buf).resize(target_size, Image.LANCZOS)
        spectrogram_pil_gray = spectrogram_pil_rgb.convert('L')
        spectrogram_arr_gray = np.array(spectrogram_pil_gray).astype(np.uint8)
        return spectrogram_arr_gray, spectrogram_pil_rgb
    except Exception as e:
        print(f"Error procesando audio: {e}")
        return None, None
    
def analyze_audio(audio_path):
    
    if audio_model is None:
        raise gr.Error(f"El modelo de audio '{AUDIO_MODEL_PATH}' no está cargado. Esta es una demostración.")
    spectrogram_arr_gray, spectrogram_pil = preprocess_audio(audio_path)
    if spectrogram_arr_gray is None:
        return "Error procesando el audio", None
    x = (spectrogram_arr_gray.astype(np.float32) / 255.0)
    x = np.expand_dims(x, axis=-1) 
    x = np.expand_dims(x, axis=0) 
    pred = audio_model.predict(x, verbose=0)
    pred_val = float(np.squeeze(pred))
    prob_fake = np.clip(pred_val, 0.0, 1.0)
    threshold = 0.5
    if prob_fake > threshold:
        label = f"Voz Potencialmente Falsa (Fake)\n(Confianza: {prob_fake:.2%})"
    else:
        label = f"Voz Potencialmente Real\n(Confianza: {1.0 - prob_fake:.2%})"
    return gr.Label(label, value=label), spectrogram_pil


# ===========================================================================
# SECCIÓN 3: INTERFAZ DE GRADIO 
# ===========================================================================

with gr.Blocks(title="Detector Multimodal de Deepfakes") as demo:
    gr.Markdown(
        """
        # Detector Multimodal de Deepfakes con XAI (Ensamble)
        Proyecto de Software (CIB02-N). Sistema de Ensamble y XAI para un análisis forense más robusto.
        """
    )
    # ... (El resto de la interfaz Gradio, tabs y clicks, permanece igual) ...
    with gr.Tabs():
        with gr.TabItem("Análisis de Imagen "):
            gr.Markdown("Sube una **imagen** para la detección de deepfake facial, XAI y análisis de rasgos (Clasificación por Ensamble).")
            with gr.Row(variant="panel"):
                with gr.Column(scale=1):
                    image_in = gr.Image(type="pil", label="Subir Imagen")
                    image_button = gr.Button("Analizar Imagen", variant="primary")
                with gr.Column(scale=2):
                    image_out_label = gr.Textbox(label="Resultado del Ensamble")
                    image_out_ai_meta = gr.Textbox(label="Análisis Metadatos IA")
                    image_out_features = gr.Image(type="pil", label="Imagen con Rasgos Detectados")
                    with gr.Accordion("Ver Análisis Forense Detallado", open=False):
                        with gr.Row():
                            image_out_heatmap = gr.Image(type="pil", label="Mapa de Calor (Xception, XAI)")
                            image_out_meta = gr.Textbox(label="Metadatos EXIF / IA (Raw)")

        with gr.TabItem("Análisis de Video"):
            gr.Markdown("Sube un archivo de **video** para la detección de deepfake por Votación de Ensamble (a ~1 fps).")
            with gr.Row(variant="panel"):
                with gr.Column(scale=1):
                    video_in = gr.Video(label="Subir Video")
                    video_button = gr.Button("Analizar Video", variant="primary")
                with gr.Column(scale=1):
                    video_out_report = gr.Textbox(label="Reporte de Votación del Ensamble", lines=15)

        with gr.TabItem("Análisis de Audio"):
            gr.Markdown("Sube un archivo de **audio** para la detección de clonación de voz.")
            with gr.Row(variant="panel"):
                with gr.Column(scale=1):
                    audio_in = gr.Audio(type="filepath", label="Subir Audio")
                    audio_button = gr.Button("Analizar Audio", variant="primary")
                with gr.Column(scale=1):
                    audio_out_label = gr.Label(label="Resultado del Modelo")
                    audio_out_spec = gr.Image(type="pil", label="Espectrograma Mel")

    image_button.click(
        fn=analyze_image,
        inputs=[image_in],
        outputs=[
            image_out_features,
            image_out_label,
            image_out_heatmap,
            image_out_meta,
            image_out_ai_meta
        ]
    )

    video_button.click(
        fn=analyze_video,
        inputs=[video_in],
        outputs=[video_out_report]
    )

    audio_button.click(
        fn=analyze_audio,
        inputs=[audio_in],
        outputs=[audio_out_label, audio_out_spec]
    )

if __name__ == "__main__":
    print("Iniciando interfaz de Gradio en pestañas...")
    demo.launch(server_port=SERVER_PORT, share=USE_PUBLIC_SHARE)