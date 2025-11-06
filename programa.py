import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import warnings
import gradio as gr
from PIL.ExifTags import TAGS, GPSTAGS
import librosa  # Para procesar audio
import matplotlib.pyplot as plt  # Para crear espectrogramas
from io import BytesIO  # Para manejar imágenes en memoria

warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
IMAGE_MODEL_PATH = "xception_deepfake_video.h5"
AUDIO_MODEL_PATH = "deepfake_audio_model.h5"
SERVER_PORT = 7860
AUDIO_HEIGHT = 128 
AUDIO_WIDTH = 256
USE_PUBLIC_SHARE = True

# ---------------------------------------------------------------------------
# CARGA DE MODELOS (Imagen y Audio)
# ---------------------------------------------------------------------------

# --- Modelo de Imagen (tu código original) ---
print("Cargando modelo de IMAGEN...")
if not os.path.exists(IMAGE_MODEL_PATH):
    print(f"ADVERTENCIA: No se encontró el modelo en: {IMAGE_MODEL_PATH}.")
    print("La pestaña de IMAGEN y VIDEO no funcionará hasta que el modelo esté en la carpeta.")
    image_model = None
else:
    try:
        image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH, compile=False)
        print(f"Modelo de IMAGEN cargado desde {IMAGE_MODEL_PATH}")
    except Exception as e:
        print(f"Error al cargar el modelo de IMAGEN: {e}")
        image_model = None

# --- Modelo de Audio (NUEVO, con marcador de posición) ---
print("Cargando modelo de AUDIO...")
if not os.path.exists(AUDIO_MODEL_PATH):
    print(f"INFO: No se encontró el modelo de AUDIO en: {AUDIO_MODEL_PATH}.")
    print("La pestaña de AUDIO usará marcadores de posición.")
    audio_model = None
else:
    try:
        audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH, compile=False)
        print(f"Modelo de AUDIO cargado desde {AUDIO_MODEL_PATH}")
    except Exception as e:
        print(f"Error al cargar el modelo de AUDIO: {e}")
        audio_model = None

# ---------------------------------------------------------------------------
# HAAR CASCADES (Detección de rasgos )
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

def find_last_conv_layer(model):
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

# --- NUEVA FUNCIÓN ---
def analyze_metadata_for_ai(metadata_text):
    """
    Analiza el texto de metadatos en busca de palabras clave
    comunes de generación de IA.
    """
    clues = []
    metadata_lower = metadata_text.lower()

    # Lista de palabras clave que buscamos
    ai_keywords = [
        "stable diffusion", "sd-v1", "sd-v2", "sdxl",
        "midjourney", "parameters:", "prompt:", "negative prompt:",
        "model hash:", "comfyui", "a1111", "dall-e", "civitai",
        "generat(ed|ive) ai", "photoshop (ai|generative)","photoshop","x0cAI"
    ]

    for key in ai_keywords:
        if key in metadata_lower:
            # Añade la palabra clave encontrada (sin duplicados)
            if key not in clues:
                clues.append(key)

    if not clues:
        return "Metadatos Limpios: No se encontraron pistas obvias de IA."
    else:
        # Unimos las pistas encontradas en un string
        return f"⚠️ Pistas de IA Encontradas: {', '.join(clues)}"

def preprocess_face(image_array_rgb, target_size=(244,244)): # Usa 224x224
    try:
        image_np = np.asarray(image_array_rgb).astype(np.uint8)
        img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return None, image_np, None, gray

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        face_rgb = image_np[y:y+h, x:x+w]
        
        # El modelo Xception espera 224x224
        face_pil = Image.fromarray(face_rgb).resize(target_size, Image.LANCZOS)
        face_arr = np.array(face_pil).astype(np.uint8)
        return face_arr, image_np, (x, y, w, h), gray
    except Exception as e:
        print(f"Error preprocess_face: {e}")
        return None, None, None, None

def classify_face(face_image, model):
    if face_image is None: return None, None
    face = face_image.astype(np.float32)
    x = (face.copy() / 255.0)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x, verbose=0)
    pred_val = float(np.squeeze(pred))
    prob_fake = np.clip(pred_val, 0.0, 1.0)
    threshold = 0.5
    if prob_fake > threshold:
        label = "Potencialmente Falso (Fake)"
        confidence = prob_fake
    else:
        label = "Potencialmente Real"
        confidence = 1.0 - prob_fake
    return label, f"{confidence:.2%}"

def generate_grad_cam(face_image, model):

    if face_image is None: return None
    last_conv_layer_name = find_last_conv_layer(model)
    if last_conv_layer_name is None:
        print("No se encontró una capa convolucional en el modelo para Grad-CAM.")
        return None
    try:
        face = face_image.astype(np.float32)
        x = (face.copy() / 255.0)
        x_exp = np.expand_dims(x, axis=0)
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x_exp)
            class_channel = predictions[:, 0]
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, (face_image.shape[1], face_image.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        superimposed_bgr = cv2.addWeighted(face_bgr, 0.6, heatmap_color, 0.4, 0)
        superimposed_rgb = cv2.cvtColor(superimposed_bgr, cv2.COLOR_BGR2RGB)
        return superimposed_rgb
    except Exception as e:
        print(f"Error en generate_grad_cam: {e}")
        return None

# --- FUNCIÓN PRINCIPAL DE IMAGEN---
def analyze_image(pil_img):
    if image_model is None:
        raise gr.Error(f"El modelo de imagen '{IMAGE_MODEL_PATH}' no está cargado.")

    image_np = np.array(pil_img.convert("RGB"))
    face_arr, original_img, coords, original_gray = preprocess_face(image_np)

    # --- Llamar a ambas funciones de metadatos ---
    metadata_summary = extract_image_metadata(pil_img)
    ai_metadata_verdict = analyze_metadata_for_ai(metadata_summary)

    if face_arr is None:
        label, confidence, heatmap_img = "No se detectó rostro", "N/A", None
    else:
        label, confidence = classify_face(face_arr, image_model)
        heatmap_img = generate_grad_cam(face_arr, image_model)

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
# SECCIÓN 1.5: LÓGICA DE ANÁLISIS DE VIDEO (NUEVA SECCIÓN)
# ===========================================================================

def analyze_video(video_path):
    """
    Analiza un archivo de video cuadro por cuadro (1fps)
    para detectar deepfakes.
    """
    if video_path is None:
        return "Por favor, sube un archivo de video."

    # Validar que el modelo de imagen esté cargado
    if image_model is None:
        raise gr.Error(f"El modelo de imagen '{IMAGE_MODEL_PATH}' no está cargado. El análisis de video no puede continuar.")

    print(f"Iniciando análisis de video: {video_path}")

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "Error: No se pudo abrir el archivo de video."

        fps = cap.get(cv2.CAP_PROP_FPS)
        # Asegurarnos que fps sea un número válido > 0
        frame_skip = int(fps) if fps > 0 else 1

        fake_scores = []
        real_scores = []
        frames_analyzed = 0
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Termina el bucle si no hay más cuadros

            # Estrategia de mitigación: Analizar ~1 cuadro por segundo
            if frame_count % frame_skip == 0:
                frames_analyzed += 1

                # Convertir el cuadro (BGR) a RGB para PIL/TF
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # REUTILIZAMOS nuestras funciones de imagen
                # Pasamos el detector de rostros (face_detector) que está cargado globalmente
                face_arr, _, _, _ = preprocess_face(frame_rgb)

                if face_arr is not None:
                    # Pasamos el modelo (image_model) que está cargado globalmente
                    label, confidence_str = classify_face(face_arr, image_model)

                    # Convertir "95.00%" a un número 0.95
                    score = float(confidence_str.replace('%', '')) / 100.0

                    if label == "Potencialmente Falso (Fake)":
                        fake_scores.append(score)
                    else:
                        real_scores.append(score)

            frame_count += 1

        cap.release()

        # --- Generar el Reporte Final ---
        total_seconds = frame_count / (fps if fps > 0 else 1)

        if not fake_scores and not real_scores:
            return (
                f"Análisis Completo (Duración: {total_seconds:.1f}s):\n"
                f"No se detectaron rostros en los {frames_analyzed} cuadros analizados."
            )

        avg_fake = np.mean(fake_scores) if fake_scores else 0
        max_fake = np.max(fake_scores) if fake_scores else 0
        avg_real = np.mean(real_scores) if real_scores else 0

        report = (
            f"--- Reporte de Análisis de Video ---\n"
            f"Duración Total: {total_seconds:.1f} segundos\n"
            f"Cuadros Analizados: {frames_analyzed} (a ~1 fps)\n"
            f"Cuadros con Rostros 'Fake': {len(fake_scores)}\n"
            f"Cuadros con Rostros 'Real': {len(real_scores)}\n\n"
            f"--- Estadísticas 'Fake' ---\n"
            f"Puntuación Máxima de Falsedad: {max_fake:.2%}\n"
            f"Puntuación Promedio de Falsedad: {avg_fake:.2%}\n\n"
            f"--- Estadísticas 'Real' ---\n"
            f"Puntuación Promedio de Realidad: {avg_real:.2%}\n"
        )

        print("Análisis de video completado.")
        return report

    except Exception as e:
        print(f"Error en analyze_video: {e}")
        return f"Error durante el análisis: {e}"


# ===========================================================================
# SECCIÓN 2: LÓGICA DE ANÁLISIS DE AUDIO (CORRECCIÓN FINAL)
# ===========================================================================

def preprocess_audio(audio_path, target_height=AUDIO_HEIGHT, target_width=AUDIO_WIDTH):
    """
    Carga un audio, crea un espectrograma Mel de 128x256,
    y lo retorna como un array de 1 canal (Grises) y la imagen PIL RGB.
    """
    try:
        # Cargar archivo de audio
        y, sr = librosa.load(audio_path, sr=None)

        # 1. Crear espectrograma Mel con los n_mels correctos (128)
        # NOTA: librosa.feature.melspectrogram ya maneja el padding/corte 
        # para una longitud fija si lo hubieras implementado en entrenamiento, 
        # pero aquí vamos a usar la imagen para redimensionar.
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_height)
        log_mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

        # --- 2. Generación de la imagen PIL desde Matplotlib ---
        target_size = (target_width, target_height) # Nota: PIL espera (W, H)
        fig = plt.figure(figsize=(target_width/100, target_height/100), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        librosa.display.specshow(log_mel_spect, sr=sr, ax=ax)

        # Guardar la imagen en un buffer de memoria
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        # Cargar la imagen del buffer usando PIL
        spectrogram_pil_rgb = Image.open(buf).resize(target_size, Image.LANCZOS)
        
        # 3. CONVERSIÓN A ESCALA DE GRISES ('L') para el modelo CNN
        spectrogram_pil_gray = spectrogram_pil_rgb.convert('L') # Forma (H, W)
        
        # 4. Convertir a array de numpy (Forma: 128, 256)
        spectrogram_arr_gray = np.array(spectrogram_pil_gray).astype(np.uint8)
        
        # Retornamos el array de 1 canal (sin la dimensión final) y la imagen RGB
        return spectrogram_arr_gray, spectrogram_pil_rgb

    except Exception as e:
        print(f"Error procesando audio: {e}")
        return None, None
    
# --- FUNCIÓN PRINCIPAL DE AUDIO ---
def analyze_audio(audio_path):
    """
    Función principal para la pestaña de Audio.
    Carga el modelo de audio y analiza el espectrograma.
    """
    
    if audio_model is None:
        # Si el modelo no está cargado, mostramos un marcador de posición
        raise gr.Error(f"El modelo de audio '{AUDIO_MODEL_PATH}' no está cargado. Esta es una demostración.")

    # 1. Pre-procesar el audio
    spectrogram_arr_gray, spectrogram_pil = preprocess_audio(audio_path)

    if spectrogram_arr_gray is None:
        return "Error procesando el audio", None

    # 2. Preparar para el modelo (igual que 'classify_face')
   # Normalización
    x = (spectrogram_arr_gray.astype(np.float32) / 255.0)
    
    # Agrega la dimensión del canal: (128, 128) -> (128, 128, 1)
    x = np.expand_dims(x, axis=-1) 
    
    # Agrega la dimensión del Batch: (128, 128, 1) -> (1, 128, 128, 1)
    x = np.expand_dims(x, axis=0) 
    # ¡LA FORMA ES COMPATIBLE AHORA!

    # 3. Predecir (Usando el modelo de audio)
    pred = audio_model.predict(x, verbose=0)
    pred_val = float(np.squeeze(pred))
    prob_fake = np.clip(pred_val, 0.0, 1.0)

    threshold = 0.5
    if prob_fake > threshold:
        label = f"Voz Potencialmente Falsa (Fake)\n(Confianza: {prob_fake:.2%})"
    else:
        label = f"Voz Potencialmente Real\n(Confianza: {1.0 - prob_fake:.2%})"

    # 4. Devolver la etiqueta y la imagen del espectrograma
    return gr.Label(label, value=label), spectrogram_pil


# ===========================================================================
# SECCIÓN 3: INTERFAZ DE GRADIO (MODIFICADA)
# ===========================================================================

with gr.Blocks(title="Detector Multimodal de Deepfakes") as demo:
    gr.Markdown(
        """
        # Detector Multimodal de Deepfakes con XAI
        Proyecto de Software (CIB02-N). Sube una imagen, video o audio para el análisis forense.
        """
    )

    with gr.Tabs():
        # --- PESTAÑA 1: ANÁLISIS DE IMAGEN (MODIFICADA) ---
        with gr.TabItem("Análisis de Imagen "):
            gr.Markdown("Sube una **imagen** para la detección de deepfake facial, XAI y análisis de rasgos.")
            with gr.Row(variant="panel"):
                # Columna de Entradas
                with gr.Column(scale=1):
                    image_in = gr.Image(type="pil", label="Subir Imagen")
                    image_button = gr.Button("Analizar Imagen", variant="primary")

                # Columna de Salidas
                with gr.Column(scale=2):
                    image_out_label = gr.Textbox(label="Resultado del Modelo")
                    # --- NUEVO COMPONENTE (Indentación corregida) ---
                    image_out_ai_meta = gr.Textbox(label="Análisis Metadatos IA")
                    image_out_features = gr.Image(type="pil", label="Imagen con Rasgos Detectados")

                    with gr.Accordion("Ver Análisis Forense Detallado", open=False):
                        with gr.Row():
                            image_out_heatmap = gr.Image(type="pil", label="Mapa de Calor (Grad-CAM)")
                            image_out_meta = gr.Textbox(label="Metadatos EXIF / IA (Raw)")

        # --- PESTAÑA 2: ANÁLISIS DE VIDEO (NUEVA PESTAÑA) ---
        with gr.TabItem("Análisis de Video"):
            gr.Markdown("Sube un archivo de **video** (.mp4, .mov, etc.) para la detección de deepfake cuadro por cuadro (a ~1 fps).")
            with gr.Row(variant="panel"):
                # Columna de Entradas
                with gr.Column(scale=1):
                    video_in = gr.Video(label="Subir Video")
                    video_button = gr.Button("Analizar Video", variant="primary")

                # Columna de Salidas
                with gr.Column(scale=1):
                    video_out_report = gr.Textbox(label="Reporte de Análisis de Video", lines=15)

        # --- PESTAÑA 3: ANÁLISIS DE AUDIO ---
        with gr.TabItem("Análisis de Audio"):
            gr.Markdown("Sube un archivo de **audio** (.wav, .mp3) para la detección de clonación de voz.")
            with gr.Row(variant="panel"):
                # Columna de Entradas
                with gr.Column(scale=1):
                    audio_in = gr.Audio(type="filepath", label="Subir Audio")
                    audio_button = gr.Button("Analizar Audio", variant="primary")

                # Columna de Salidas
                with gr.Column(scale=1):
                    audio_out_label = gr.Label(label="Resultado del Modelo")
                    audio_out_spec = gr.Image(type="pil", label="Espectrograma Mel")

    # Clics de los botones

    # Clic de la pestaña de Imagen (MODIFICADO)
    image_button.click(
        fn=analyze_image,
        inputs=[image_in],
        # Orden de salidas actualizado a 5 componentes
        outputs=[
            image_out_features,
            image_out_label,
            image_out_heatmap,
            image_out_meta,
            image_out_ai_meta
        ]
    )

    # Clic de la pestaña de Video (NUEVO)
    video_button.click(
        fn=analyze_video,
        inputs=[video_in],
        outputs=[video_out_report]
    )

    # Clic de la pestaña de Audio
    audio_button.click(
        fn=analyze_audio,
        inputs=[audio_in],
        outputs=[audio_out_label, audio_out_spec]
    )

# ---------------------------------------------------------------------------
# EJECUCIÓN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Iniciando interfaz de Gradio en pestañas...")
    demo.launch(server_port=SERVER_PORT, share=USE_PUBLIC_SHARE)