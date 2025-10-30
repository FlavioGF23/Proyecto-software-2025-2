import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import warnings
import gradio as gr
from PIL.ExifTags import TAGS, GPSTAGS
import librosa               #  Para procesar audio
import matplotlib.pyplot as plt # Para crear espectrogramas
from io import BytesIO       # Para manejar imágenes en memoria

warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
IMAGE_MODEL_PATH = "cnn_model.h5"  # El que estás entrenando en Colab
AUDIO_MODEL_PATH = "audio_model.h5"  # El que entrenaremos después
SERVER_PORT = 7860
USE_PUBLIC_SHARE = True

# ---------------------------------------------------------------------------
# CARGA DE MODELOS (Imagen y Audio)
# ---------------------------------------------------------------------------

# --- Modelo de Imagen (tu código original) ---
print("Cargando modelo de IMAGEN...")
if not os.path.exists(IMAGE_MODEL_PATH):
    print(f"ADVERTENCIA: No se encontró el modelo en: {IMAGE_MODEL_PATH}.")
    print("La pestaña de IMAGEN no funcionará hasta que el modelo esté en la carpeta.")
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

def preprocess_face(image_array_rgb, target_size=(128, 128)):
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

# --- FUNCIÓN PRINCIPAL DE IMAGEN ---
def analyze_image(pil_img):
    if image_model is None:
        raise gr.Error(f"El modelo de imagen '{IMAGE_MODEL_PATH}' no está cargado.")
    
    image_np = np.array(pil_img.convert("RGB"))
    face_arr, original_img, coords, original_gray = preprocess_face(image_np)
    metadata_summary = extract_image_metadata(pil_img)

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

    return (
        Image.fromarray(img_box),
        f"{label} (Confianza: {confidence})",
        Image.fromarray(heatmap_img) if heatmap_img is not None else None,
        metadata_summary
    )

# ===========================================================================
# SECCIÓN 2: LÓGICA DE ANÁLISIS DE AUDIO 
# ===========================================================================

def preprocess_audio(audio_path, target_size=(128, 128)):
    """
    Carga un audio, crea un espectrograma Mel y lo convierte en una
    imagen de 3 canales (RGB) lista para la CNN.
    """
    try:
        # Cargar archivo de audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Crear espectrograma Mel
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
        
        # --- Convertir el espectrograma (array) en una imagen PIL ---
        # Usamos Matplotlib para "dibujar" el espectrograma
        fig = plt.figure(figsize=(target_size[0]/100, target_size[1]/100), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # Dibujar el espectrograma sin bordes, ejes, etc.
        librosa.display.specshow(log_mel_spect, sr=sr, ax=ax)
        
        # Guardar la imagen en un buffer de memoria
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        
        # Cargar la imagen del buffer usando PIL
        spectrogram_pil = Image.open(buf).convert('RGB').resize(target_size, Image.LANCZOS)
        
        # Convertir a array de numpy
        spectrogram_arr = np.array(spectrogram_pil).astype(np.uint8)
        
        return spectrogram_arr, spectrogram_pil

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
    spectrogram_arr, spectrogram_pil = preprocess_audio(audio_path)
    
    if spectrogram_arr is None:
        return "Error procesando el audio", None

    # 2. Preparar para el modelo (igual que 'classify_face')
    x = (spectrogram_arr.astype(np.float32) / 255.0)
    x = np.expand_dims(x, axis=0)
    
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
# SECCIÓN 3: INTERFAZ DE GRADIO 
# ===========================================================================

with gr.Blocks(title="Detector Multimodal de Deepfakes") as demo:
    gr.Markdown(
        """
        # Detector Multimodal de Deepfakes con XAI
        Proyecto de Software (CIB02-N). Sube una imagen o un audio para el análisis forense.
        """
    )
    
    with gr.Tabs():
        # --- PESTAÑA 1: ANÁLISIS DE IMAGEN ---
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
                    image_out_features = gr.Image(type="pil", label="Imagen con Rasgos Detectados")
                    
                    with gr.Row():
                        image_out_heatmap = gr.Image(type="pil", label="Mapa de Calor (Grad-CAM)")
                        image_out_meta = gr.Textbox(label="Metadatos EXIF / IA")

            # (Futuro: Añadir componente del video)
            

        # --- PESTAÑA 2: ANÁLISIS DE AUDIO ---
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
    
    # Clic de la pestaña de Imagen
    image_button.click(
        fn=analyze_image,
        inputs=[image_in],
        outputs=[image_out_features, image_out_label, image_out_heatmap, image_out_meta]
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