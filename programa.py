import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import warnings
import gradio as gr
from PIL.ExifTags import TAGS, GPSTAGS

warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODEL_PATH = "cnn_model.h5" 
SERVER_PORT = 7860
USE_PUBLIC_SHARE = True # Para que funcione el enlace público

# ---------------------------------------------------------------------------
# CARGA DEL MODELO (robusta)
# ---------------------------------------------------------------------------
print("Inicializando componentes... Por favor espera.")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}. Coloca el .h5 en el mismo folder o ajusta MODEL_PATH.")

try:
    classification_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"Modelo cargado desde {MODEL_PATH}")
    
except Exception as e:
    print("Error al cargar el modelo. Verifica que sea un modelo Keras/TensorFlow compatible.")
    raise

# ---------------------------------------------------------------------------
# HAAR CASCADE (detección de rostros y rasgos)
# ---------------------------------------------------------------------------
# Detector de cara frontal 
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(haar_cascade_path)
if face_detector.empty():
    raise IOError("No se pudo cargar el Haar Cascade para detección de rostros.")

# --- Cargar los otros detectores (asegúrate que los .xml estén en la carpeta) ---
print("Cargando detectores de rasgos (ojos, sonrisa, perfil)...")
eye_cascade_path = "haarcascade_eye.xml"
smile_cascade_path = "haarcascade_smile.xml"
profile_cascade_path = "haarcascade_profileface.xml"

eye_detector = cv2.CascadeClassifier(eye_cascade_path)
smile_detector = cv2.CascadeClassifier(smile_cascade_path)
profile_detector = cv2.CascadeClassifier(profile_cascade_path)

if eye_detector.empty():
    print("ADVERTENCIA: No se pudo cargar 'haarcascade_eye.xml'. Asegúrate que esté en la carpeta.")
if smile_detector.empty():
    print("ADVERTENCIA: No se pudo cargar 'haarcascade_smile.xml'. Asegúrate que esté en la carpeta.")
if profile_detector.empty():
    print("ADVERTENCIA: No se pudo cargar 'haarcascade_profileface.xml'. Asegúrate que esté en la carpeta.")
# ---------------------------------------------------------------------------


# ... (El resto de tus funciones find_last_conv_layer y extract_image_metadata no cambian) ...

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        shape = layer.output_shape if hasattr(layer, "output_shape") else None
        if shape is None:
            continue
        if len(shape) == 4:
            return layer.name
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
        meta_text += "\n Datos adicionales (XMP / PNGInfo / prompts IA):\n"
        for key, value in pil_img.info.items():
            meta_text += f"{key}: {value}\n"
    else:
        meta_text += "\nNo hay información extra en PIL.info.\n"
    return meta_text.strip()

# ---------------------------------------------------------------------------
# PREPROCESS (extrae la cara principal y la redimensiona)
# --- MODIFICADO: Ahora devuelve la imagen en gris ---
# ---------------------------------------------------------------------------
def preprocess_face(image_array_rgb, target_size=(128, 128)):
    try:
        image_np = np.asarray(image_array_rgb).astype(np.uint8)
        img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            # --- Devolvemos 'gray' incluso si no hay cara ---
            return None, image_np, None, gray

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        face_rgb = image_np[y:y+h, x:x+w]

        face_pil = Image.fromarray(face_rgb).resize(target_size, Image.LANCZOS)
        face_arr = np.array(face_pil).astype(np.uint8)

        # --- Devolvemos 'gray' ---
        return face_arr, image_np, (x, y, w, h), gray
    except Exception as e:
        print(f"Error preprocess_face: {e}")
        return None, None, None, None

# ... (Tus funciones classify_face y generate_grad_cam no cambian) ...

def classify_face(face_image, model):
    if face_image is None:
        return None, None
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

def generate_grad_cam(face_image, model, last_conv_layer_name=None):
    if face_image is None:
        return None
    if last_conv_layer_name is None:
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
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET) # BGR
        face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        superimposed_bgr = cv2.addWeighted(face_bgr, 0.6, heatmap_color, 0.4, 0)
        superimposed_rgb = cv2.cvtColor(superimposed_bgr, cv2.COLOR_BGR2RGB)
        return superimposed_rgb
    except Exception as e:
        print(f"Error en generate_grad_cam: {e}")
        return None

# ---------------------------------------------------------------------------
# FUNCION PRINCIPAL PARA GRADIO
# --- MODIFICADO: Para detectar y dibujar todos los rasgos ---
# ---------------------------------------------------------------------------
def run_detection_gradio(pil_img):
    image_np = np.array(pil_img.convert("RGB"))
    
    # --- MODIFICADO: Recibimos 'original_gray' ---
    face_arr, original_img, coords, original_gray = preprocess_face(image_np)

    metadata_summary = extract_image_metadata(pil_img)

    if face_arr is None:
        label, confidence, heatmap_img = "No se detectó rostro", "N/A", None
    else:
        label, confidence = classify_face(face_arr, classification_model)
        heatmap_img = generate_grad_cam(face_arr, classification_model)

    # --- NUEVO: Dibujar TODOS los rasgos detectados ---
    img_box = original_img.copy()
    img_box_bgr = cv2.cvtColor(img_box, cv2.COLOR_RGB2BGR) # Convertir a BGR para dibujar

    # 1. Dibujar cara de perfil (en toda la imagen)
    if not profile_detector.empty():
        profiles = profile_detector.detectMultiScale(original_gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        for (px, py, pw, ph) in profiles:
            cv2.rectangle(img_box_bgr, (px, py), (px + pw, py + ph), (0, 255, 255), 2) # Amarillo

    # 2. Dibujar cara frontal, ojos y sonrisa (si se encontró una cara frontal)
    if coords is not None:
        x, y, w, h = coords
        # Dibujar cara frontal (verde)
        cv2.rectangle(img_box_bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Crear Región de Interés (ROI) para buscar rasgos DENTRO de la cara
        face_roi_gray = original_gray[y:y+h, x:x+w]

        # Dibujar ojos (azul)
        if not eye_detector.empty():
            eyes = eye_detector.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=5)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img_box_bgr, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)

        # Dibujar sonrisa (rojo)
        if not smile_detector.empty():
            smiles = smile_detector.detectMultiScale(face_roi_gray, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(img_box_bgr, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 0, 255), 2)

    # Convertir la imagen final de vuelta a RGB para Gradio
    img_box = cv2.cvtColor(img_box_bgr, cv2.COLOR_BGR2RGB)
    # ----------------------------------------------------

    return (
        Image.fromarray(img_box),
        f"{label} (Confianza: {confidence})",
        Image.fromarray(heatmap_img) if heatmap_img is not None else None,
        metadata_summary
    )
# ---------------------------------------------------------------------------
# INTERFAZ GRADIO (sin cambios)
# ---------------------------------------------------------------------------
demo = gr.Interface(
    fn=run_detection_gradio,
    inputs=gr.Image(type="pil", label="Sube una imagen"),
    outputs=[
        gr.Image(type="pil", label="Imagen con rasgos detectados"), # Texto actualizado
        gr.Textbox(label="Resultado del modelo"),
        gr.Image(type="pil", label="Mapa de Calor (Grad-CAM)"),
        gr.Textbox(label="Metadatos EXIF")
    ],
    title="Detector de Deepfakes con Explicabilidad (XAI) y Detección de Rasgos",
    description="Sube una imagen para analizarla con el modelo, detectar rasgos (cara, perfil, ojos, sonrisa) y visualizar metadatos."
)

# ---------------------------------------------------------------------------
# EJECUCIÓN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Iniciando interfaz de Gradio...")
    # Si quieres el enlace público, inicia sesión con 'gradio login' en tu terminal
    demo.launch(server_port=SERVER_PORT, share=USE_PUBLIC_SHARE)