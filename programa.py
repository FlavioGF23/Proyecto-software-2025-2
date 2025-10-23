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
USE_PUBLIC_SHARE = False  # cambiar a True solo si entiendes implicaciones de exponer tu app

# ---------------------------------------------------------------------------
# CARGA DEL MODELO (robusta)
# ---------------------------------------------------------------------------
print("Inicializando componentes... Por favor espera.")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}. Coloca el .h5 en el mismo folder o ajusta MODEL_PATH.")

try:
    # load_model(..., compile=False) evita problemas si faltan objetos de compilación
    classification_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"Modelo cargado desde {MODEL_PATH}")
    classification_model.summary()
except Exception as e:
    print("Error al cargar el modelo. Verifica que sea un modelo Keras/TensorFlow compatible.")
    raise

# ---------------------------------------------------------------------------
# HAAR CASCADE (detección de rostros)
# ---------------------------------------------------------------------------
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(haar_cascade_path)
if face_detector.empty():
    raise IOError("No se pudo cargar el Haar Cascade para detección de rostros.")

# ---------------------------------------------------------------------------
# UTIL: obtener nombre de la última capa conv si no conoces el nombre exacto
# ---------------------------------------------------------------------------
def find_last_conv_layer(model):
    # Busca la última capa con 4D output (batch, h, w, channels)
    for layer in reversed(model.layers):
        shape = layer.output_shape if hasattr(layer, "output_shape") else None
        if shape is None:
            continue
        # shape puede ser (None, H, W, C) o similar
        if len(shape) == 4:
            return layer.name
    return None
# ---------------------------------------------------------------------------
# LECTURA DE METADATOS
# ---------------------------------------------------------------------------
def extract_image_metadata(pil_img):
    meta_text = "Metadatos encontrados:\n"
    try:
        # 1. EXIF tradicional (cámaras)
        exif_data = pil_img._getexif() if hasattr(pil_img, "_getexif") else None
        if exif_data:
            for tag, value in exif_data.items():
                meta_text += f"{tag}: {value}\n"
        else:
            meta_text += "No hay metadatos EXIF.\n"
    except Exception as e:
        meta_text += f"Error leyendo EXIF: {e}\n"

    # 2. PIL info (XMP, PNGInfo, prompts IA, etc.)
    if hasattr(pil_img, "info") and pil_img.info:
        meta_text += "\n Datos adicionales (XMP / PNGInfo / prompts IA):\n"
        for key, value in pil_img.info.items():
            meta_text += f"{key}: {value}\n"
    else:
        meta_text += "\nNo hay información extra en PIL.info.\n"

    return meta_text.strip()
# ---------------------------------------------------------------------------
# PREPROCESS (extrae la cara principal y la redimensiona)
# ---------------------------------------------------------------------------
def preprocess_face(image_array_rgb, target_size=(128, 128)):
    try:
        # Asegurar tipo y rango
        image_np = np.asarray(image_array_rgb).astype(np.uint8)
        img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return None, image_np, None

        # elegir la cara más grande
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        face_rgb = image_np[y:y+h, x:x+w]

        # resize con PIL para mejor calidad
        face_pil = Image.fromarray(face_rgb).resize(target_size, Image.LANCZOS)
        face_arr = np.array(face_pil).astype(np.uint8)

        return face_arr, image_np, (x, y, w, h)
    except Exception as e:
        print(f"Error preprocess_face: {e}")
        return None, None, None

# ---------------------------------------------------------------------------
# CLASIFICACIÓN
# ---------------------------------------------------------------------------
def classify_face(face_image, model):
    if face_image is None:
        return None, None

    # Asegurar float32 antes del preprocess_input
    face = face_image.astype(np.float32)
    x = (face.copy() / 255.0)
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x, verbose=0)
    # Manejar formas (1,), (1,1) o (1,n)
    pred_val = float(np.squeeze(pred))

    # Interpretación: modelo con salida sigmoide -> prob de "fake"
    prob_fake = np.clip(pred_val, 0.0, 1.0)
    threshold = 0.5

    if prob_fake > threshold:
        label = "Potencialmente Falso (Fake)"
        confidence = prob_fake
    else:
        label = "Potencialmente Real"
        confidence = 1.0 - prob_fake

    return label, f"{confidence:.2%}"

# ---------------------------------------------------------------------------
# GRAD-CAM (corrección del cálculo del heatmap)
# ---------------------------------------------------------------------------
def generate_grad_cam(face_image, model, last_conv_layer_name=None):
    """
    face_image: RGB uint8 array (H,W,3)
    Retorna: RGB uint8 array con heatmap superpuesto
    """

    if face_image is None:
        return None

    # Determinar capa conv si no se pasó explicitamente
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
            # explicación de la neurona 0 (salida sigmoide)
            class_channel = predictions[:, 0]

        grads = tape.gradient(class_channel, conv_outputs)
        # pooled_grads: promedio en ejes de H y W
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]  # quitar dimensión batch -> (H, W, C)
        # multiplicación por canales y suma -> heatmap 2D
        heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
        heatmap = heatmap.numpy()

        # resize a tamaño de la cara original y crear color map
        heatmap = cv2.resize(heatmap, (face_image.shape[1], face_image.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR

        # convertir face a BGR para mezclar, luego volver a RGB
        face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        superimposed_bgr = cv2.addWeighted(face_bgr, 0.6, heatmap_color, 0.4, 0)
        superimposed_rgb = cv2.cvtColor(superimposed_bgr, cv2.COLOR_BGR2RGB)

        return superimposed_rgb
    except Exception as e:
        print(f"Error en generate_grad_cam: {e}")
        return None

# ---------------------------------------------------------------------------
# FUNCION PRINCIPAL PARA GRADIO
# ---------------------------------------------------------------------------
def run_detection_gradio(pil_img):
    # Convertir PIL a numpy RGB
    image_np = np.array(pil_img.convert("RGB"))
    face_arr, original_img, coords = preprocess_face(image_np)

    # Leer metadatos
    metadata_summary = extract_image_metadata(pil_img)

    if face_arr is None:
        return pil_img, "No se detectó ningún rostro en la imagen.", None, metadata_summary

    label, confidence = classify_face(face_arr, classification_model)
    heatmap_img = generate_grad_cam(face_arr, classification_model)

    # Dibujar bounding box sobre la imagen original (RGB)
    img_box = original_img.copy()
    if coords is not None:
        x, y, w, h = coords
        img_box_bgr = cv2.cvtColor(img_box, cv2.COLOR_RGB2BGR)
        cv2.rectangle(img_box_bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)
        img_box = cv2.cvtColor(img_box_bgr, cv2.COLOR_BGR2RGB)

    return (
        Image.fromarray(img_box),
        f"{label} (Confianza: {confidence})",
        Image.fromarray(heatmap_img) if heatmap_img is not None else None,
        metadata_summary
    )
# ---------------------------------------------------------------------------
# INTERFAZ GRADIO
# ---------------------------------------------------------------------------
demo = gr.Interface(
    fn=run_detection_gradio,
    inputs=gr.Image(type="pil", label="Sube una imagen"),
    outputs=[
        gr.Image(type="pil", label="Imagen con rostro"),
        gr.Textbox(label="Resultado del modelo"),
        gr.Image(type="pil", label="Mapa de Calor (Grad-CAM)"),
        gr.Textbox(label="Metadatos EXIF")
    ],
    title="Detector de Deepfakes con Explicabilidad (XAI)",
    description="Sube una imagen para analizarla con el modelo y visualizar sus metadatos EXIF."
)

# ---------------------------------------------------------------------------
# EJECUCIÓN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Iniciando interfaz de Gradio...")
    demo.launch(server_port=SERVER_PORT, share=USE_PUBLIC_SHARE)
