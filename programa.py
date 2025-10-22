# ---------------------------------------------------------------------------
# PASO 1: IMPORTACIÓN DE BIBLIOTECAS
# ---------------------------------------------------------------------------
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import gradio as gr

warnings.filterwarnings('ignore', category=FutureWarning)
tf.get_logger().setLevel('ERROR')

# ---------------------------------------------------------------------------
# PASO 2: INICIALIZACIÓN DE MODELOS
# ---------------------------------------------------------------------------
print("Inicializando componentes... Por favor espera.")
try:
    # Clasificador para rostro principal
    haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(haar_cascade_path)
    if face_detector.empty():
        raise IOError("No se pudo cargar el clasificador Haar Cascade para rostros.")
except Exception as e:
    print(e)
    exit()

# Modelo de clasificación Xception
classification_model = tf.keras.applications.Xception(weights='imagenet')
print("... Componentes listos.")

# ---------------------------------------------------------------------------
# COMPONENTE 1: PROCESAMIENTO DE DATOS
# ---------------------------------------------------------------------------
def preprocess_face(image_path, target_size=(299, 299)):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, None, None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return None, None, None

        main_face_coords = max(faces, key=lambda box: box[2] * box[3])
        x, y, w, h = main_face_coords
        face_cropped_rgb = img_rgb[y:y+h, x:x+w]
        
        face_image_resized = Image.fromarray(face_cropped_rgb)
        face_image_resized = face_image_resized.resize(target_size)
        
        return np.array(face_image_resized), img_rgb, main_face_coords
    except Exception:
        return None, None, None

# ---------------------------------------------------------------------------
# COMPONENTE 2: DETECCIÓN DE DEEPFAKES
# ---------------------------------------------------------------------------
def classify_face(face_image, model):
    if face_image is None:
        return None, None

    img_array = tf.keras.applications.xception.preprocess_input(face_image)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    confidence_score = float(np.max(predictions[0]))
    
    if confidence_score > 0.4:
        label = "Potencialmente Falso (Fake)"
        confidence_percent = confidence_score
    else:
        label = "Potencialmente Real"
        confidence_percent = 1 - confidence_score
        
    return label, f"{confidence_percent:.2%}"

# ---------------------------------------------------------------------------
# COMPONENTE 3: EXPLICABILIDAD CON GRAD-CAM
# ---------------------------------------------------------------------------
def generate_grad_cam(face_image, model, last_conv_layer_name="block14_sepconv2_act"):
    if face_image is None:
        return None
        
    img_array = tf.keras.applications.xception.preprocess_input(face_image.copy())
    img_array = np.expand_dims(img_array, axis=0)

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    heatmap = heatmap.numpy()

    heatmap = cv2.resize(heatmap, (face_image.shape[1], face_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(face_image, 0.6, heatmap_colored, 0.4, 0)
    
    return superimposed_img

# ---------------------------------------------------------------------------
# NUEVO: FUNCIÓN PARA GRADIO (usa tu pipeline original)
# ---------------------------------------------------------------------------
def run_detection_gradio(image):
    temp_path = "temp_input.jpg"
    image.save(temp_path)

    processed_face, original_img_rgb, face_coords = preprocess_face(temp_path)
    if processed_face is None:
        return None, "No se detectó ningún rostro en la imagen.", None

    label, confidence = classify_face(processed_face, classification_model)
    heatmap_img = generate_grad_cam(processed_face, classification_model)

    # Dibujar recuadro en la imagen original
    img_with_box = original_img_rgb.copy()
    if face_coords is not None:
        x, y, w, h = face_coords
        cv2.rectangle(img_with_box, (x, y), (x+w, y+h), (0, 255, 0), 3)

    return (
        Image.fromarray(img_with_box),
        f"{label} (Confianza: {confidence})",
        Image.fromarray(heatmap_img)
    )

# ---------------------------------------------------------------------------
# INTERFAZ GRADIO
# ---------------------------------------------------------------------------
demo = gr.Interface(
    fn=run_detection_gradio,
    inputs=gr.Image(type="pil", label="Sube una imagen"),
    outputs=[
        gr.Image(label="Rostro Detectado"),
        gr.Textbox(label="Resultado del modelo"),
        gr.Image(label="Mapa de Calor (Grad-CAM)")
    ],
    title="Detector de Deepfakes con Explicabilidad (Grad-CAM)",
    description="Sube una imagen para analizarla con el modelo Xception y visualizar la zona de atención del modelo."
)

# ---------------------------------------------------------------------------
# EJECUCIÓN LOCAL O EN HUGGING FACE
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
