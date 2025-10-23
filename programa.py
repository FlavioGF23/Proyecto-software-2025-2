# ---------------------------------------------------------------------------
# PASO 1: IMPORTACIÓN de BIBLIOTECAS
# ---------------------------------------------------------------------------
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import warnings
import gradio as gr

# Configuración de logs
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

    # --- CAMBIO CRÍTICAL 1: CARGAR TU MODELO ENTRENADO ---
    MODEL_PATH = 'deepfake-detection-models-other-default-v12' 
    
    print(f"Cargando modelo desde: {MODEL_PATH}...")
    classification_model = tf.keras.models.load_model(MODEL_PATH)
    print("... Componentes listos.")

except Exception as e:
    print(f"Error fatal al cargar el modelo desde '{MODEL_PATH}'.")
    print(f"Asegúrate que la ruta sea correcta y el modelo sea compatible con Keras.")
    print(f"Error original: {e}")
    exit()

# ---------------------------------------------------------------------------
# COMPONENTE 1: PROCESAMIENTO DE DATOS (Modificado para eficiencia)
# ---------------------------------------------------------------------------
def preprocess_face(image_array_rgb, target_size=(299, 299)):
    """
    Procesa un array de NumPy (RGB) directamente desde PIL/Gradio.
    Ya no necesita leer desde un archivo.
    """
    try:
        # Convertir a BGR para OpenCV (detección de rostros)
        img_bgr = cv2.cvtColor(image_array_rgb, cv2.COLOR_RGB2BGR)
        gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            return None, None, None # No se encontró rostro

        # Extraer el rostro más grande
        main_face_coords = max(faces, key=lambda box: box[2] * box[3])
        x, y, w, h = main_face_coords
        
        # Recortar de la imagen RGB original
        face_cropped_rgb = image_array_rgb[y:y+h, x:x+w]
        
        # Redimensionar usando PIL (mejor calidad con antialias)
        face_image_pil = Image.fromarray(face_cropped_rgb)
        face_image_pil = face_image_pil.resize(target_size, Image.LANCZOS) # Usar LANCZOS es mejor
        
        return np.array(face_image_pil), image_array_rgb, main_face_coords
    except Exception as e:
        print(f"Error en preprocess_face: {e}")
        return None, None, None

# ---------------------------------------------------------------------------
# COMPONENTE 2: DETECCIÓN DE DEEPFAKES (Lógica corregida)
# ---------------------------------------------------------------------------
def classify_face(face_image, model):
    """
    Clasifica el rostro usando un modelo binario (0=Real, 1=Fake).
    """
    if face_image is None:
        return None, None

    # Preprocesar para Xception
    img_array = tf.keras.applications.xception.preprocess_input(face_image.copy())
    img_array = np.expand_dims(img_array, axis=0) # Crear batch de 1
    
    # --- CAMBIO CRÍTICAL 2: LÓGICA DE CLASIFICACIÓN BINARIA ---
    # Asume que el modelo tiene 1 neurona de salida (con sigmoide)
    prediction = model.predict(img_array, verbose=0)[0][0]
    confidence_score = float(prediction)
    
    # Usar 0.5 como umbral estándar
    if confidence_score > 0.5:
        label = "Potencialmente Falso (Fake)"
        confidence_percent = confidence_score
    else:
        label = "Potencialmente Real"
        confidence_percent = 1 - confidence_score
        
    return label, f"{confidence_percent:.2%}"

# ---------------------------------------------------------------------------
# COMPONENTE 3: EXPLICABILIDAD CON GRAD-CAM (Lógica corregida)
# ---------------------------------------------------------------------------
def generate_grad_cam(face_image, model, last_conv_layer_name="block14_sepconv2_act"):
    """
    Genera Grad-CAM para la única salida binaria del modelo.
    """
    if face_image is None:
        return None
        
    img_array = tf.keras.applications.xception.preprocess_input(face_image.copy())
    img_array_expanded = np.expand_dims(img_array, axis=0)

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array_expanded)
        
        # --- CAMBIO CRÍTICAL 3: EXPLICAR LA SALIDA BINARIA ---
        # No usamos argmax (eso es para 1000 clases)
        # Explicamos la única neurona de salida (índice 0)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
    heatmap = heatmap.numpy()

    # Redimensionar y superponer
    heatmap = cv2.resize(heatmap, (face_image.shape[1], face_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) # Mapa de calor (BGR)
    
    # --- Corrección de color: CV2 necesita BGR, PIL/Gradio necesita RGB ---
    face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR) # Convertir cara a BGR
    superimposed_img_bgr = cv2.addWeighted(face_image_bgr, 0.6, heatmap_colored, 0.4, 0)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img_bgr, cv2.COLOR_BGR2RGB) # Convertir de vuelta a RGB
    
    return superimposed_img_rgb

# ---------------------------------------------------------------------------
# FUNCIÓN INTEGRADORA PARA GRADIO (Modificada para eficiencia)
# ---------------------------------------------------------------------------
def run_detection_gradio(pil_image):
    """
    Función principal que usa Gradio. No guarda archivos temporales.
    """
    # --- CAMBIO CRÍTICAL 4: PROCESAR EN MEMORIA ---
    # Convertir de PIL (de Gradio) a NumPy array (RGB)
    image_np_rgb = np.array(pil_image)

    # 1. Preprocesar la imagen en memoria
    processed_face, original_img_rgb, face_coords = preprocess_face(image_np_rgb)
    
    if processed_face is None:
        # Si no hay cara, solo muestra la imagen original
        return pil_image, "No se detectó ningún rostro en la imagen.", None

    # 2. Clasificar
    label, confidence = classify_face(processed_face, classification_model)
    
    # 3. Generar XAI
    heatmap_img = generate_grad_cam(processed_face, classification_model)

    # 4. Dibujar recuadro en la imagen original
    img_with_box = original_img_rgb.copy()
    if face_coords is not None:
        x, y, w, h = face_coords
        # Dibujar en la imagen RGB (por eso el color es (0, 255, 0))
        cv2.rectangle(img_with_box, (x, y), (x+w, y+h), (0, 255, 0), 3)

    return (
        Image.fromarray(img_with_box),
        f"{label} (Confianza: {confidence})",
        Image.fromarray(heatmap_img) if heatmap_img is not None else None
    )

# ---------------------------------------------------------------------------
# INTERFAZ GRADIO
# ---------------------------------------------------------------------------
demo = gr.Interface(
    fn=run_detection_gradio,
    inputs=gr.Image(type="pil", label="Sube una imagen"),
    outputs=[
        gr.Image(type="pil", label="Rostro Detectado"),
        gr.Textbox(label="Resultado del modelo"),
        gr.Image(type="pil", label="Mapa de Calor (Grad-CAM)")
    ],
    title="Detector de Deepfakes con Explicabilidad (XAI)",
    description="Sube una imagen para analizarla con el modelo Xception. El modelo clasificará el rostro y Grad-CAM mostrará *por qué*."
)

# ---------------------------------------------------------------------------
# EJECUCIÓN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Iniciando interfaz de Gradio...")
    print("Abre la siguiente URL en tu navegador:")
    # 'share=False' es más seguro y te dará el enlace local http://127.0.0.1:7860
    demo.launch(server_port=7860, share=False)
