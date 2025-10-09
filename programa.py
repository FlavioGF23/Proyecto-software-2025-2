# ---------------------------------------------------------------------------
# PASO 1: IMPORTACIÓN DE BIBLIOTECAS
# ---------------------------------------------------------------------------
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import warnings

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
    """
    Detecta, recorta y alinea un rostro en una imagen usando Haar Cascade.
    Devuelve el rostro procesado, la imagen original y las coordenadas del rostro.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: No se pudo cargar la imagen desde la ruta: {image_path}")
            return None, None, None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Usamos la versión RGB de la original
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Para detección
        
        faces = face_detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            print("Advertencia: No se detectaron rostros en la imagen.")
            return None, None, None

        main_face_coords = max(faces, key=lambda box: box[2] * box[3])
        x, y, w, h = main_face_coords
        
        # Recortar el rostro de la imagen original a color
        face_cropped_rgb = img_rgb[y:y+h, x:x+w]
        
        face_image_resized = Image.fromarray(face_cropped_rgb)
        face_image_resized = face_image_resized.resize(target_size)
        
        return np.array(face_image_resized), img_rgb, main_face_coords # Devolver también coords
    except Exception as e:
        print(f"Ocurrió un error durante el preprocesamiento: {e}")
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
# FUNCIÓN DE VISUALIZACIÓN
# ---------------------------------------------------------------------------
def display_results(original_img_rgb, face_coords, processed_face, heatmap_face, label, confidence):
    plt.figure(figsize=(18, 6)) # Aumentar el tamaño para 3 imágenes

    # 1. Imagen Original con Recuadro
    plt.subplot(1, 3, 1)
    img_with_box = original_img_rgb.copy()
    if face_coords is not None:
        x, y, w, h = face_coords
        cv2.rectangle(img_with_box, (x, y), (x+w, y+h), (0, 255, 0), 3) # Cuadro verde
    plt.imshow(img_with_box)
    plt.title("Imagen Original con Rostro Detectado")
    plt.axis('off')
    
    # 2. Rostro Recortado y Procesado
    plt.subplot(1, 3, 2)
    plt.imshow(processed_face)
    plt.title("Rostro Recortado (Input al Modelo)")
    plt.axis('off')
    
    # 3. Mapa de Calor (Explicabilidad)
    plt.subplot(1, 3, 3)
    plt.imshow(heatmap_face)
    plt.title("Explicabilidad (Grad-CAM)")
    plt.axis('off')
    
    plt.suptitle(f"Resultado: {label} (Confianza: {confidence})", fontsize=18, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

# ---------------------------------------------------------------------------
# ESTRATEGIA DE CONTROL (FLUJO DE OPERACIÓN)
# ---------------------------------------------------------------------------
def run_detection_pipeline(image_path):
    print("-" * 50)
    print(f"1. Recepción de Solicitud para: {image_path}")
    
    print("\n2. [RF-05] Iniciando preprocesamiento con Haar Cascade...")
    # Ahora preprocess_face devuelve 3 cosas
    processed_face, original_img_rgb, face_coords = preprocess_face(image_path)
    
    if processed_face is None:
        print("Pipeline finalizado: No se pudo procesar un rostro válido.")
        return
    print("   ... Rostro aislado exitosamente.")
    
    print("\n3. [RF-02] Pasando el rostro al modelo de clasificación...")
    label, confidence = classify_face(processed_face, classification_model)
    print(f"   ... Predicción: {label} (Confianza: {confidence})")

    print("\n4. [RF-04] Generando explicación visual (Grad-CAM)...")
    heatmap_img = generate_grad_cam(processed_face, classification_model)
    print("   ... Mapa de calor generado.")

    print("\n5. Mostrando resultados combinados...")
    # Pasamos las 3 imágenes a la función de visualización
    display_results(original_img_rgb, face_coords, processed_face, heatmap_img, label, confidence)
    print("-" * 50)

# ---------------------------------------------------------------------------
# PUNTO DE ENTRADA PRINCIPAL DEL PROGRAMA
# ---------------------------------------------------------------------------
if __name__ == '__main__':
   
    path_de_la_imagen = 'foto.jpg'
    
    run_detection_pipeline(path_de_la_imagen)