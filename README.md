Detector de Deepfakes con XAI y Análisis Forense

Aplicación web para el análisis forense de imágenes. Detecta deepfakes usando un modelo CNN y proporciona evidencia visual (XAI, detección de rasgos) y metadatos (EXIF) para el análisis.

Construido con TensorFlow/Keras, OpenCV y Gradio.

Características:

Clasificación de Deepfakes: "Real" vs. "Fake".
IA Explicable (XAI): Mapa de calor Grad-CAM para mostrar por qué el modelo toma una decisión.
Detección de Rasgos: Identifica caras (frontales y de perfil), ojos y sonrisas.
Extractor de Metadatos: Lee datos EXIF y prompts de IA (PNGInfo) ocultos en la imagen.

Setup y Ejecución
Sigue estos pasos en tu terminal:

Clonar el repositorio:

Instalar dependencias:

Bash

pip install tensorflow opencv-python numpy pillow gradio
Descargar Modelos: Coloca los siguientes archivos en la misma carpeta que programa.py:

cnn_model.h5 ( modelo CNN entrenado para deepfakes)

haarcascade_eye.xml

haarcascade_smile.xml

haarcascade_profileface.xml

Ejecutar la aplicación:

Bash

python programa.py
Abrir en el navegador: Abre la URL local que aparece en tu terminal: http://127.0.0.1:7860